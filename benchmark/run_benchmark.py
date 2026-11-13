# =============================================================================
#  BioTool benchmark  |  Rumiii/Qwen3.8-27B-ScientificBM (GGUF)  |  1x Kaggle T4
# -----------------------------------------------------------------------------
#  Faithful to the official gxx27/BioTool pipeline (script/unified_evaluate.py,
#  script/concurrent_evaluate.py). This script ONLY replaces the inference step
#  (LLaMA-Factory -> llama.cpp GGUF); scoring uses the OFFICIAL functions:
#     EM  -> live BioTool API calls (BLAST queue / other queue)
#         -> MedCPT-Query-Encoder similarity -> compute_metrics (EM / AS / Score)
#  Prompt = the model's own HF chat template, non-thinking, system prompt and
#  per-example tools (gold + distractors) taken straight from BioTool_test.json.
#  Everything is saved in /kaggle/working/. Resumable if the session restarts.
#  Kaggle settings: Accelerator = GPU T4 x1 (single GPU), Internet = ON.
# =============================================================================
import os, sys, re, json, time, shutil, subprocess, threading, queue, gc
from pathlib import Path
from collections import Counter, defaultdict

# ------------------------------- CONFIG --------------------------------------
HF_TOKEN       = ""                       # optional (only if a repo is gated/private)
MODEL_TAG      = "Qwen3.8-27B-ScientificBM"
TOKENIZER_REPO = "Rumiii/Qwen3.8-27B-ScientificBM"            # chat template source
GGUF_REPO      = "mradermacher/Qwen3.8-27B-ScientificBM-GGUF"
GGUF_PREFIX    = "Qwen3.8-27B-ScientificBM"
QUANT_ORDER    = ["Q3_K_M", "Q3_K_S", "Q2_K"]  # mid-tier first; auto-fallback if it does not fit the T4
N_CTX          = 8192                     # context window for llama-server
MAX_NEW_TOKENS = 1536                     # cap for one tool call
MAX_SAMPLES    = 0                        # 0 = full test split (1,408). e.g. 30 = quick smoke test
BLAST_WORKERS  = 1                        # official design: 1 slow BLAST queue ...
OTHER_WORKERS  = 1                        # ... + 1 fast non-BLAST queue (NCBI rate limits)
SKIP_BLAST_LIVE = True                    # True = do NOT wait for slow live BLAST calls; report lower/upper bounds instead
PORT           = 8080
DEFAULT_SYSTEM = "You are a helpful assistant that can call tools. Use the provided tools when appropriate."

WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("./kaggle_working_local")
TMP  = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp/biotool_tmp")
REPO = WORK / "BioTool"                   # official repo clone (code + data/)
RES, ANA, REP, LOGS = WORK / "results", WORK / "analysis", WORK / "report", WORK / "logs"
for _d in (WORK, TMP, RES, ANA, REP, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
RUN_TAG = MODEL_TAG + (f"_n{MAX_SAMPLES}" if MAX_SAMPLES else "")
PRED_PATH, ANA_PATH, CKPT_PATH = RES / f"{RUN_TAG}.jsonl", ANA / f"{RUN_TAG}.json", ANA / f"{RUN_TAG}.ckpt.json"
INFO_PATH = REP / f"{RUN_TAG}_run_info.json"

# --------- Reference numbers copied from the BioTool paper (Tables 1 & 2) ----
# order: [NCBI, UniProt, Ensembl, Overall]
PAPER = {
 "GPT-5.1":            dict(g="Proprietary",     score=[15.5,78.7,72.8,55.4], em=[0.0,1.7,8.9,3.5],    as_=[16.6,97.9,79.4,64.4]),
 "GPT-5.1-Codex":      dict(g="Proprietary",     score=[14.8,80.5,74.5,56.3], em=[0.0,1.7,6.3,2.6],    as_=[16.2,98.3,80.5,64.7]),
 "Gemini 3 Pro":       dict(g="Proprietary",     score=[72.5,87.8,77.3,79.2], em=[3.2,2.1,16.3,7.1],   as_=[76.7,99.8,82.0,86.2]),
 "Claude 4.5 Sonnet":  dict(g="Proprietary",     score=[79.8,87.3,77.0,81.4], em=[3.6,1.1,15.0,6.5],   as_=[85.5,100.0,82.4,89.4]),
 "Llama3.1-8B (base)": dict(g="Base open-source",score=[28.4,77.5,58.8,54.8], em=[0.0,1.0,5.5,2.1],    as_=[33.2,96.7,68.9,66.1]),
 "Qwen3-8B (base)":    dict(g="Base open-source",score=[26.2,73.1,63.5,54.1], em=[0.5,1.2,10.0,3.8],   as_=[28.9,87.8,70.1,62.1]),
 "Qwen2.5-7B (base)":  dict(g="Base open-source",score=[48.0,79.0,65.3,64.0], em=[0.0,1.6,6.3,2.6],    as_=[54.6,97.6,74.1,75.4]),
 "Qwen3-4B (base)":    dict(g="Base open-source",score=[53.4,71.9,64.0,63.1], em=[1.3,1.7,7.8,3.6],    as_=[58.5,88.8,72.8,73.3]),
 "Llama3.1-8B (BioTool-FT)": dict(g="BioTool-FT",score=[91.1,88.9,93.7,91.2], em=[43.1,7.4,43.1,31.2], as_=[93.3,99.9,96.6,96.6]),
 "Qwen3-8B (BioTool-FT)":    dict(g="BioTool-FT",score=[89.5,84.8,83.9,86.1], em=[40.8,4.5,35.6,26.9], as_=[92.3,99.8,88.8,93.7]),
 "Qwen2.5-7B (BioTool-FT)":  dict(g="BioTool-FT",score=[91.8,87.6,88.0,89.2], em=[44.6,5.4,42.4,30.8], as_=[93.8,99.9,91.5,95.1]),
 "Qwen3-4B (BioTool-FT)":    dict(g="BioTool-FT",score=[93.7,91.0,96.3,93.6], em=[66.7,9.3,51.1,42.4], as_=[94.8,100.0,98.8,97.8]),
}
DBS = ["ncbi", "uniprot", "ensembl", "overall"]

ue = ce = None   # official modules (loaded after the repo is cloned)


# ------------------------------- helpers -------------------------------------
def sh(cmd, log=None, cwd=None, env=None, check=True):
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    if log:
        with open(log, "w") as f:
            r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env)
    if check and r.returncode != 0:
        tail = ""
        if log:
            tail = "\n".join(Path(log).read_text(errors="ignore").splitlines()[-40:])
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}\n{tail}")
    return r


def tail(path, n=40):
    try:
        return "\n".join(Path(path).read_text(errors="ignore").splitlines()[-n:])
    except Exception:
        return ""


def install_python_deps():
    sh([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5", "huggingface_hub", "datasets",
        "tqdm", "scikit-learn", "matplotlib", "pandas", "requests", "numpy", "jinja2"])


def setup_repo():
    if not (REPO / "script" / "unified_evaluate.py").exists():
        if REPO.exists():
            shutil.rmtree(REPO)
        sh(["git", "clone", "--depth", "1", "https://github.com/gxx27/BioTool.git", str(REPO)])
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="gxx27/BioTool", repo_type="dataset", local_dir=str(REPO / "data"),
                      token=HF_TOKEN or None)
    for f in ("BioTool_test.json", "tools.json", "function_mapping.json"):
        assert (REPO / "data" / f).exists(), f"missing {f} in {REPO/'data'}"


def load_official():
    global ue, ce
    import importlib
    for p in (str(REPO), str(REPO / "script")):
        if p not in sys.path:
            sys.path.insert(0, p)
    ue = importlib.import_module("unified_evaluate")
    ce = importlib.import_module("concurrent_evaluate")
    assert Path(ue.ROOT_DIR).resolve() == REPO.resolve(), "official ROOT_DIR mismatch"


def check_apis():
    import requests
    checks = {
        "NCBI E-utilities": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?retmode=json",
        "UniProt REST":     "https://rest.uniprot.org/uniprotkb/P04637.json",
        "Ensembl REST":     "https://rest.ensembl.org/info/ping?content-type=application/json",
        "NCBI BLAST":       "https://blast.ncbi.nlm.nih.gov/Blast.cgi",
    }
    bad = []
    for k, u in checks.items():
        try:
            ok = requests.get(u, timeout=30).status_code < 400
        except Exception as e:
            ok = False
        print(f"  [{'OK' if ok else 'FAIL'}] {k}")
        if not ok:
            bad.append(k)
    if bad:
        print("  WARNING: live APIs unreachable ->", bad, "\n  Kaggle: Settings > Internet must be ON. Scores need live API calls.")


def load_test_examples():
    test = json.load(open(REPO / "data" / "BioTool_test.json", encoding="utf-8"))
    try:  # cross-check with the loader the user asked for (non-fatal)
        from datasets import load_dataset
        ds = load_dataset("gxx27/BioTool")
        if "test" in ds:
            print(f"  load_dataset test rows = {len(ds['test'])} | BioTool_test.json rows = {len(test)}")
        else:
            print("  load_dataset splits:", list(ds.keys()), "(using BioTool_test.json, as the official scripts do)")
    except Exception as e:
        print("  (load_dataset cross-check skipped:", str(e)[:120], ")")
    if MAX_SAMPLES:
        test = test[:MAX_SAMPLES]
    elif len(test) != 1408:
        print(f"  WARNING: expected 1408 test samples, found {len(test)}")
    return test


# ------------------------ prompt building / parsing --------------------------
def parse_tools(ex):
    t = ex.get("tools")
    if isinstance(t, str):
        t = json.loads(t)
    return t if isinstance(t, list) else []


def build_messages(ex):
    sys_msg = user = None
    for turn in ex.get("conversations", []):
        role = (turn.get("from") or turn.get("role") or "").lower()
        val = turn.get("value") or turn.get("content") or ""
        if role == "system" and sys_msg is None:
            sys_msg = val
        elif role in ("human", "user") and user is None:
            user = val
    return [{"role": "system", "content": sys_msg or DEFAULT_SYSTEM}, {"role": "user", "content": user or ""}]


def gold_label(ex):  # identical to official openai_eval_func.build_gold_label
    parts = []
    for turn in ex.get("conversations", []):
        if (turn.get("from") or turn.get("role") or "").lower() == "function_call":
            val = (turn.get("value") or "").strip()
            if val:
                parts.append(val if "<tool_call>" in val else "<tool_call>\n" + val + "\n</tool_call>")
    return "\n".join(parts)


def render_prompt(tok, ex):
    return tok.apply_chat_template(build_messages(ex), tools=parse_tools(ex), add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)


_FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|\Z)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)(?=</parameter>|<parameter=|\Z)", re.S)


TYPE_PRIOR = {}   # (function, param) -> dominant JSON type of the GOLD value in BioTool_train.json


def build_type_prior():
    """The Qwen XML tool-call format prints 1 and "1" identically, so types must be restored when decoding.
    Schema types are used first; for params the schema omits (e.g. blast timeout) or declares as a union with
    string, fall back to the dominant gold type in the TRAIN split (no test leakage)."""
    global TYPE_PRIOR
    cnt = {}
    for ex in json.load(open(REPO / "data" / "BioTool_train.json", encoding="utf-8")):
        for turn in ex.get("conversations", []):
            if (turn.get("from") or turn.get("role") or "").lower() != "function_call":
                continue
            try:
                call = json.loads(turn.get("value") or "{}")
            except Exception:
                continue
            for k, v in (call.get("arguments") or {}).items():
                cnt.setdefault((call.get("name"), k), Counter())[type(v).__name__] += 1
    TYPE_PRIOR = {k: c.most_common(1)[0][0] for k, c in cnt.items()}
    print(f"  type prior built for {len(TYPE_PRIOR)} (function, param) pairs from the train split")


def _json_nonstring(s):
    try:
        j = json.loads(s)
    except Exception:
        return None
    return j if isinstance(j, (int, float, bool, list, dict)) else None


def coerce_value(v, spec, prior=None):
    typ = (spec or {}).get("type")
    types = typ if isinstance(typ, list) else [typ]
    if typ is not None and types == ["string"]:
        return v                                   # schema says: plain string
    unknown = typ is None or "string" in types      # not declared / union with string -> use training prior
    if unknown:
        if prior in ("int", "float", "bool", "list", "dict"):
            j = _json_nonstring(v.strip())
            if j is not None:
                return j
        return v
    s = v.strip()
    if "boolean" in types and s.lower() in ("true", "false"):
        return s.lower() == "true"
    if "integer" in types:
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
    if "number" in types:
        j = _json_nonstring(s)
        if isinstance(j, (int, float)) and not isinstance(j, bool):
            return j
    if "array" in types or "object" in types:
        j = _json_nonstring(s)
        if j is not None:
            return j
    return v


def to_official_call_text(raw, tools):
    """Model XML (<function=..><parameter=..>) -> '<tool_call>\\n{json}\\n</tool_call>' read by official extract_tool_call."""
    raw = raw or ""
    props = {}
    for t in tools or []:
        fn = t.get("function", {})
        props[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    m = _FUNC_RE.search(raw)
    if not m:  # model answered in JSON <tool_call> style (or nothing): hand it over unchanged
        out = raw.strip()
        if "<tool_call>" in out and "</tool_call>" not in out:
            out += "\n</tool_call>"
        return out
    name, body = m.group(1).strip(), m.group(2)
    args = {}
    for pm in _PARAM_RE.finditer(body):
        k, v = pm.group(1).strip(), pm.group(2)
        if v.startswith("\r\n"): v = v[2:]
        elif v.startswith("\n"): v = v[1:]
        if v.endswith("\r\n"): v = v[:-2]
        elif v.endswith("\n"): v = v[:-1]
        args[k] = coerce_value(v, props.get(name, {}).get(k), TYPE_PRIOR.get((name, k)))
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


# ------------------------------ llama.cpp ------------------------------------
def build_llama_server():
    binary = TMP / "llama.cpp" / "build" / "bin" / "llama-server"
    if binary.exists():
        return binary
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)
    if not nvcc:
        raise RuntimeError("nvcc not found - enable a GPU accelerator (T4) in the Kaggle notebook settings.")
    src = TMP / "llama.cpp"
    if not src.exists():
        sh(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(src)])
    env = os.environ.copy()
    env["PATH"] = str(Path(nvcc).parent) + ":" + env["PATH"]
    sh(["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=OFF", f"-DCMAKE_CUDA_COMPILER={nvcc}"], cwd=src, env=env, log=LOGS / "llama_cmake.log")
    sh(["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count() or 4), "--target", "llama-server"],
       cwd=src, env=env, log=LOGS / "llama_build.log")
    assert binary.exists(), "llama-server build failed"
    return binary


class LlamaServer:
    def __init__(self, binary, gguf, n_ctx=N_CTX, port=PORT):
        self.binary, self.gguf, self.n_ctx, self.port = Path(binary), gguf, n_ctx, port
        self.url, self.proc, self.log = f"http://127.0.0.1:{port}", None, LOGS / "llama_server.log"

    def start(self, timeout=1500):
        import requests
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"  # single T4 only
        env["LD_LIBRARY_PATH"] = f"{self.binary.parent}:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        cmd = [str(self.binary), "-m", str(self.gguf), "-ngl", "99", "-c", str(self.n_ctx), "-np", "1",
               "--host", "127.0.0.1", "--port", str(self.port), "-t", str(min(4, os.cpu_count() or 4))]
        self.proc = subprocess.Popen(cmd, stdout=open(self.log, "w"), stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}):\n{tail(self.log)}")
            try:
                if requests.get(self.url + "/health", timeout=5).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError("llama-server did not become ready:\n" + tail(self.log))

    def complete(self, prompt, n_predict):
        import requests
        body = {"prompt": prompt, "n_predict": int(n_predict), "temperature": 0.0, "top_k": 1, "seed": 0,
                "stop": ["</tool_call>"], "stream": False, "cache_prompt": True}
        last = None
        for attempt in range(3):
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server died:\n" + tail(self.log))
            try:
                r = requests.post(self.url + "/completion", json=body, timeout=3600)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
                time.sleep(2)
        raise RuntimeError(f"completion failed: {last}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except Exception:
                self.proc.kill()
        time.sleep(3)


def start_best_server(binary, probe_prompt):
    from huggingface_hub import hf_hub_download
    last_err = None
    for q in QUANT_ORDER:
        fname = f"{GGUF_PREFIX}.{q}.gguf"
        print(f"\n>>> Trying {fname}", flush=True)
        try:
            path = hf_hub_download(GGUF_REPO, fname, local_dir=str(TMP / "gguf"), token=HF_TOKEN or None)
            srv = LlamaServer(binary, path)
            srv.start()
            srv.complete(probe_prompt, 4)  # pre-flight with the LONGEST prompt: catches OOM before the long run
            print(f">>> {q} loaded on the T4 and survived the longest-prompt pre-flight.", flush=True)
            return srv, q, fname
        except Exception as e:
            last_err = e
            print(f">>> {q} failed: {str(e)[:600]}", flush=True)
            try:
                srv.stop()
            except Exception:
                pass
    raise RuntimeError(f"No GGUF variant fit the T4. Last error: {last_err}")


# ------------------- background scorer (official evaluate_single) ------------
class BackgroundEvaluator:
    """Runs the OFFICIAL concurrent_evaluate.evaluate_single on predictions while the GPU keeps generating.
    Same two-queue design as the official script (BLAST queue + fast queue). Checkpointed + resumable."""

    def __init__(self, n, ckpt, fm=None):
        self.n, self.ckpt = n, Path(ckpt)
        self.fm = fm or ue.load_function_mapping()
        self.results = [None] * n
        if self.ckpt.exists():
            try:
                old = json.load(open(self.ckpt, encoding="utf-8"))
                if isinstance(old, list) and len(old) == n:
                    self.results = old
            except Exception:
                pass
        self.preds = {}
        self.lock, self.since = threading.Lock(), 0
        self.qb, self.qo, self.workers = queue.Queue(), queue.Queue(), []
        for q, k in ((self.qb, 0 if SKIP_BLAST_LIVE else BLAST_WORKERS), (self.qo, OTHER_WORKERS)):
            for _ in range(k):
                t = threading.Thread(target=self._run, args=(q,), daemon=True)
                t.start()
                self.workers.append((q, t))

    def done_count(self):
        return sum(r is not None and r.get("error") != "blast_not_scored" for r in self.results)

    def submit(self, idx, pred):
        self.preds[idx] = pred
        r = self.results[idx]
        if r is not None and r.get("error") != "blast_not_scored":
            return
        if ce.is_blast_label(pred):
            if SKIP_BLAST_LIVE:
                return  # filled with a placeholder in finish()
            self.qb.put((idx, pred))
        else:
            self.qo.put((idx, pred))

    def _run(self, q):
        while True:
            item = q.get()
            if item is None:
                return
            idx, pred = item
            try:
                res = ce.evaluate_single(pred, self.fm)
            except Exception as e:  # never lose a sample to a scorer crash
                try: pc = ue.extract_tool_call(pred.get("predict", ""))
                except Exception: pc = None
                try: lc = ue.extract_tool_call(pred.get("label", ""))
                except Exception: lc = None
                res = {"prompt": pred.get("prompt", ""), "exact_match": False, "predict_call": pc, "label_call": lc,
                       "api_success": False, "predict_response": None, "label_response": None,
                       "similarity": None, "error": f"{type(e).__name__}: {e}"}
            with self.lock:
                self.results[idx] = res
                self.since += 1
                if self.since >= 50:
                    self._save()

    def _save(self):
        tmp = self.ckpt.with_suffix(".tmp")
        json.dump(self.results, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(tmp, self.ckpt)
        self.since = 0

    def finish(self):
        for q, _ in self.workers:
            q.put(None)
        last = -1
        while any(t.is_alive() for _, t in self.workers):
            d = self.done_count()
            if d != last:
                print(f"  waiting for live-API scoring: {d}/{self.n}", flush=True)
                last = d
            time.sleep(15)
        for i, r in enumerate(self.results):
            if r is None and SKIP_BLAST_LIVE and i in self.preds and ce.is_blast_label(self.preds[i]):
                pc, lc = ue.extract_tool_call(self.preds[i]["predict"]), ue.extract_tool_call(self.preds[i]["label"])
                em = bool(pc and lc and ue.normalize_function_call(pc) == ue.normalize_function_call(lc))
                self.results[i] = {"prompt": self.preds[i].get("prompt", ""), "exact_match": em, "predict_call": pc,
                                   "label_call": lc, "api_success": False, "predict_response": None,
                                   "label_response": None, "similarity": None, "error": "blast_not_scored"}
        with self.lock:
            self._save()
        missing = [i for i, r in enumerate(self.results) if r is None]
        assert not missing, f"{len(missing)} samples were never scored (first: {missing[:5]})"
        return self.results


def reconvert_predictions(examples, evaluator):
    """Re-derive every 'predict' from the saved raw_output with the corrected type decoding (no GPU).
    Samples whose parsed call changed are re-scored (their old results are dropped)."""
    if not PRED_PATH.exists():
        return
    recs = [json.loads(l) for l in open(PRED_PATH, encoding="utf-8") if l.strip()]
    changed = 0
    for i, rec in enumerate(recs[:len(examples)]):
        new = to_official_call_text(rec.get("raw_output", ""), parse_tools(examples[i]))
        a, b = ue.extract_tool_call(rec.get("predict", "")), ue.extract_tool_call(new)
        same = (a is None and b is None) or (a and b and ue.normalize_function_call(a) == ue.normalize_function_call(b))
        if not same:
            rec["predict"] = new
            evaluator.results[i] = None
            changed += 1
    if changed:
        shutil.copy(PRED_PATH, str(PRED_PATH) + ".bak_before_typefix")
        with open(PRED_PATH, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  type-fix: {changed} predictions changed and will be re-scored")


# ------------------------------- generation ----------------------------------
def generate_all(examples, tok, prompts, ptoks, server, evaluator):
    from tqdm.auto import tqdm
    done = []
    if PRED_PATH.exists():
        for line in open(PRED_PATH, encoding="utf-8"):
            try:
                done.append(json.loads(line))
            except Exception:
                break
    done = done[:len(examples)]
    print(f"  resuming from {len(done)}/{len(examples)} saved predictions")
    for i, rec in enumerate(done):
        evaluator.submit(i, rec)
    if len(done) >= len(examples):
        return
    with open(PRED_PATH, "w", encoding="utf-8") as f:  # rewrite only the clean, parsed prefix
        for rec in done:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        bar = tqdm(range(len(done), len(examples)), desc="generate", initial=len(done), total=len(examples))
        for i in range(len(done), len(examples)):
            ex, tools = examples[i], parse_tools(examples[i])
            user = build_messages(ex)[1]["content"]
            raw, meta, err = "", {}, None
            if ptoks[i] > N_CTX - 256:
                err = "context_overflow"
            else:
                t0 = time.time()
                meta = server.complete(prompts[i], min(MAX_NEW_TOKENS, N_CTX - ptoks[i] - 16))
                meta["_sec"] = time.time() - t0
                raw = meta.get("content", "")
            rec = {"prompt": user, "predict": to_official_call_text(raw, tools), "label": gold_label(ex),
                   "raw_output": raw, "prompt_tokens": ptoks[i], "gen_tokens": meta.get("tokens_predicted"),
                   "gen_seconds": meta.get("_sec"), "stop_type": meta.get("stop_type"), "error": err,
                   "tool_names": [t.get("function", {}).get("name") for t in tools]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            evaluator.submit(i, rec)
            bar.update(1)
            bar.set_postfix(scored=evaluator.done_count())
        bar.close()


# ------------------------------ analysis -------------------------------------
def error_taxonomy(results):
    n, c = len(results), Counter()
    for r in results:
        if r.get("exact_match"):
            continue
        p, l = r.get("predict_call"), r.get("label_call") or {}
        if not isinstance(p, dict):
            c["no_parseable_call"] += 1
            continue
        if p.get("name") != l.get("name"):
            c["wrong_tool"] += 1
            continue
        c["right_tool_not_exact"] += 1
        pa, la = p.get("arguments"), l.get("arguments")
        if not isinstance(pa, dict) or not isinstance(la, dict):
            c["malformed_arguments"] += 1
            continue
        if set(la) - set(pa): c["missing_params"] += 1
        if set(pa) - set(la): c["extra_params"] += 1
        if any(k in pa and pa[k] != la[k] for k in la): c["wrong_values"] += 1
    return {k: v / n * 100 for k, v in c.items()}, dict(c)


def per_tool_table(results):
    g = defaultdict(list)
    for r in results:
        lc = r.get("label_call")
        g[(lc or {}).get("name", "?")].append(r)
    rows = []
    for name, items in g.items():
        sc = [1.0 if r.get("exact_match") else (float(r["similarity"]) if r.get("api_success") and r.get("similarity") is not None else 0.0) for r in items]
        rows.append({"tool": name, "n": len(items),
                     "EM_%": 100 * sum(bool(r.get("exact_match")) for r in items) / len(items),
                     "score_%": 100 * sum(sc) / len(items)})
    return sorted(rows, key=lambda x: (x["score_%"], -x["n"]))


def speed_stats():
    p, g, s = [], [], []
    if PRED_PATH.exists():
        for line in open(PRED_PATH, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("prompt_tokens"): p.append(r["prompt_tokens"])
            if r.get("gen_tokens") is not None: g.append(r["gen_tokens"])
            if r.get("gen_seconds") is not None: s.append(r["gen_seconds"])
    avg = lambda a: sum(a) / len(a) if a else 0.0
    return {"avg_prompt_tokens": avg(p), "max_prompt_tokens": max(p) if p else 0, "avg_gen_tokens": avg(g),
            "avg_sec_per_sample": avg(s), "total_gen_hours": sum(s) / 3600}


def compare_with_paper(ours):
    """ours = official compute_metrics() dict. Returns table rows + who-was-outperformed lists."""
    o = {"score": [ours[d]["biotool_score"] for d in DBS], "em": [ours[d]["em"] for d in DBS], "as_": [ours[d]["as"] for d in DBS]}
    rows = [{"model": f"** {MODEL_TAG} (GGUF, ours) **", "group": "Ours", **{k: v for k, v in o.items()}}]
    for name, v in PAPER.items():
        rows.append({"model": name, "group": v["g"], "score": v["score"], "em": v["em"], "as_": v["as_"]})
    beat = {"score": [], "em": [], "as_": []}
    lose = {"score": [], "em": [], "as_": []}
    for name, v in PAPER.items():
        for k in beat:
            (beat if o[k][3] > v[k][3] else lose)[k].append(name)
    per_db = {d: [n for n, v in PAPER.items() if o["score"][i] > v["score"][i]] for i, d in enumerate(DBS)}
    return rows, o, beat, lose, per_db


def make_charts(ours, rows, o, tool_rows, tax_pct):
    import matplotlib
    try:
        get_ipython()  # noqa
    except NameError:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Ours": "#d62728", "Proprietary": "#7f7f7f", "Base open-source": "#aec7e8", "BioTool-FT": "#2ca02c"}
    paths = []

    def bars(ax, key, idx, title):
        data = sorted(((r["model"].replace("** ", "").replace(" **", ""), r[key][idx], r["group"]) for r in rows), key=lambda x: x[1])
        ax.barh([d[0] for d in data], [d[1] for d in data], color=[colors[d[2]] for d in data])
        for y, d in enumerate(data):
            ax.text(d[1] + 0.5, y, f"{d[1]:.1f}", va="center", fontsize=7)
        ax.set_title(title, fontsize=10); ax.set_xlim(0, 112); ax.tick_params(axis="y", labelsize=7)

    fig, axs = plt.subplots(1, 4, figsize=(26, 7))
    for ax, i, t in zip(axs, range(4), ["NCBI", "UniProt", "Ensembl", "Overall"]):
        bars(ax, "score", i, f"BioTool Score (%) - {t}")
    plt.tight_layout(); p = REP / "01_biotool_score_by_database.png"; plt.savefig(p, dpi=140); paths.append(p); plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    bars(axs[0], "em", 3, "Overall Exact Match (%)"); bars(axs[1], "as_", 3, "Overall API Success (%)")
    plt.tight_layout(); p = REP / "02_overall_em_and_api_success.png"; plt.savefig(p, dpi=140); paths.append(p); plt.show()

    worst = [r for r in tool_rows if r["n"] >= 3][:20]
    if worst:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh([f"{r['tool']} (n={r['n']})" for r in worst][::-1], [r["score_%"] for r in worst][::-1], color="#d62728")
        ax.set_title("Weakest 20 tools by BioTool Score (tools with n>=3)"); ax.set_xlabel("BioTool Score (%)")
        plt.tight_layout(); p = REP / "03_weakest_tools.png"; plt.savefig(p, dpi=140); paths.append(p); plt.show()

    if tax_pct:
        fig, ax = plt.subplots(figsize=(9, 5)); ks = sorted(tax_pct, key=tax_pct.get)
        ax.barh(ks, [tax_pct[k] for k in ks], color="#ff7f0e")
        ax.set_title("Error taxonomy (% of ALL test samples; param errors are non-exclusive)")
        plt.tight_layout(); p = REP / "04_error_taxonomy.png"; plt.savefig(p, dpi=140); paths.append(p); plt.show()
    return paths


def write_report(ours, info, results, speed):
    rows, o, beat, lose, per_db = compare_with_paper(ours)
    tool_rows = per_tool_table(results)
    tax_pct, tax_cnt = error_taxonomy(results)
    n = len(results)
    api_fail_label = sum(1 for r in results if r.get("label_call") and not r.get("exact_match") and r.get("label_response") is None and r.get("error") != "blast_not_scored")
    charts = make_charts(ours, rows, o, tool_rows, tax_pct)

    import csv
    with open(REP / f"{RUN_TAG}_comparison.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "group"] + [f"{m}_{d}" for m in ("score", "em", "as") for d in DBS])
        for r in rows:
            w.writerow([r["model"], r["group"]] + [round(x, 2) for k in ("score", "em", "as_") for x in r[k]])
    with open(REP / f"{RUN_TAG}_per_tool.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tool", "n", "EM_%", "score_%"]); w.writeheader(); w.writerows(tool_rows)
    json.dump({"ours": ours, "run_info": info, "speed": speed, "error_taxonomy_pct": tax_pct,
               "error_taxonomy_count": tax_cnt, "outperformed_overall": beat, "not_outperformed_overall": lose,
               "outperformed_per_db_score": per_db}, open(REP / f"{RUN_TAG}_metrics.json", "w"), indent=2)

    rank = sorted([(r["model"], r["score"][3]) for r in rows], key=lambda x: -x[1])
    pos = [i for i, (m, _) in enumerate(rank, 1) if "ours" in m][0]
    L = [f"# BioTool benchmark - {MODEL_TAG}", "",
         f"- Quantization used: **{info.get('quant')}** (`{info.get('gguf')}`), ctx {N_CTX}, greedy decoding, non-thinking, 1x T4",
         f"- Samples scored: **{n}** | live-API scoring exactly as the official repo | MedCPT-Query-Encoder similarity", "",
         "## Headline (official metric definitions)", "",
         "| Metric (%) | NCBI | UniProt | Ensembl | Overall |", "|---|---|---|---|---|"]
    for name, key in (("Exact Match", "em"), ("API Success", "as"), ("BioTool Score", "biotool_score")):
        L.append(f"| {name} | " + " | ".join(f"{ours[d][key]:.1f}" for d in DBS) + " |")
    L.append("| Sample count | " + " | ".join(str(ours[d]["count"]) for d in DBS) + " |")
    pend = [r for r in results if r.get("error") == "blast_not_scored"]
    if pend:
        up = [dict(r, exact_match=True) if r.get("error") == "blast_not_scored" and not r.get("exact_match") else r for r in results]
        ou = ue.compute_metrics(up, ue.load_db_mapping())
        L += ["", f"**BLAST live scoring SKIPPED for {len(pend)} samples.** Table above = LOWER bound (those samples count 0 unless exact match).",
              "UPPER bound (all of them scored 1.0): " + " | ".join(f"{d}: {ou[d]['biotool_score']:.1f}" for d in DBS),
              "LOWER bound: " + " | ".join(f"{d}: {ours[d]['biotool_score']:.1f}" for d in DBS),
              "The true score lies between them; the ranking below uses the lower bound."]
    L += ["", f"## Rank on overall BioTool Score: **#{pos} of {len(rank)}** (paper's 12 reference models + yours)", ""]
    L += [f"{i}. {m} - {s:.1f}" for i, (m, s) in enumerate(rank, 1)]
    L += ["", "## Which models your model outperformed (overall)", ""]
    for key, nm in (("score", "BioTool Score"), ("em", "Exact Match"), ("as_", "API Success")):
        L.append(f"- **{nm}** - beats {len(beat[key])}/12: {', '.join(beat[key]) or 'none'}")
        L.append(f"  - does not beat: {', '.join(lose[key]) or 'none'}")
    L += ["", "## Per-database BioTool Score wins", ""]
    for d in DBS:
        L.append(f"- {d.upper()}: beats {len(per_db[d])}/12 -> {', '.join(per_db[d]) or 'none'}")
    L += ["", "## Error analysis (% of all test samples)", ""] + [f"- {k}: {v:.1f}% ({tax_cnt[k]})" for k, v in sorted(tax_pct.items(), key=lambda x: -x[1])]
    L += [f"- gold (label) API call failed live (not EM): {api_fail_label} samples - these cannot earn similarity credit (official behaviour)",
          "", "## Speed", "",
          f"- avg prompt {speed['avg_prompt_tokens']:.0f} tok (max {speed['max_prompt_tokens']}), avg output {speed['avg_gen_tokens']:.0f} tok, "
          f"{speed['avg_sec_per_sample']:.1f}s/sample, {speed['total_gen_hours']:.2f} GPU-hours total", "",
         "## Read this before comparing with the paper", "",
         "- Paper numbers are Table 1/2 means over 3 runs; this is ONE greedy run of a 3-bit GGUF quant (small quality loss vs 16-bit).",
         "- Paper BioTool Score/EM/AS use the same definitions as the official code used here; paper AS = similarity>0, repo AS = EM or both calls succeed (near-identical).",
         "- Live APIs change over time, so scores can drift a little from the paper even for identical predictions.",
         "- Charts: " + ", ".join(p.name for p in charts)]
    (REP / f"{RUN_TAG}_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# --------------------------------- main --------------------------------------
def main():
    print("== 1/8 python deps"); install_python_deps()
    print("== 2/8 official repo + data"); setup_repo(); load_official()
    print("== 3/8 live API reachability"); check_apis()
    print("== 4/8 test split + prompts")
    from transformers import AutoTokenizer
    examples = load_test_examples()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, token=HF_TOKEN or None)
    prompts = [render_prompt(tok, ex) for ex in examples]
    names0 = parse_tools(examples[0])[0]["function"]["name"]
    assert names0 in prompts[0] and "<function=" in prompts[0], "chat template did not inject the tool schema/format"
    print("  prompt tail:", repr(prompts[0][-120:]))
    ptoks = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    over = sum(t > N_CTX - 256 for t in ptoks)
    print(f"  prompt tokens: avg {sum(ptoks)/len(ptoks):.0f}, max {max(ptoks)}, > ctx budget: {over}")

    build_type_prior()
    evaluator = BackgroundEvaluator(len(examples), CKPT_PATH)
    reconvert_predictions(examples, evaluator)
    info = json.load(open(INFO_PATH)) if INFO_PATH.exists() else {}
    server = None
    if not (PRED_PATH.exists() and sum(1 for _ in open(PRED_PATH, encoding="utf-8")) >= len(examples)):
        print("== 5/8 build llama.cpp (CUDA, sm_75) + load GGUF")
        binary = build_llama_server()
        fits = [i for i, t in enumerate(ptoks) if t <= N_CTX - 256]
        longest = max(fits, key=lambda i: ptoks[i])
        server, q, fname = start_best_server(binary, prompts[longest])
        info.update({"quant": q, "gguf": fname, "n_ctx": N_CTX})
        json.dump(info, open(INFO_PATH, "w"), indent=2)
    print("== 6/8 generation (+ live-API scoring in the background)")
    try:
        generate_all(examples, tok, prompts, ptoks, server, evaluator)
    finally:
        if server:
            server.stop()
    results = evaluator.finish()
    json.dump(results, open(ANA_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("== 7/8 MedCPT similarity (official)")
    gc.collect()
    ue.run_similarity_phase2([str(ANA_PATH)])
    results = json.load(open(ANA_PATH, encoding="utf-8"))
    print("== 8/8 metrics + comparison")
    ours = ue.compute_metrics(results, ue.load_db_mapping())
    ue.print_metrics(ours, f"{RUN_TAG} (official compute_metrics)")
    write_report(ours, info, results, speed_stats())
    print(f"\nDONE. Everything is in {WORK}\n  predictions: {PRED_PATH}\n  analysis:    {ANA_PATH}\n  report:      {REP}")


if __name__ == "__main__":
    main()
