# =============================================================================
#  BioTool RE-VALIDATION  |  Rumiii/Qwen3.8-27B-ScientificBM  |  Kaggle (T4 or CPU)
# -----------------------------------------------------------------------------
#  Re-checks ONLY the samples that need real validation (about 150 of 1,408) and
#  re-uses every other result from your first run. No model, no llama.cpp, no GPU
#  generation. Uses the OFFICIAL BioTool scoring functions.
#
#  Re-checked samples:
#    1. predictions changed by the corrected value-type decoding (e.g. "120.0" -> 120.0)
#    2. all BLAST samples (were skipped last time)        [time-boxed, see BLAST_BUDGET_MIN]
#    3. samples whose GOLD call failed live (not the model's fault - retried once)
#    4. samples whose PREDICTED call failed live while the gold call worked
#    5. the two tools that scored exactly 0 (get_uniparc_by_upi, overlap_by_id)
#
#  FINISHING RUN: in the SAME Kaggle session as before it reuses /kaggle/working/revalidation/rescored_ckpt.json and
#  only re-runs the unscored BLAST samples (3 BLAST workers, 180 min budget). In a NEW session it redoes everything.
#  NOTHING TO UPLOAD: the model's saved answers and the first-run scores are embedded at the
#  bottom of this file (_PAYLOAD). Settings: Internet ON. GPU optional (only speeds up MedCPT).
#  Paste this whole file into one cell and run. Everything is saved in /kaggle/working/revalidation/
# =============================================================================
import os, sys, re, json, time, shutil, subprocess, threading, queue, gc
from pathlib import Path
from collections import Counter, defaultdict

# ------------------------------- CONFIG --------------------------------------
HF_TOKEN        = ""
MODEL_TAG       = "Qwen3.8-27B-ScientificBM"
API_WAIT_MIN    = 10      # if a live API is unreachable at start, keep retrying this many minutes, then stop
BLAST_WORKERS   = 3       # parallel BLAST jobs (NCBI tolerates a few; more can get throttled)
OTHER_WORKERS   = 1       # non-BLAST live calls (NCBI/UniProt/Ensembl rate limits)
BLAST_TIMEOUT_S = 1200    # give up on ONE BLAST sample after this many seconds
BLAST_BUDGET_MIN = 180    # stop starting new BLAST samples after this many minutes (rest -> lower/upper bound)
WEAK_TOOLS      = ["get_uniparc_by_upi", "overlap_by_id"]
LIST_RULE       = True    # text that is a JSON list/object (e.g. ["gene"]) becomes a real list/object
QUICK_TEST_N    = 0       # >0 = smoke test: re-check only the first N flagged samples (metrics then NOT final)
MAX_SAMPLES     = 0       # unused here (kept for the shared helper functions)
DEFAULT_SYSTEM  = "You are a helpful assistant that can call tools. Use the provided tools when appropriate."

WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("./kaggle_working_local")
REPO = WORK / "BioTool"
OUTD = WORK / "revalidation"
for _d in (WORK, OUTD):
    _d.mkdir(parents=True, exist_ok=True)
CKPT = OUTD / "rescored_ckpt.json"

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
NONE_SEEN = set() # (function, param) pairs whose GOLD value is JSON null somewhere in BioTool_train.json


def build_type_prior():
    """The Qwen XML tool-call format prints 1 and "1" identically (and null and "null"), so types must be restored
    when decoding. Schema types are used first; for params the schema omits (e.g. blast timeout) or declares as a
    union with string, fall back to the dominant gold type in the TRAIN split (no test leakage).
    The text `null` becomes a real JSON null only for params whose gold value is null somewhere in the TRAIN split."""
    global TYPE_PRIOR, NONE_SEEN
    cnt, none_seen = {}, set()
    for ex in json.load(open(REPO / "data" / "BioTool_train.json", encoding="utf-8")):
        for turn in ex.get("conversations", []):
            if (turn.get("from") or turn.get("role") or "").lower() != "function_call":
                continue
            try:
                call = json.loads(turn.get("value") or "{}")
            except Exception:
                continue
            for k, v in (call.get("arguments") or {}).items():
                if v is None:
                    none_seen.add((call.get("name"), k))
                else:
                    cnt.setdefault((call.get("name"), k), Counter())[type(v).__name__] += 1
    TYPE_PRIOR = {k: c.most_common(1)[0][0] for k, c in cnt.items()}
    NONE_SEEN = none_seen
    print(f"  type prior built for {len(TYPE_PRIOR)} (function, param) pairs; {len(NONE_SEEN)} pairs can be null")


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
        if LIST_RULE:
            j = _json_nonstring(v.strip())
            if isinstance(j, (list, dict)):
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
        if v.strip() == "null" and (name, k) in NONE_SEEN:
            args[k] = None
            continue
        args[k] = coerce_value(v, props.get(name, {}).get(k), TYPE_PRIOR.get((name, k)))
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"



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



ue = ce = None   # official modules (loaded after the repo is cloned)


# ------------------------------ new logic ------------------------------------
def same_call(a, b):
    return (a is None and b is None) or bool(a and b and ue.normalize_function_call(a) == ue.normalize_function_call(b))


def lfail(r):
    return r["_lf"] if "_lf" in r else r.get("label_response") is None


def pfail(r):
    return r["_pf"] if "_pf" in r else r.get("predict_response") is None


def install_python_deps():   # minimal: do NOT upgrade numpy/pandas (it breaks Kaggle's preinstalled stack)
    sh([sys.executable, "-m", "pip", "install", "-q", "transformers>=5", "huggingface_hub", "datasets", "tqdm", "scikit-learn"])


def wait_for_apis():
    import requests
    checks = {"NCBI E-utilities": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?retmode=json",
              "UniProt REST": "https://rest.uniprot.org/uniprotkb/P04637.json",
              "Ensembl REST": "https://rest.ensembl.org/info/ping?content-type=application/json",
              "NCBI BLAST": "https://blast.ncbi.nlm.nih.gov/Blast.cgi"}
    t0 = time.time()
    while True:
        bad = []
        for k, u in checks.items():
            try:
                ok = requests.get(u, timeout=60).status_code < 400
            except Exception:
                ok = False
            print(f"  [{'OK' if ok else 'FAIL'}] {k}", flush=True)
            if not ok:
                bad.append(k)
        if not bad:
            return
        if time.time() - t0 > API_WAIT_MIN * 60:
            raise RuntimeError(f"{bad} unreachable for {API_WAIT_MIN} min. Scoring now would wrongly count those samples as "
                               "failures. Re-run later (or check Settings > Internet is ON).")
        print("  waiting 60s and retrying...", flush=True)
        time.sleep(60)


def load_payload():
    import base64, zlib
    d = json.loads(zlib.decompress(base64.b64decode(_PAYLOAD.replace("\n", ""))))
    return {int(k): v for k, v in d["raw"].items()}, d["old"]


def build_inputs(examples, raw, comp):
    recs, old = [], []
    for i, ex in enumerate(examples):
        label, user = gold_label(ex), build_messages(ex)[1]["content"]
        recs.append({"prompt": user, "label": label, "predict": "", "raw_output": raw.get(i)})
        em, api, sim, lf, pf, bu = comp[i]
        old.append({"prompt": user, "exact_match": bool(em), "api_success": bool(api), "similarity": sim,
                    "label_call": ue.extract_tool_call(label), "predict_call": None, "label_response": None,
                    "predict_response": None, "_lf": bool(lf), "_pf": bool(pf), "error": "blast_not_scored" if bu else None})
    return recs, old


def reconvert(recs, examples):
    """Decode the saved raw answers with the corrected typing. Returns (samples carrying typed values,
    samples whose call changed because of the list/object rule -> their cached live result must be redone)."""
    global LIST_RULE
    changed, force = [], set()
    for i, rec in enumerate(recs):
        if rec["raw_output"] is None:
            continue
        tools = parse_tools(examples[i])
        LIST_RULE = False
        before = to_official_call_text(rec["raw_output"], tools)
        LIST_RULE = True
        rec["predict"] = to_official_call_text(rec["raw_output"], tools)
        if before != rec["predict"]:
            force.add(i)
        call = ue.extract_tool_call(rec["predict"]) or {}
        if any(not isinstance(v, str) for v in (call.get("arguments") or {}).values()):
            changed.append(i)
    return changed, force


def select_todo(recs, old, changed):
    cats = {k: set() for k in ("typed_values_redecoded", "blast", "gold_call_failed", "pred_call_failed", "weak_tools", "not_scored")}
    cats["typed_values_redecoded"] = set(changed)
    for i, (rec, r) in enumerate(zip(recs, old)):
        lc = ue.extract_tool_call(rec["label"]) or {}
        if lc.get("name") == "blast":
            cats["blast"].add(i)
        if lc.get("name") in WEAK_TOOLS:
            cats["weak_tools"].add(i)
        if r is None:
            cats["not_scored"].add(i); continue
        if not r.get("exact_match") and r.get("error") != "blast_not_scored":
            if lfail(r):
                cats["gold_call_failed"].add(i)
            elif pfail(r):
                cats["pred_call_failed"].add(i)
    todo = sorted(set().union(*cats.values()))
    if QUICK_TEST_N:
        todo = todo[:QUICK_TEST_N]
    return todo, cats


def placeholder(rec, why):
    pc, lc = ue.extract_tool_call(rec["predict"]), ue.extract_tool_call(rec["label"])
    return {"prompt": rec.get("prompt", ""), "exact_match": same_call(pc, lc) and pc is not None, "predict_call": pc,
            "label_call": lc, "api_success": False, "predict_response": None, "label_response": None,
            "similarity": None, "error": "blast_not_scored", "note": why}


class Rescorer:
    def __init__(self, todo, recs, fm, force=()):
        self.todo, self.recs, self.fm = todo, recs, fm
        self.out, self.lock = {}, threading.Lock()
        if CKPT.exists():
            try:
                self.out = {int(k): v for k, v in json.load(open(CKPT, encoding="utf-8")).items()}
            except Exception:
                self.out = {}
        # keep cached live results, but RE-RUN unscored BLAST placeholders and samples whose call changed
        self.out = {i: v for i, v in self.out.items()
                    if i in set(todo) and v.get("error") != "blast_not_scored" and i not in set(force)}
        print(f"  cached live results reused: {len(self.out)} | still to run: {len(todo) - len(self.out)}")
        self.deadline = time.time() + BLAST_BUDGET_MIN * 60
        self.qb, self.qo = queue.Queue(), queue.Queue()
        for i in todo:
            if i in self.out:
                continue
            (self.qb if ce.is_blast_label(recs[i]) else self.qo).put(i)

    def _score(self, rec, limit):
        box = {}
        def run():
            try:
                box["r"] = ce.evaluate_single(rec, self.fm)
            except Exception as e:
                box["r"] = {"prompt": rec.get("prompt", ""), "exact_match": False,
                            "predict_call": ue.extract_tool_call(rec["predict"]), "label_call": ue.extract_tool_call(rec["label"]),
                            "api_success": False, "predict_response": None, "label_response": None, "similarity": None,
                            "error": f"{type(e).__name__}: {e}"}
        t = threading.Thread(target=run, daemon=True); t.start(); t.join(limit)
        return box.get("r")

    def _worker(self, q, is_blast):
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            rec = self.recs[i]
            if is_blast and time.time() > self.deadline:
                res = placeholder(rec, "blast time budget used up")
            else:
                res = self._score(rec, BLAST_TIMEOUT_S if is_blast else None)
                if res is None:
                    res = placeholder(rec, f"blast timeout > {BLAST_TIMEOUT_S}s")
            with self.lock:
                self.out[i] = res
                if len(self.out) % 20 == 0:
                    self._save()

    def _save(self):
        tmp = CKPT.with_suffix(".tmp")
        json.dump({str(k): v for k, v in self.out.items()}, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(tmp, CKPT)

    def run(self):
        ths = [threading.Thread(target=self._worker, args=(self.qb, True), daemon=True) for _ in range(BLAST_WORKERS)]
        ths += [threading.Thread(target=self._worker, args=(self.qo, False), daemon=True) for _ in range(OTHER_WORKERS)]
        for t in ths:
            t.start()
        t0, last = time.time(), -1
        while any(t.is_alive() for t in ths):
            if len(self.out) != last:
                last = len(self.out)
                print(f"  re-checked {last}/{len(self.todo)} | {int(time.time()-t0)//60} min | "
                      f"BLAST left {self.qb.qsize()}, other left {self.qo.qsize()}", flush=True)
            time.sleep(10)
        with self.lock:
            self._save()
        assert all(i in self.out for i in self.todo), "some flagged samples were not re-checked"
        return self.out


def sample_score(r):
    if r is None:
        return 0.0
    if r.get("exact_match"):
        return 1.0
    return float(r["similarity"]) if r.get("api_success") and r.get("similarity") is not None else 0.0


def pct(x):
    return f"{x:.1f}"


def discrepancy_checks(recs, examples, old, new, todo):
    L = ["## Extra checks (things that could silently distort the benchmark)", ""]
    # 1. leftover type-only mismatches
    left = Counter()
    for r in new:
        p, l = r.get("predict_call") or {}, r.get("label_call") or {}
        if p.get("name") == l.get("name"):
            for k, v in (l.get("arguments") or {}).items():
                pv = (p.get("arguments") or {}).get(k)
                if isinstance(pv, str) and not isinstance(v, str) and pv != v:
                    try:
                        if json.loads(pv) == v:
                            left[(l["name"], k)] += 1
                    except Exception:
                        pass
    L.append(f"1. **Type-only mismatches still left after the fix:** {sum(left.values())}"
             + (" -> " + ", ".join(f"{a}.{b} x{c}" for (a, b), c in left.most_common(5)) if left else " (none)"))
    # 2. gold-call failures
    gf = Counter((r.get("label_call") or {}).get("name") for r in new
                 if not r.get("exact_match") and lfail(r) and r.get("error") != "blast_not_scored")
    L.append(f"2. **Samples where the REFERENCE call itself fails live (not the model's fault):** {sum(gf.values())}"
             + (" -> " + ", ".join(f"{k} x{v}" for k, v in gf.most_common(6)) if gf else " (none)"))
    # 3. predicted-call failures
    pf = Counter((r.get("label_call") or {}).get("name") for r in new
                 if not r.get("exact_match") and pfail(r) and not lfail(r))
    L.append(f"3. **Samples where only the model's call fails live:** {sum(pf.values())}"
             + (" -> " + ", ".join(f"{k} x{v}" for k, v in pf.most_common(6)) if pf else " (none)"))
    # 4. zero-score tools
    by = defaultdict(list)
    for r in new:
        by[(r.get("label_call") or {}).get("name")].append(sample_score(r))
    zero = [(k, len(v)) for k, v in by.items() if len(v) >= 3 and sum(v) == 0]
    L.append("4. **Tools that still score exactly 0:** " + (", ".join(f"{k} (n={n})" for k, n in zero) if zero else "none"))
    # 5. arguments not in the schema / placeholder values
    extra, nulls = Counter(), 0
    for rec, ex in zip(recs, examples):
        pc = ue.extract_tool_call(rec["predict"]) or {}
        props = {}
        for t in parse_tools(ex):
            if t.get("function", {}).get("name") == pc.get("name"):
                props = (t["function"].get("parameters") or {}).get("properties") or {}
        for k, v in (pc.get("arguments") or {}).items():
            if props and k not in props:
                extra[(pc.get("name"), k)] += 1
            if v in ("null", "None", "NULL"):
                nulls += 1
    L.append(f"5. **Arguments the model added that are not in the tool definition:** {sum(extra.values())}"
             + (" -> " + ", ".join(f"{a}.{b} x{c}" for (a, b), c in extra.most_common(5)) if extra else "")
             + f" | placeholder values ('null'): {nulls}")
    # 6. biggest movers
    mv = sorted(((sample_score(new[i]) - sample_score(old[i]) if old[i] else sample_score(new[i]), i) for i in todo), key=lambda x: x[0])
    ups = [(d, i) for d, i in mv[::-1][:5] if d > 0]
    dns = [(d, i) for d, i in mv[:5] if d < 0]
    nm = lambda i: (new[i].get("label_call") or {}).get("name")
    L.append("6. **Biggest score gains from re-checking:** " + (", ".join(f"{nm(i)} +{d:.2f}" for d, i in ups) or "none"))
    L.append("   **Biggest score drops:** " + (", ".join(f"{nm(i)} {d:.2f}" for d, i in dns) or "none"))
    return L


def make_ranking_chart(rows, o):
    import matplotlib
    try:
        get_ipython()  # noqa
    except NameError:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Ours": "#F26419", "BioTool-FT": "#33658A", "Proprietary": "#6D7587", "Base open-source": "#B9C0CE"}
    data = sorted(((r["model"].replace("** ", "").replace(" **", ""), r["score"][3], r["group"]) for r in rows), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([d[0] for d in data], [d[1] for d in data], color=[colors[d[2]] for d in data])
    for y, d in enumerate(data):
        ax.text(d[1] + 0.5, y, f"{d[1]:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, 105); ax.set_xlabel("Overall BioTool Score"); ax.set_title(f"{MODEL_TAG} after re-validation")
    plt.tight_layout(); p = OUTD / "ranking_after_revalidation.png"; plt.savefig(p, dpi=170); plt.close(fig)
    return p


def main():
    print("== 1/7 deps + official repo + data"); install_python_deps(); setup_repo(); load_official()
    print("== 2/7 live API reachability"); wait_for_apis()
    print("== 3/7 embedded first-run data + test split")
    examples = load_test_examples()
    n = len(examples)
    raw, comp = load_payload()
    assert len(comp) == n, f"embedded data has {len(comp)} samples but the test split has {n}"
    recs, old = build_inputs(examples, raw, comp)
    print(f"  {n} samples, {len(raw)} saved model answers available for re-check")
    print("== 4/7 corrected value-type decoding (typed values and null)")
    build_type_prior()
    changed, force = reconvert(recs, examples)
    print(f"  {len(changed)} predictions carry typed values and will be re-checked ({len(force)} changed by the list rule)")
    todo, cats = select_todo(recs, old, changed)
    assert all(recs[i]["raw_output"] is not None for i in todo), "a flagged sample has no saved answer"
    print("  flagged for re-check:", {k: len(v) for k, v in cats.items()}, "| total unique:", len(todo))
    print("== 5/7 live re-validation (official evaluate_single)")
    out = Rescorer(todo, recs, ue.load_function_mapping(), force).run()
    new = list(old)
    for i, r in out.items():
        new[i] = r
    assert all(r is not None for r in new), "unscored samples remain"
    with open(OUTD / f"{MODEL_TAG}.fixed_predictions.jsonl", "w", encoding="utf-8") as f:
        for rec in recs:
            if rec["raw_output"] is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    merged = OUTD / f"{MODEL_TAG}.revalidated.json"
    json.dump(new, open(merged, "w", encoding="utf-8"), ensure_ascii=False)
    print("== 6/7 MedCPT similarity (official, only where missing)")
    gc.collect()
    ue.run_similarity_phase2([str(merged)])
    new = json.load(open(merged, encoding="utf-8"))
    print("== 7/7 metrics + comparison")
    dbm = ue.load_db_mapping()
    before = ue.compute_metrics([r for r in old if r], dbm)
    after = ue.compute_metrics(new, dbm)
    ue.print_metrics(before, "BEFORE (first run)")
    ue.print_metrics(after, "AFTER (re-validated)" + ("  [SMOKE TEST - NOT FINAL]" if QUICK_TEST_N else ""))
    rows, o, beat, lose, per_db = compare_with_paper(after)
    pend = [r for r in new if r.get("error") == "blast_not_scored"]
    L = [f"# Re-validation - {MODEL_TAG}", "",
         f"- Re-checked **{len(todo)}** of {n} samples; the other {n-len(todo)} reuse the first-run results.",
         "- Flagged: " + ", ".join(f"{k} {len(v)}" for k, v in cats.items()),
         "- **SMOKE TEST ONLY - numbers are not final**" if QUICK_TEST_N else "", "",
         "## Before vs after (%)", "", "| Metric | | NCBI | UniProt | Ensembl | Overall |", "|---|---|---|---|---|---|"]
    for nm, key in (("Exact Match", "em"), ("API Success", "as"), ("BioTool Score", "biotool_score")):
        L.append(f"| {nm} | before | " + " | ".join(pct(before[d][key]) for d in DBS) + " |")
        L.append(f"| | **after** | " + " | ".join("**" + pct(after[d][key]) + "**" for d in DBS) + " |")
    if pend:
        up = [dict(r, exact_match=True) if r.get("error") == "blast_not_scored" and not r.get("exact_match") else r for r in new]
        ou = ue.compute_metrics(up, dbm)
        L += ["", f"**{len(pend)} BLAST samples could not be scored in time** (counted as 0). "
                  f"BioTool Score is therefore between **{pct(after['overall']['biotool_score'])}** and **{pct(ou['overall']['biotool_score'])}** overall "
                  f"(NCBI {pct(after['ncbi']['biotool_score'])} - {pct(ou['ncbi']['biotool_score'])})."]
    else:
        L += ["", "All BLAST samples were scored live, so the numbers above are final (no lower/upper bound)."]
    rank = sorted([(r["model"], r["score"][3]) for r in rows], key=lambda x: -x[1])
    pos = [i for i, (m, _) in enumerate(rank, 1) if "ours" in m][0]
    L += ["", f"## Rank on overall BioTool Score: **#{pos} of {len(rank)}**", ""] + [f"{i}. {m} - {s:.1f}" for i, (m, s) in enumerate(rank, 1)]
    for key, nm in (("score", "BioTool Score"), ("em", "Exact Match"), ("as_", "API Success")):
        L.append(f"- **{nm}**: beats {len(beat[key])}/12 -> {', '.join(beat[key]) or 'none'}; does not beat: {', '.join(lose[key]) or 'none'}")
    L += [""] + discrepancy_checks(recs, examples, old, new, todo)
    chart = make_ranking_chart(rows, o)
    (OUTD / f"{MODEL_TAG}_revalidation_report.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"before": before, "after": after, "flagged": {k: len(v) for k, v in cats.items()}, "rechecked": len(todo),
               "blast_unscored": len(pend)}, open(OUTD / "revalidation_metrics.json", "w"), indent=2)
    print("\n" + "\n".join(L))
    print(f"\nDONE. Everything is in {OUTD}\n  report: {OUTD / (MODEL_TAG + '_revalidation_report.md')}\n  chart: {chart}")


_PAYLOAD = """
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
"""


if __name__ == "__main__":
    main()
