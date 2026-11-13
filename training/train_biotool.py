# ==================================================================
# Qwen3.8-27B -> BioTool (biomedical tool calling) | Unsloth QLoRA
# GPU: A100 40GB (Jarvis Labs) | Non-thinking mode | Adapter-only checkpoints
# Run as a script (python train_biotool.py) or paste into one notebook cell.
# ==================================================================
import os, sys, subprocess, shutil, json

# ---------------- CONFIG ----------------
SAVE_ROOT   = os.path.abspath("Fine Tuned")
CKPT_DIR    = os.path.join(SAVE_ROOT, "checkpoints")          # adapter checkpoints
ADAPTER_DIR = os.path.join(SAVE_ROOT, "final_adapter")        # final LoRA adapter
MERGED_DIR  = os.path.join(SAVE_ROOT, "Qwen3.8-27B-BioTool-merged-16bit")

MODEL_NAME  = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
DATASET     = "gxx27/BioTool"
MAX_SEQ_LEN = 3072          # samples longer than this are dropped, never truncated
LORA_R, LORA_ALPHA = 16, 16
LR          = 2e-4
BATCH, GRAD_ACC = 1, 16     # effective batch size 16
EPOCHS      = 1
MAX_STEPS   = -1            # set to e.g. 20 for a quick speed test
MAX_TRAIN_SAMPLES = None    # e.g. 3000 to train on a subset
SAVE_STEPS  = 50
DO_MERGE    = True
SEED        = 3407

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------- 1. INSTALL ----------------
# (If torch/transformers were already imported in this kernel, restart it once after the first install.)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "unsloth", "unsloth_zoo", "transformers>=5.0.0", "trl", "peft", "accelerate",
    "bitsandbytes", "datasets", "flash-linear-attention", "sentencepiece", "protobuf",
])

# ---------------- 2. IMPORTS (unsloth first) ----------------
from unsloth import FastModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import torch, transformers, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig

assert int(transformers.__version__.split(".")[0]) >= 5, f"Need transformers v5+, got {transformers.__version__}"
assert torch.cuda.is_available(), "No GPU found"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | {p.total_memory/1e9:.1f} GB | transformers {transformers.__version__}")
free_gb = shutil.disk_usage(SAVE_ROOT).free / 1e9
print(f"Free disk: {free_gb:.0f} GB")
if free_gb < 120:
    print("WARNING: merging to 16-bit can need up to ~110 GB (output + cached 16-bit shards). Set DO_MERGE=False or free space.")

# ---------------- 3. TOKENIZER + DATA HELPERS ----------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
assert tok.chat_template, "Tokenizer has no chat template"
print("EOS:", tok.eos_token, "| PAD:", tok.pad_token)

def pj(x):
    return json.loads(x) if isinstance(x, str) else x

def norm_tools(field):
    tools = pj(field) if field else []
    if isinstance(tools, dict):
        tools = [tools]
    out = []
    for t in tools:
        t = pj(t)
        out.append(t if (t.get("type") == "function" and "function" in t)
                   else {"type": "function", "function": t})
    return out

def norm_calls(val):
    calls = pj(val)
    if isinstance(calls, dict):
        calls = [calls]
    out = []
    for c in calls:
        c = pj(c)
        args = c.get("arguments")
        args = {} if args in (None, "") else pj(args)
        if not isinstance(args, dict):
            raise ValueError("arguments is not a dict")
        out.append({"type": "function", "function": {"name": c["name"], "arguments": args}})
    return out

def build_messages(conv):
    msgs = []
    for turn in conv:
        role, val = turn["from"], turn["value"]
        if role == "system":
            msgs.append({"role": "system", "content": val})
        elif role in ("human", "user"):
            msgs.append({"role": "user", "content": val})
        elif role in ("gpt", "assistant"):
            msgs.append({"role": "assistant", "content": val})
        elif role == "function_call":
            calls = norm_calls(val)
            if msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("tool_calls") and not msgs[-1]["content"]:
                msgs[-1]["tool_calls"].extend(calls)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        elif role == "observation":
            msgs.append({"role": "tool", "content": val if isinstance(val, str) else json.dumps(val)})
        else:
            raise ValueError(f"Unknown role: {role}")
    return msgs

ERRS = []
def format_example(ex):
    try:
        msgs = build_messages(ex["conversations"])
        assert msgs[-1]["role"] == "assistant", "conversation must end with an assistant turn"
        tools = norm_tools(ex["tools"]) or None
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                       add_generation_prompt=False, enable_thinking=False)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        return {"text": text, "n_tokens": n, "ok": True}
    except Exception as e:
        if len(ERRS) < 5:
            ERRS.append(repr(e))
        return {"text": "", "n_tokens": 0, "ok": False}

# ---------------- 4. LOAD + FORMAT DATASET ----------------
raw = load_dataset(DATASET, split="train")
print(raw)
print("Roles in first sample:", [t["from"] for t in raw[0]["conversations"]])

fmt = raw.map(format_example, remove_columns=raw.column_names,
              load_from_cache_file=False, desc="Applying chat template")
n_total = len(fmt)
fmt = fmt.filter(lambda e: e["ok"])
print(f"Formatted OK: {len(fmt)}/{n_total}")
if len(fmt) < n_total:
    print("First errors:", ERRS)
assert len(fmt) > 0, "No samples formatted - paste the errors above"

lens = np.array(fmt["n_tokens"])
print("Token lengths -> mean %.0f | p50 %d | p95 %d | p99 %d | max %d" % (
    lens.mean(), *np.percentile(lens, [50, 95, 99]), lens.max()))
before = len(fmt)
fmt = fmt.filter(lambda e: e["n_tokens"] <= MAX_SEQ_LEN)
print(f"Dropped {before - len(fmt)} samples longer than {MAX_SEQ_LEN} tokens (no truncation)")
if MAX_TRAIN_SAMPLES:
    fmt = fmt.shuffle(seed=SEED).select(range(min(MAX_TRAIN_SAMPLES, len(fmt))))
train_ds = fmt.remove_columns(["n_tokens", "ok"])
print("Training samples:", len(train_ds))

# ---------------- 5. VERIFY CHAT TEMPLATE + FIND RESPONSE MARKER ----------------
msgs0, tools0 = build_messages(raw[0]["conversations"]), norm_tools(raw[0]["tools"]) or None
full0 = tok.apply_chat_template(msgs0, tools=tools0, tokenize=False, enable_thinking=False)
prompt0 = tok.apply_chat_template(msgs0[:-1], tools=tools0, tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
print("\n===== RENDERED SAMPLE (head) =====\n" + full0[:1200])
print("\n===== RENDERED SAMPLE (tail) =====\n" + full0[-900:])
if not (full0.startswith(prompt0) and "<|im_start|>assistant" in prompt0 and "<|im_end|>" in full0):
    raise RuntimeError("Chat template does not match the expected structure. "
                       "Paste the rendered sample above and I'll adapt the masking.")
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = prompt0[prompt0.rfind("<|im_start|>assistant"):]
print("\nResponse marker (loss starts after this):", repr(RESPONSE_PART))

# ---------------- 6. LOAD MODEL (4-bit QLoRA) ----------------
model, tokenizer = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,
    full_finetuning = False,
    offload_embedding = True,     # keeps the huge embedding in CPU RAM -> saves VRAM
    dtype = None,
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # text-only data
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = LORA_R, lora_alpha = LORA_ALPHA, lora_dropout = 0, bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = SEED, use_rslora = False, loftq_config = None,
)

# ---------------- 7. TRAINER ----------------
cfg = dict(
    dataset_text_field = "text",
    per_device_train_batch_size = BATCH,
    gradient_accumulation_steps = GRAD_ACC,
    warmup_steps = 20,
    num_train_epochs = EPOCHS,
    max_steps = MAX_STEPS,
    learning_rate = LR,
    lr_scheduler_type = "linear",
    weight_decay = 0.01,
    optim = "adamw_8bit",
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 5,
    save_strategy = "steps",
    save_steps = SAVE_STEPS,
    save_total_limit = 3,          # PEFT checkpoints contain the adapter only, not the base model
    output_dir = CKPT_DIR,
    seed = SEED,
    dataset_num_proc = 1,
    packing = False,
    report_to = "none",
)
try:
    args = SFTConfig(max_seq_length=MAX_SEQ_LEN, **cfg)
except TypeError:
    args = SFTConfig(max_length=MAX_SEQ_LEN, **cfg)

trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=train_ds, args=args)

try:
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART,
                                      response_part=RESPONSE_PART, tokenizer=tok)
except TypeError:
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART,
                                      response_part=RESPONSE_PART)

# Verify masking: only the tool call should be trained on
ex = trainer.train_dataset[0]
trained = tok.decode([t for t in ex["labels"] if t != -100])
gold_name = norm_calls(next(t["value"] for t in raw[0]["conversations"] if t["from"] == "function_call"))[0]["function"]["name"]
print("\n===== TRAINED-ON TEXT (sample 0) =====\n" + trained)
assert gold_name in trained, "Masking check failed: tool name not in trained tokens"
assert len(trained) < len(tok.decode(ex["input_ids"])) * 0.6, "Masking check failed: too much text is trained on"

# ---------------- 8. TRAIN (auto-resume from adapter checkpoint) ----------------
last = get_last_checkpoint(CKPT_DIR)
print("Resuming from:", last) if last else print("Starting fresh")
stats = trainer.train(resume_from_checkpoint=last) if last else trainer.train()
print(stats.metrics)
print(f"Peak VRAM reserved: {torch.cuda.max_memory_reserved()/1e9:.1f} GB")

# ---------------- 9. SAVE FINAL ADAPTER ----------------
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Adapter saved to:", ADAPTER_DIR)

# ---------------- 10. MERGE TO 16-BIT ----------------
if DO_MERGE:
    try:
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
        print("Merged 16-bit model saved to:", MERGED_DIR)
    except Exception as e:
        print("MERGE FAILED (adapter is safe in", ADAPTER_DIR, "):", repr(e))

# ---------------- 11. QUICK SANITY CHECK ON A TEST SAMPLE ----------------
try:
    test = load_dataset(DATASET, split="test")
    t = test[0]
    tm = build_messages(t["conversations"])
    inputs = tok.apply_chat_template(tm[:-1], tools=norm_tools(t["tools"]) or None,
                                     add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True, enable_thinking=False).to("cuda")
    FastModel.for_inference(model)
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    print("\nQUERY :", tm[-2]["content"] if len(tm) > 1 else "")
    print("MODEL :", tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False))
    print("GOLD  :", json.dumps(tm[-1].get("tool_calls", tm[-1]["content"])))
except Exception as e:
    print("Sanity check skipped:", repr(e))

print("\nDONE. Everything is in:", SAVE_ROOT)
