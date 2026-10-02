"""Load only the first K decoder layers of a chat model, from a Hugging Face folder (safetensors) or a GGUF file (e.g. an
Ollama blob). The reader never generates, so upper layers and the LM head are never loaded.

GGUF: transformers' own loader dequantizes EVERY tensor to fp32 before discarding unused layers (a 1B model peaks at ~8 GB,
an 8B one would not fit a 16 GB Mac). Here tensors are read one at a time, layers >= K are skipped before dequantization,
and each tensor is cast to bf16 immediately. Name mapping and per-architecture fixes (e.g. Llama q/k permutation) reuse
transformers' GGUF utilities, so supported architectures are the same as transformers' GGUF support."""
import os, re
import numpy as np, torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


def is_gguf(path):
    return os.path.isfile(path)            # a .gguf file or an Ollama blob (a file); HF models are folders


def _cache_dir(path):
    """~/.nous-lens/cache/gguf/<key>: the GGUF's config and tokenizer, converted once (parsing a vocab takes seconds)."""
    import hashlib, json
    st = os.stat(path); key = hashlib.sha1(f"{os.path.realpath(path)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    d = os.path.expanduser(f"~/.nous-lens/cache/gguf/{key}")
    if not os.path.exists(os.path.join(d, "ok")):
        from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
        os.makedirs(d, exist_ok=True)
        c = dict(load_gguf_checkpoint(path, return_tensors=False)["config"])
        AutoConfig.for_model(c.pop("model_type"), **c).save_pretrained(d)
        _gguf_tokenizer(path).save_pretrained(d)
        json.dump({"source": os.path.realpath(path)}, open(os.path.join(d, "ok"), "w"))
    return d


def config_of(path):
    """(config, total decoder layers)"""
    cfg = AutoConfig.from_pretrained(_cache_dir(path) if is_gguf(path) else path)
    n = getattr(cfg, "num_hidden_layers", None) or cfg.get_text_config().num_hidden_layers
    return cfg, n


# pre-tokenizer regexes by GGUF `tokenizer.ggml.pre` (as in llama.cpp). transformers' GGUF converter falls back to plain
# ByteLevel for some (e.g. Llama 3), which splits text differently from the official tokenizer.
_R_LLAMA3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
_R_QWEN2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
_PRE = {"llama-bpe": _R_LLAMA3, "llama3": _R_LLAMA3, "smaug-bpe": _R_LLAMA3, "dbrx": _R_LLAMA3,
        "qwen2": _R_QWEN2, "deepseek-r1-qwen": _R_QWEN2}


def _gguf_str(reader, key):
    f = reader.fields.get(key)
    return bytes(f.parts[-1]).decode("utf-8", "replace") if f is not None else None


def tokenizer_of(path):
    return AutoTokenizer.from_pretrained(_cache_dir(path) if is_gguf(path) else path)


def _gguf_tokenizer(path):
    import gguf, json
    from tokenizers import Tokenizer
    tok = AutoTokenizer.from_pretrained(os.path.dirname(path) or ".", gguf_file=os.path.basename(path))
    rx = _PRE.get(_gguf_str(gguf.GGUFReader(path), "tokenizer.ggml.pre") or "")
    j = json.loads(tok.backend_tokenizer.to_str())
    if rx and (j.get("pre_tokenizer") or {}).get("type") == "ByteLevel":
        j["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": rx}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]}
        j["normalizer"] = None; j["decoder"] = {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
        tok._tokenizer = Tokenizer.from_str(json.dumps(j))
    return tok


def _truncate(cfg, keep):
    cfg.num_hidden_layers = keep
    if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:keep]
    return cfg


def _dequant(gguf, t, rows=4096):
    """Dequantize in row chunks into one fp32 array (whole-tensor dequantization peaks at several times the output size)."""
    d = t.data
    if d.ndim < 2 or d.shape[0] <= rows: return gguf.dequantize(d, t.tensor_type)
    first = gguf.dequantize(d[:rows], t.tensor_type)
    out = np.empty((d.shape[0],) + first.shape[1:], dtype=np.float32); out[:rows] = first
    for i in range(rows, d.shape[0], rows): out[i:i + rows] = gguf.dequantize(d[i:i + rows], t.tensor_type)
    return out


def load_layers(path, keep, device, dtype=None):
    """-> (tokenizer, base model with `keep` decoder layers on device)"""
    dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)
    tok = tokenizer_of(path)
    if not is_gguf(path):
        cfg, _ = config_of(path)
        return tok, AutoModel.from_pretrained(path, config=_truncate(cfg, keep), dtype=dtype).to(device).eval()
    import gguf
    from transformers.modeling_gguf_pytorch_utils import TENSOR_PROCESSORS, TensorProcessor, get_gguf_hf_weights_map
    from transformers.modeling_utils import no_init_weights
    full_cfg, _ = config_of(path); arch = full_cfg.model_type; cdict = full_cfg.to_dict()
    with torch.device("meta"):                    # full causal-LM skeleton, only to get the GGUF -> HF name mapping
        name_map = get_gguf_hf_weights_map(AutoModelForCausalLM.from_config(full_cfg))
    cfg, _ = config_of(path); cfg = _truncate(cfg, keep)
    with no_init_weights():                       # weights come from the file; skip random init
        model = AutoModel.from_config(cfg, dtype=dtype)
    processor = TENSOR_PROCESSORS.get(arch, TensorProcessor)(config=cdict)
    params = dict(model.named_parameters()); done = set(); reader = gguf.GGUFReader(path)
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        if (m and int(m.group(1)) >= keep) or t.name.startswith("output."): continue   # skip unused layers + LM head before dequantizing (output_norm is kept)
        w = _dequant(gguf, t)
        r = processor.process(weights=w, name=t.name, tensor_key_mapping=name_map, parsed_parameters={"config": cdict})
        hf = name_map.get(r.name)
        if hf is None: continue
        k = hf[len("model."):] if hf.startswith("model.") else hf
        if k not in params: continue
        with torch.no_grad(): params[k].copy_(torch.from_numpy(np.ascontiguousarray(r.weights)).view(params[k].shape))   # one tensor at a time: low peak memory
        done.add(k); del w, r
    missing = [k for k in params if k not in done]
    if missing: raise RuntimeError(f"GGUF load incomplete for {arch}: missing {missing[:5]}")
    return tok, model.to(device).eval()
