// Environment detection: which LLMs this person actually uses, before anything is installed or downloaded.
//   runtimes : local inference servers (Ollama, LM Studio, vLLM, llama.cpp, ...) and the models they serve
//   weights  : models already on disk (Hugging Face / ModelScope caches, Ollama's store, LM Studio, Nous), safetensors or GGUF
//   hardware : OS, chip, memory, NVIDIA GPUs -> whether a local reader is even an option
// Classification (which family, white-box or reader) happens in the UI against models.json.
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

pub const REGISTRY: &str = include_str!("../../../nous_lens/models.json");

fn http_get(port: u16, path: &str) -> Option<String> {
    let addr = format!("127.0.0.1:{port}").parse().ok()?;
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(250)).ok()?;
    s.set_read_timeout(Some(Duration::from_millis(1500))).ok()?;
    write!(s, "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nAccept: application/json\r\n\r\n").ok()?;
    let mut buf = String::new(); s.read_to_string(&mut buf).ok()?;
    let (head, body) = buf.split_once("\r\n\r\n")?;
    if !head.starts_with("HTTP/1.") || !head.split_whitespace().nth(1)?.starts_with('2') { return None; }
    Some(body.to_string())
}

fn runtimes() -> Vec<Value> {
    let reg: Value = serde_json::from_str(REGISTRY).unwrap();
    let mut out = vec![];
    for r in reg["runtimes"].as_array().unwrap() {
        let port = r["port"].as_u64().unwrap() as u16;
        if port == 7869 { continue; }
        let Some(body) = http_get(port, r["list"].as_str().unwrap()) else { continue };
        let Ok(v) = serde_json::from_str::<Value>(&body) else { continue };
        let models: Vec<Value> = if r["format"] == "ollama" {
            v["models"].as_array().cloned().unwrap_or_default().iter().map(|m| json!({
                "id": m["name"], "format": m["details"]["format"], "family": m["details"]["family"], "size": m["details"]["parameter_size"], "quant": m["details"]["quantization_level"] })).collect()
        } else {
            v["data"].as_array().cloned().unwrap_or_default().iter().map(|m| json!({ "id": m["id"] })).collect()
        };
        out.push(json!({ "id": r["id"], "name": r["name"], "port": port, "models": models }));
    }
    out
}

fn home() -> PathBuf { PathBuf::from(std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default()) }

// a folder that transformers can load directly: config.json + safetensors
fn loadable(p: &Path) -> bool {
    p.join("config.json").exists() && std::fs::read_dir(p).map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))).unwrap_or(false)
}

// GGUF architectures the engine can load (transformers' GGUF support; names as in the GGUF header)
const GGUF_ARCHS: &[&str] = &["llama", "mistral", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "gemma2", "gemma3", "phi3", "falcon", "stablelm",
    "starcoder2", "nemotron", "gpt2", "bloom", "lfm2", "deci"];
// encoder / embedding / vision-only architectures are not chat models
const NOT_LLM: &[&str] = &["bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "t5encoder", "clip", "mllama-vision", "neo-bert", "modern-bert"];

// read the GGUF header: (architecture, block count, file_type). Stops as soon as both are found, so token arrays are never read.
fn gguf_header(path: &Path) -> Option<(String, Option<u64>, Option<u64>)> {
    use std::io::{BufReader, Seek, SeekFrom};
    let mut f = BufReader::new(std::fs::File::open(path).ok()?);
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    f.read_exact(&mut b4).ok()?; if &b4 != b"GGUF" { return None; }
    f.read_exact(&mut b4).ok()?; f.read_exact(&mut b8).ok()?; f.read_exact(&mut b8).ok()?;
    let n_kv = u64::from_le_bytes(b8);
    fn rd_u64(f: &mut impl Read) -> Option<u64> { let mut b = [0u8; 8]; f.read_exact(&mut b).ok()?; Some(u64::from_le_bytes(b)) }
    fn rd_u32(f: &mut impl Read) -> Option<u32> { let mut b = [0u8; 4]; f.read_exact(&mut b).ok()?; Some(u32::from_le_bytes(b)) }
    fn rd_str(f: &mut impl Read) -> Option<String> { let n = rd_u64(f)?; if n > 1 << 20 { return None; } let mut v = vec![0u8; n as usize]; f.read_exact(&mut v).ok()?; Some(String::from_utf8_lossy(&v).into()) }
    fn size_of(t: u32) -> Option<i64> { Some(match t { 0 | 1 | 7 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4, 10 | 11 | 12 => 8, _ => return None }) }
    // read a scalar as u64 (for integer types), or skip a value of type t
    fn value(f: &mut (impl Read + Seek), t: u32) -> Option<Option<u64>> {
        match t {
            8 => { let n = rd_u64(f)?; f.seek(SeekFrom::Current(n as i64)).ok()?; Some(None) }
            9 => { let et = rd_u32(f)?; let n = rd_u64(f)?;
                   if let Some(sz) = size_of(et) { f.seek(SeekFrom::Current(sz * n as i64)).ok()?; } else { for _ in 0..n { value(f, et)?; } }
                   Some(None) }
            _ => { let sz = size_of(t)?; let mut v = [0u8; 8]; f.read_exact(&mut v[..sz as usize]).ok()?; Some(Some(u64::from_le_bytes(v))) }
        }
    }
    let (mut arch, mut blocks, mut ftype): (Option<String>, Option<u64>, Option<u64>) = (None, None, None);
    for _ in 0..n_kv.min(4096) {
        let k = rd_str(&mut f)?; let t = rd_u32(&mut f)?;
        if k == "general.architecture" && t == 8 { arch = Some(rd_str(&mut f)?); continue; }
        let v = value(&mut f, t)?;
        if k == "general.file_type" { ftype = v; }
        if k.ends_with(".block_count") { blocks = v; }
        if arch.is_some() && blocks.is_some() && ftype.is_some() { break; }
    }
    Some((arch?, blocks, ftype))
}

// approximate bits per weight of a GGUF file_type, to estimate the memory the dequantized (bf16) reader needs
fn gguf_bpw(ftype: Option<u64>) -> f64 {
    match ftype.unwrap_or(15) { 0 => 32.0, 1 | 32 => 16.0, 7 => 8.5, 2 | 3 => 4.5, 8 | 9 => 5.5, 10 => 3.0, 11 | 12 | 13 => 3.9, 14 => 4.6, 16 => 5.5, 17 => 5.7, 18 => 6.6, _ => 4.85 }
}

// one GGUF file (a .gguf or an Ollama blob) -> description, or None if it is not a chat LLM
fn gguf_entry(source: &str, id: String, file: PathBuf) -> Option<Value> {
    let (arch, layers, ftype) = gguf_header(&file)?;
    if NOT_LLM.contains(&arch.as_str()) || arch.contains("bert") || arch.contains("clip") { return None; }
    let size = std::fs::metadata(&file).ok()?.len() as f64 / 1e9;
    let r1 = |x: f64| (x * 10.0).round() / 10.0;
    Some(json!({ "source": source, "id": id, "path": file, "safetensors": false, "gguf": true, "arch": arch, "adaptable": GGUF_ARCHS.contains(&arch.as_str()),
                 "size_gb": r1(size), "bf16_gb": r1(size * 16.0 / gguf_bpw(ftype)), "layers": layers }))
}

// Ollama's model store: manifests/<host>/<namespace>/<model>/<tag> -> the model layer's blob (a GGUF file)
fn ollama_store() -> Vec<Value> {
    let root = std::env::var("OLLAMA_MODELS").map(PathBuf::from).unwrap_or(home().join(".ollama/models"));
    let mut out = vec![];
    for host in std::fs::read_dir(root.join("manifests")).into_iter().flatten().flatten() {
        for ns in std::fs::read_dir(host.path()).into_iter().flatten().flatten() {
            for model in std::fs::read_dir(ns.path()).into_iter().flatten().flatten() {
                for tag in std::fs::read_dir(model.path()).into_iter().flatten().flatten() {
                    let Some(m) = std::fs::read_to_string(tag.path()).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()) else { continue };
                    let Some(digest) = m["layers"].as_array().and_then(|l| l.iter().find(|x| x["mediaType"] == "application/vnd.ollama.image.model")).and_then(|x| x["digest"].as_str()) else { continue };
                    let blob = root.join("blobs").join(digest.replace(':', "-"));
                    let (h, n) = (host.file_name().to_string_lossy().to_string(), ns.file_name().to_string_lossy().to_string());
                    let base = if h == "registry.ollama.ai" && n == "library" { String::new() } else if h == "registry.ollama.ai" { format!("{n}/") } else { format!("{h}/{n}/") };
                    let id = format!("{base}{}:{}", model.file_name().to_string_lossy(), tag.file_name().to_string_lossy());
                    if let Some(v) = gguf_entry("Ollama", id, blob) { out.push(v); }
                }
            }
        }
    }
    out
}

// one model folder -> description, or None if it is not a chat/generative LLM in a loadable format
fn entry(source: &str, id: String, path: PathBuf) -> Option<Value> {
    if let Some(arch) = std::fs::read_to_string(path.join("config.json")).ok().and_then(|c| serde_json::from_str::<Value>(&c).ok()).and_then(|c| c["architectures"].as_array().cloned()) {
        if !arch.iter().any(|a| a.as_str().map(|s| s.ends_with("ForCausalLM") || s.ends_with("ForConditionalGeneration")).unwrap_or(false)) { return None; }
    }
    let ok = loadable(&path);
    if !ok {
        if path.is_file() { return gguf_entry(source, id, path); }
        let g = std::fs::read_dir(&path).ok()?.flatten().filter(|e| { let n = e.file_name().to_string_lossy().to_lowercase(); n.ends_with(".gguf") && !n.contains("mmproj") })
            .max_by_key(|e| e.metadata().map(|m| m.len()).unwrap_or(0))?;
        return gguf_entry(source, id, g.path());
    }
    let size: u64 = std::fs::read_dir(&path).map(|d| d.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()).unwrap_or(0);
    let layers = std::fs::read_to_string(path.join("config.json")).ok().and_then(|c| serde_json::from_str::<Value>(&c).ok())
        .and_then(|c| c["num_hidden_layers"].as_u64().or_else(|| c["text_config"]["num_hidden_layers"].as_u64()));
    let gb = (size as f64 / 1e9 * 10.0).round() / 10.0;
    Some(json!({ "source": source, "id": id, "path": path, "safetensors": true, "gguf": false, "adaptable": true, "size_gb": gb, "bf16_gb": gb, "layers": layers }))
}

// a folder the person points at (models downloaded to arbitrary places)
pub fn inspect(path: &str) -> Value {
    let p = PathBuf::from(path.trim().trim_matches('"').trim_matches('\''));
    entry("手动添加", p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), p).unwrap_or(Value::Null)
}

fn weights() -> Vec<Value> {
    let h = home(); let mut out = vec![];
    let mut push = |source: &str, id: String, path: PathBuf| { if let Some(v) = entry(source, id, path) { out.push(v); } };
    // Hugging Face hub cache: models--Org--Name/snapshots/<rev>/
    let hf = std::env::var("HF_HOME").map(PathBuf::from).unwrap_or(h.join(".cache/huggingface")).join("hub");
    for e in std::fs::read_dir(&hf).into_iter().flatten().flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if let Some(rest) = n.strip_prefix("models--") {
            if let Some(snap) = std::fs::read_dir(e.path().join("snapshots")).ok().and_then(|d| d.flatten().next()) { push("Hugging Face 缓存", rest.replace("--", "/"), snap.path()); }
        }
    }
    // ModelScope cache: hub/models/Org/Name or hub/Org/Name
    let ms = h.join(".cache/modelscope/hub");
    for base in [ms.join("models"), ms.clone()] {
        for org in std::fs::read_dir(&base).into_iter().flatten().flatten() {
            for m in std::fs::read_dir(org.path()).into_iter().flatten().flatten() {
                push("ModelScope 缓存", format!("{}/{}", org.file_name().to_string_lossy(), m.file_name().to_string_lossy()), m.path());
            }
        }
    }
    // LM Studio and Nous's own folder
    for base in [h.join(".lmstudio/models"), h.join(".cache/lm-studio/models")] {
        for org in std::fs::read_dir(&base).into_iter().flatten().flatten() {
            for m in std::fs::read_dir(org.path()).into_iter().flatten().flatten() {
                push("LM Studio", format!("{}/{}", org.file_name().to_string_lossy(), m.file_name().to_string_lossy()), m.path());
            }
        }
    }
    for m in std::fs::read_dir(h.join(".nous-lens/models")).into_iter().flatten().flatten() {
        push("Nous", m.file_name().to_string_lossy().to_string(), m.path());
    }
    out.extend(ollama_store());
    out
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let mut c = Command::new(cmd); c.args(args);
    #[cfg(windows)] { use std::os::windows::process::CommandExt; c.creation_flags(0x08000000); }
    let o = c.output().ok()?; if !o.status.success() { return None; }
    Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn hardware() -> Value {
    let os = std::env::consts::OS; let arch = std::env::consts::ARCH;
    let (chip, mem_gb) = match os {
        "macos" => (run("sysctl", &["-n", "machdep.cpu.brand_string"]), run("sysctl", &["-n", "hw.memsize"]).and_then(|s| s.parse::<f64>().ok()).map(|b| b / 1073741824.0)),
        "windows" => (run("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]),
                      run("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]).and_then(|s| s.parse::<f64>().ok()).map(|b| b / 1073741824.0)),
        _ => (None, std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().next().and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).map(|k| k / 1048576.0)),
    };
    let gpus: Vec<Value> = run("nvidia-smi", &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).map(|s| s.lines().map(|l| {
        let mut p = l.split(','); json!({ "name": p.next().unwrap_or("").trim(), "vram_gb": p.next().and_then(|v| v.trim().parse::<f64>().ok()).map(|m| m / 1024.0) }) }).collect()).unwrap_or_default();
    let apple = os == "macos" && arch == "aarch64";
    let mem = mem_gb.unwrap_or(0.0);
    let best_vram = gpus.iter().filter_map(|g| g["vram_gb"].as_f64()).fold(0.0, f64::max);
    // a 4B reader (layers 0-22 only) needs ~5 GB of accelerator memory
    let local_reader_ok = (apple && mem >= 15.5) || best_vram >= 7.5;
    json!({ "os": os, "arch": arch, "chip": chip, "mem_gb": mem_gb.map(|m| (m * 10.0).round() / 10.0), "gpus": gpus, "apple_silicon": apple, "local_reader_ok": local_reader_ok })
}

pub fn detect() -> Value {
    json!({ "runtimes": runtimes(), "weights": weights(), "hardware": hardware() })
}
