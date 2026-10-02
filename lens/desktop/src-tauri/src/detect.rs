// Environment detection: which LLMs this person actually uses, before anything is installed or downloaded.
//   runtimes : local inference servers (Ollama, LM Studio, vLLM, llama.cpp, ...) and the models they serve
//   weights  : model folders already on disk (Hugging Face / ModelScope caches, LM Studio, Nous)
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

fn weights() -> Vec<Value> {
    let h = home(); let mut out = vec![];
    let mut push = |source: &str, id: String, path: PathBuf| {
        // only chat / generative LLMs (skip embedding, reranker, vision-only models)
        if let Some(arch) = std::fs::read_to_string(path.join("config.json")).ok().and_then(|c| serde_json::from_str::<Value>(&c).ok()).and_then(|c| c["architectures"].as_array().cloned()) {
            if !arch.iter().any(|a| a.as_str().map(|s| s.ends_with("ForCausalLM") || s.ends_with("ForConditionalGeneration")).unwrap_or(false)) { return; }
        }
        let ok = loadable(&path);
        let gguf = std::fs::read_dir(&path).map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".gguf"))).unwrap_or(false);
        if ok || gguf { out.push(json!({ "source": source, "id": id, "path": path, "safetensors": ok, "gguf": gguf })); }
    };
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
