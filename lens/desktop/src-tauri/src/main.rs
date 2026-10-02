// Nous 知情助手 · desktop guardian.
// First run DETECTS which LLMs the person uses (local runtimes, weights on disk, sites, API models) and recommends how to
// read them; nothing is installed or downloaded until they choose. The app then owns the local engine (127.0.0.1:7869).
// A small always-on-top companion shows what the engine reads; the panel is the engine's page.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod detect;

use std::io::Read;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, RunEvent, WebviewUrl, WebviewWindowBuilder};

const ENGINE: &str = "127.0.0.1:7869";
const MIRROR: &str = "https://pypi.tuna.tsinghua.edu.cn/simple";

struct Engine(Mutex<Option<Child>>);

fn home() -> PathBuf { PathBuf::from(std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| ".".into())) }
fn base() -> PathBuf { home().join(".nous-lens") }
fn venv_bin(name: &str) -> PathBuf {
    if cfg!(windows) { base().join("venv").join("Scripts").join(format!("{name}.exe")) } else { base().join("venv").join("bin").join(name) }
}
fn config_path() -> PathBuf { base().join("config.json") }
fn config() -> serde_json::Value { std::fs::read_to_string(config_path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(serde_json::json!({})) }
fn save_config(v: &serde_json::Value) { let _ = std::fs::create_dir_all(base()); let _ = std::fs::write(config_path(), serde_json::to_string_pretty(v).unwrap()); }
fn download_dir() -> PathBuf { base().join("models").join("Qwen3-4B-Instruct-2507") }
// the reader's weights: an existing folder the person already has, or the optional download
fn model_dir() -> PathBuf { config()["reader"]["path"].as_str().map(PathBuf::from).unwrap_or_else(download_dir) }
fn reader_ok() -> bool { let m = model_dir(); m.join("config.json").exists() && (config()["reader"]["source"] != "download" || m.join(".complete").exists()) }
fn ready() -> bool { venv_bin("python").exists() && config()["reader"].is_object() && reader_ok() }
fn engine_up() -> bool { TcpStream::connect_timeout(&ENGINE.parse().unwrap(), Duration::from_millis(300)).is_ok() }
fn engine_root(app: &AppHandle) -> PathBuf { app.path().resource_dir().expect("resources").join("engine") }

// GUI apps on macOS start with a minimal PATH: add the usual places Python lives.
fn path_env() -> String {
    let mut p = vec!["/opt/homebrew/bin".to_string(), "/usr/local/bin".into(), format!("{}/.local/bin", home().display())];
    p.push(std::env::var("PATH").unwrap_or_default());
    p.join(if cfg!(windows) { ";" } else { ":" })
}
fn cmd(prog: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut c = Command::new(prog);
    c.env("PATH", path_env()).env("PYTHONUNBUFFERED", "1");
    #[cfg(windows)] { use std::os::windows::process::CommandExt; c.creation_flags(0x08000000); }
    c
}

fn emit(app: &AppHandle, kind: &str, text: &str) { let _ = app.emit("setup", serde_json::json!({ "kind": kind, "text": text })); }

// run a command, streaming its output lines (tqdm uses \r) to the setup window
fn run_logged(app: &AppHandle, mut c: Command) -> bool {
    c.stdout(Stdio::piped()).stderr(Stdio::piped());
    let Ok(mut child) = c.spawn() else { return false };
    let mut pipes: Vec<Box<dyn Read + Send>> = vec![];
    if let Some(o) = child.stdout.take() { pipes.push(Box::new(o)); }
    if let Some(e) = child.stderr.take() { pipes.push(Box::new(e)); }
    let hs: Vec<_> = pipes.into_iter().map(|mut p| { let a = app.clone(); thread::spawn(move || {
        let mut buf = [0u8; 4096]; let mut line = Vec::new();
        while let Ok(n) = p.read(&mut buf) { if n == 0 { break; }
            for &b in &buf[..n] { if b == b'\n' || b == b'\r' { if !line.is_empty() { emit(&a, "log", &String::from_utf8_lossy(&line)); line.clear(); } } else { line.push(b); } } }
    }) }).collect();
    let ok = child.wait().map(|s| s.success()).unwrap_or(false);
    for h in hs { let _ = h.join(); }
    ok
}

fn find_python() -> Option<String> {
    let h = home().display().to_string();
    let mut cands: Vec<String> = if cfg!(windows) { vec!["py".into(), "python".into()] } else { vec![] };
    for v in ["3.13", "3.12", "3.11", "3.10"] {
        for d in ["/opt/homebrew/bin", "/usr/local/bin", &format!("{h}/.local/bin")] { cands.push(format!("{d}/python{v}")); }
        cands.push(format!("/Library/Frameworks/Python.framework/Versions/{v}/bin/python3"));
    }
    cands.push("python3".into());
    cands.into_iter().find(|c| {
        let mut k = cmd(c); if c == "py" { k.arg("-3"); }
        k.args(["-c", "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"]).stdout(Stdio::null()).stderr(Stdio::null())
            .status().map(|s| s.success()).unwrap_or(false)
    })
}

fn setup(app: AppHandle) {
    thread::spawn(move || {
        let _ = std::fs::create_dir_all(base());
        if !venv_bin("python").exists() {
            emit(&app, "step", "1/3 准备运行环境");
            let Some(py) = find_python() else { emit(&app, "error", "没有找到 Python 3.10 或更新版本。请从 python.org 安装后重新打开本应用。"); return; };
            let mut c = cmd(&py); if py == "py" { c.arg("-3"); } c.args(["-m", "venv"]).arg(base().join("venv"));
            if !run_logged(&app, c) { emit(&app, "error", "创建运行环境失败。"); return; }
            let req = engine_root(&app).join("requirements.txt");
            emit(&app, "step", "2/3 安装组件（约 3–10 分钟）");
            if cfg!(windows) && cmd("nvidia-smi").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false) {
                let mut c = cmd(venv_bin("pip")); c.args(["install", "torch", "--index-url", "https://download.pytorch.org/whl/cu128"]); run_logged(&app, c);
            }
            let mut c = cmd(venv_bin("pip")); c.args(["install", "-i", MIRROR, "-r"]).arg(&req);
            if !run_logged(&app, c) {
                emit(&app, "log", "国内镜像暂不可用，改用官方源继续安装…");
                let mut c = cmd(venv_bin("pip")); c.args(["install", "-r"]).arg(&req);
                if !run_logged(&app, c) { let _ = std::fs::remove_dir_all(base().join("venv")); emit(&app, "error", "安装失败，请检查网络后重新打开本应用。"); return; }
            }
        }
        if config()["reader"]["source"] == "download" && !download_dir().join(".complete").exists() {
            emit(&app, "step", "3/3 下载本地读者 Qwen3-4B（约 8 GB，只需一次）");
            let mut c = cmd(venv_bin("modelscope")); c.args(["download", "--model", "Qwen/Qwen3-4B-Instruct-2507", "--local_dir"]).arg(download_dir());
            if !run_logged(&app, c) { emit(&app, "error", "模型下载失败，请检查网络后重试（会从断点继续）。"); return; }
            let _ = std::fs::write(download_dir().join(".complete"), b"");
        } else { emit(&app, "step", "3/3 使用你电脑上已有的模型，无需下载"); }
        emit(&app, "done", "准备完成");
        start_engine(&app);
    });
}

fn start_engine(app: &AppHandle) {
    if engine_up() { return; }                                    // already running (e.g. started from the terminal)
    let root = engine_root(app);
    let mut c = cmd(venv_bin("python"));
    c.args(["-m", "nous_lens.server", "--no-browser", "--model"]).arg(model_dir())
        .current_dir(&root).env("PYTHONPATH", &root).stdout(Stdio::null()).stderr(Stdio::null());
    if let Ok(child) = c.spawn() { *app.state::<Engine>().0.lock().unwrap() = Some(child); }
}

fn stop_engine(app: &AppHandle) {
    if let Some(mut c) = app.state::<Engine>().0.lock().unwrap().take() { let _ = c.kill(); let _ = c.wait(); }
}

#[tauri::command]
fn open_panel(app: AppHandle, event: Option<String>) {
    let url = match event.as_deref() { Some("install") => format!("http://{ENGINE}/?install=1"), Some(e) => format!("http://{ENGINE}/?event={e}"), None => format!("http://{ENGINE}/") };
    let u: tauri::Url = url.parse().unwrap();
    if let Some(w) = app.get_webview_window("panel") { let _ = w.navigate(u); let _ = w.show(); let _ = w.set_focus(); return; }
    let _ = WebviewWindowBuilder::new(&app, "panel", WebviewUrl::External(u)).title("Nous 知情助手").inner_size(1320.0, 840.0).min_inner_size(900.0, 600.0).build();
}

#[tauri::command]
fn engine_status() -> serde_json::Value {
    serde_json::json!({ "up": engine_up(), "ready": ready(), "config": config() })
}

#[tauri::command]
fn detect_env() -> serde_json::Value { detect::detect() }

#[tauri::command]
fn registry() -> serde_json::Value { serde_json::from_str(detect::REGISTRY).unwrap() }

// remember the API models a person (or a company) uses, e.g. Kimi K3 behind their own base_url
#[tauri::command]
fn set_api_models(models: serde_json::Value) { let mut c = config(); c["api_models"] = models; save_config(&c); }

// the person's choice: read with weights they already have ("existing", path), or download a local reader ("download")
#[tauri::command]
fn choose_reader(app: AppHandle, source: String, path: Option<String>, family: Option<String>) {
    let mut c = config();
    let p = if source == "download" { download_dir() } else { PathBuf::from(path.unwrap_or_default()) };
    c["reader"] = serde_json::json!({ "source": source, "path": p, "family": family.unwrap_or_else(|| "qwen3-4b-2507".into()), "chosen": true });
    save_config(&c); stop_engine(&app); setup(app);
}

#[tauri::command]
fn open_setup(app: AppHandle) { open_wizard(&app); }

fn open_wizard(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("setup") { let _ = w.show(); let _ = w.set_focus(); return; }
    let _ = WebviewWindowBuilder::new(app, "setup", WebviewUrl::App("setup.html".into())).title("Nous 知情助手 · 识别你在用的 AI")
        .inner_size(760.0, 640.0).min_inner_size(640.0, 560.0).center().build();
}

fn place_companion(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("companion") {
        if let Ok(Some(m)) = w.current_monitor() {
            let (s, sc) = (m.size(), m.scale_factor()); let (ww, wh) = ((260.0 * sc) as i32, (320.0 * sc) as i32);
            let _ = w.set_position(PhysicalPosition::new(s.width as i32 - ww - (24.0 * sc) as i32, s.height as i32 - wh - (64.0 * sc) as i32));
        }
        let _ = w.show();
    }
}

fn main() {
    if std::env::args().any(|a| a == "--detect") { println!("{}", serde_json::to_string_pretty(&detect::detect()).unwrap()); return; }   // diagnostics
    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .manage(Engine(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![open_panel, engine_status, detect_env, registry, set_api_models, choose_reader, open_setup])
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);   // menu-bar app: no Dock icon
            let h = app.handle().clone();
            let panel = MenuItem::with_id(app, "panel", "打开面板", true, None::<&str>)?;
            let comp = CheckMenuItem::with_id(app, "companion", "显示守护伙伴", true, true, None::<&str>)?;
            let clip = CheckMenuItem::with_id(app, "clipboard", "检查复制的 AI 回复", true, false, None::<&str>)?;
            let ext = MenuItem::with_id(app, "extension", "安装浏览器插件…", true, None::<&str>)?;
            let wiz = MenuItem::with_id(app, "wizard", "识别我在用的 AI…", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "退出 Nous", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&panel, &PredefinedMenuItem::separator(app)?, &comp, &clip, &wiz, &ext, &PredefinedMenuItem::separator(app)?, &quit])?;
            let tray_img = tauri::image::Image::from_bytes(include_bytes!("../icons/tray.png"))?;
            let (comp2, clip2) = (comp.clone(), clip.clone());
            TrayIconBuilder::with_id("nous").icon(tray_img).icon_as_template(true).tooltip("Nous 知情助手").menu(&menu).show_menu_on_left_click(true)
                .on_menu_event(move |app, e| match e.id().as_ref() {
                    "panel" => open_panel(app.clone(), None),
                    "companion" => if let Some(w) = app.get_webview_window("companion") {
                        let on = comp2.is_checked().unwrap_or(true); if on { place_companion(app) } else { let _ = w.hide(); } },
                    "clipboard" => { let _ = app.emit("clipboard-toggle", clip2.is_checked().unwrap_or(false)); }
                    "extension" => {
                        let dir = app.path().resource_dir().map(|d| d.join("extension")).unwrap_or_default();
                        let opener = if cfg!(target_os = "macos") { "open" } else if cfg!(windows) { "explorer" } else { "xdg-open" };
                        let _ = Command::new(opener).arg(dir).spawn();
                        open_panel(app.clone(), Some("install".into()));
                    }
                    "wizard" => open_wizard(app),
                    "quit" => { stop_engine(app); app.exit(0); }
                    _ => {}
                })
                .build(app)?;
            place_companion(&h);
            // never install or download on launch: start what was chosen, otherwise detect and let the person choose
            if ready() { start_engine(&h); } else { open_wizard(&h); }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Nous")
        .run(|app, e| match e {
            // closing the panel or setup window must not quit a menu-bar app; only "退出 Nous" (explicit exit code) does
            RunEvent::ExitRequested { code: None, api, .. } => api.prevent_exit(),
            RunEvent::ExitRequested { .. } | RunEvent::Exit => stop_engine(app),
            _ => {}
        });
}
