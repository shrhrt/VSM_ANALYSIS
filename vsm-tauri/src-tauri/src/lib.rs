use std::io::Write;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// バックエンド子プロセスのハンドル。アプリ終了時にkillするため保持しておく
struct BackendProcess(Mutex<Option<Child>>);

/// 診断ログを %TEMP%\vsm_debug.log に追記する
fn log(msg: &str) {
    let path = std::env::temp_dir().join("vsm_debug.log");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{}", msg);
    }
}

/// 127.0.0.1:port が接続を受け付ける（＝バックエンドが既に稼働中）かを判定
fn port_open(port: u16) -> bool {
    std::net::TcpStream::connect(("127.0.0.1", port)).is_ok()
}

/// 開発モード用: Python ソースから uvicorn を起動する。
/// backend.exe（本番同梱サイドカー）が無い開発環境で、`npm run tauri dev` 単体でも
/// バックエンドが立ち上がるようにするためのフォールバック。
#[cfg(debug_assertions)]
fn spawn_dev_uvicorn() -> Result<Child, String> {
    // <root>/vsm-tauri/src-tauri（このクレート）から 2 つ上がプロジェクトルート
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| "プロジェクトルートの特定に失敗".to_string())?
        .to_path_buf();
    log(&format!("dev: project root = {root:?}"));

    // .venv の python を優先（uvicorn/fastapi が入っているため）
    let venv_py = root.join(".venv").join("Scripts").join("python.exe");
    let python = if venv_py.exists() {
        venv_py
    } else {
        std::path::PathBuf::from("python")
    };
    log(&format!("dev: python = {python:?}"));

    let mut cmd = Command::new(&python);
    cmd.args([
        "-m", "uvicorn", "backend.main:app",
        "--port", "8000", "--log-level", "warning",
    ])
    .current_dir(&root);
    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    cmd.spawn()
        .map_err(|e| format!("dev uvicorn の起動に失敗: {e}（.venv と依存関係を確認してください）"))
}

/// 実行環境に応じてバックエンドを起動し、Child を返す。
/// 本番は同梱の backend.exe、開発は Python ソースの uvicorn。
fn spawn_backend() -> Result<Child, String> {
    let current_exe =
        std::env::current_exe().map_err(|e| format!("current_exe() 失敗: {e}"))?;
    let exe_dir = current_exe
        .parent()
        .ok_or_else(|| "実行ファイルのディレクトリ特定に失敗".to_string())?
        .to_path_buf();
    let backend = exe_dir.join("backend.exe");
    log(&format!(
        "backend.exe path: {backend:?} exists: {}",
        backend.exists()
    ));

    if backend.exists() {
        // 本番: PyInstaller で固めたサイドカー
        let mut cmd = Command::new(&backend);
        #[cfg(target_os = "windows")]
        cmd.creation_flags(CREATE_NO_WINDOW);
        return cmd
            .spawn()
            .map_err(|e| format!("backend.exe の spawn 失敗: {e}"));
    }

    // backend.exe が無い → 開発モードなら uvicorn をソースから起動
    #[cfg(debug_assertions)]
    {
        spawn_dev_uvicorn()
    }
    #[cfg(not(debug_assertions))]
    {
        Err(format!("backend.exe が見つかりません: {backend:?}"))
    }
}

/// バックエンド起動コマンド。
/// - 既に 8000 が稼働中（例: `python main.py` が起動済み）なら何もしない
/// - 二重起動（React StrictMode の useEffect 二重実行）を防ぐ
#[tauri::command]
fn start_backend(app: tauri::AppHandle) -> Result<String, String> {
    log("=== start_backend called ===");

    let state = app.state::<BackendProcess>();
    // ロックを関数全体で保持し、複数回呼び出しを直列化する（二重 spawn 防止）
    let mut guard = state.0.lock().unwrap();

    if guard.is_some() {
        log("既にバックエンドを起動済み。スキップ");
        return Ok("バックエンドは起動済みです".to_string());
    }

    if port_open(8000) {
        // main.py 等が既に uvicorn を立てている。こちらでは起動しない（所有もしない）
        log("port 8000 は既に稼働中。外部バックエンドに接続します");
        return Ok("既存のバックエンドに接続しました".to_string());
    }

    match spawn_backend() {
        Ok(child) => {
            let msg = format!("バックエンド起動成功 (PID {})", child.id());
            log(&msg);
            *guard = Some(child);
            Ok(msg)
        }
        Err(e) => {
            log(&e);
            Err(e)
        }
    }
}

/// デバッグログファイルのパスを返す
#[tauri::command]
fn get_log_path() -> String {
    std::env::temp_dir()
        .join("vsm_debug.log")
        .to_string_lossy()
        .to_string()
}

/// バックエンド子プロセスをツリーごと終了させる。
/// Windows の .venv python はシム経由で本体を孫として起動するため、
/// child.kill() ではシムしか止まらず孫の uvicorn が残る。taskkill /T で子孫ごと終了する。
fn kill_backend(child: &mut Child) {
    #[cfg(target_os = "windows")]
    {
        let mut kill = Command::new("taskkill");
        kill.args(["/F", "/T", "/PID", &child.id().to_string()]);
        kill.creation_flags(CREATE_NO_WINDOW);
        let _ = kill.spawn().and_then(|mut k| k.wait());
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = child.kill();
    }
    let _ = child.wait();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_http::init())
        .invoke_handler(tauri::generate_handler![start_backend, get_log_path])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            let state = app_handle.state::<BackendProcess>();
            let child_opt = state.0.lock().unwrap().take();
            if let Some(mut child) = child_opt {
                kill_backend(&mut child);
                log("=== アプリ終了時にバックエンドを終了しました ===");
            }
        }
    });
}
