use std::io::Write;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

/// バックエンド子プロセスのハンドル。アプリ終了時にkillするため保持しておく
struct BackendProcess(Mutex<Option<Child>>);

/// 診断ログを %TEMP%\vsm_debug.log に追記する
fn log(msg: &str) {
    let path = std::env::temp_dir().join("vsm_debug.log");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{}", msg);
    }
}

/// バックエンド起動 + ログ出力。エラーメッセージは JS 側に返す。
#[tauri::command]
fn start_backend(app: tauri::AppHandle) -> Result<String, String> {
    log("=== start_backend called ===");

    let current_exe = std::env::current_exe().map_err(|e| {
        let msg = format!("current_exe() 失敗: {e}");
        log(&msg);
        msg
    })?;
    log(&format!("current_exe: {current_exe:?}"));

    let mut exe_dir = current_exe.clone();
    exe_dir.pop();
    log(&format!("exe_dir: {exe_dir:?}"));

    let backend = exe_dir.join("backend.exe");
    log(&format!("backend path: {backend:?}"));
    log(&format!("backend exists: {}", backend.exists()));

    if !backend.exists() {
        let msg = format!("backend.exe が見つかりません: {backend:?}");
        log(&msg);
        return Err(msg);
    }

    #[cfg(target_os = "windows")]
    let spawn_result = {
        use std::os::windows::process::CommandExt;
        Command::new(&backend)
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
            .spawn()
    };
    #[cfg(not(target_os = "windows"))]
    let spawn_result = Command::new(&backend).spawn();

    match spawn_result {
        Ok(child) => {
            let msg = format!("backend.exe 起動成功 (PID {})", child.id());
            log(&msg);

            // 多重起動対策: 前回分が残っていれば先にkillしてから差し替える
            let state = app.state::<BackendProcess>();
            let mut guard = state.0.lock().unwrap();
            if let Some(mut old_child) = guard.take() {
                let _ = old_child.kill();
                log("=== 既存の backend.exe をkillして置き換え ===");
            }
            *guard = Some(child);

            Ok(msg)
        }
        Err(e) => {
            let msg = format!("spawn 失敗: {e}");
            log(&msg);
            Err(msg)
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
                let _ = child.kill();
                log("=== アプリ終了時に backend.exe をkillしました ===");
            }
        }
    });
}
