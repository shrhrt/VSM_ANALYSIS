use std::process::Command;

/// バックエンド(FastAPI)をバックグラウンドで起動する。
/// 実行ファイルから親ディレクトリをさかのぼり start_backend.bat を探す。
#[tauri::command]
fn start_backend() -> Result<(), String> {
    let mut dir = std::env::current_exe()
        .map_err(|e| e.to_string())?;

    for _ in 0..10 {
        dir.pop();
        let bat = dir.join("start_backend.bat");
        if bat.exists() {
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::process::CommandExt;
                Command::new("cmd")
                    .args(["/c", "start_backend.bat"])
                    .current_dir(&dir)
                    .creation_flags(0x08000000) // CREATE_NO_WINDOW
                    .spawn()
                    .map_err(|e| format!("起動失敗: {}", e))?;
            }
            #[cfg(not(target_os = "windows"))]
            {
                Command::new("bash")
                    .args(["-c", "./start_backend.sh"])
                    .current_dir(&dir)
                    .spawn()
                    .map_err(|e| format!("起動失敗: {}", e))?;
            }
            return Ok(());
        }
    }
    Err("start_backend.bat が見つかりません".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![start_backend])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
