//! Meeting-detection nudge.
//!
//! When [`crate::audio::meeting_detector`] sees a meeting app grab the mic, we
//! pop a small always-on-top overlay ("Looks like you're in a Zoom meeting —
//! start recording?"). The overlay never forks a second recording path: its
//! "Start recording" button drives the *exact* same flow the tray uses
//! (focus main window → set `autoStartRecording` → navigate `/`), which
//! `useRecordingStart` picks up.

use std::sync::atomic::{AtomicBool, Ordering};

use tauri::{AppHandle, Manager, Runtime, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_store::StoreExt;

/// Label of the overlay window (also used in the tauri.conf capability).
pub const NUDGE_LABEL: &str = "meeting-nudge";
const STORE_FILE: &str = "meeting-nudge.json";
const STORE_KEY_ENABLED: &str = "enabled";

/// Per-session nudge state. `dismissed` guards against re-nudging the *same*
/// meeting after the user clicked Start or Ignore; it resets when the meeting
/// ends so the next meeting nudges again.
#[derive(Default)]
pub struct NudgeState {
    dismissed: AtomicBool,
}

/// Whether the feature is enabled (default: on).
pub fn is_enabled<R: Runtime>(app: &AppHandle<R>) -> bool {
    match app.store(STORE_FILE) {
        Ok(store) => store
            .get(STORE_KEY_ENABLED)
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        Err(_) => true,
    }
}

fn close_overlay<R: Runtime>(app: &AppHandle<R>) {
    if let Some(win) = app.get_webview_window(NUDGE_LABEL) {
        let _ = win.close();
    }
}

/// A meeting just started: show the overlay unless the feature is off, we're
/// already recording, the user dismissed this meeting, or it's already open.
pub fn handle_started<R: Runtime>(app: &AppHandle<R>, platform_label: String) {
    if !is_enabled(app) {
        return;
    }
    if app.state::<NudgeState>().dismissed.load(Ordering::SeqCst) {
        return;
    }
    if app.get_webview_window(NUDGE_LABEL).is_some() {
        return;
    }

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        // Don't nudge if a recording is already running.
        if crate::is_recording().await {
            return;
        }
        let app_for_main = app.clone();
        // Window creation must happen on the main thread (macOS).
        let _ = app.run_on_main_thread(move || {
            if let Err(e) = show_overlay(&app_for_main, &platform_label) {
                log::error!("Failed to show meeting nudge overlay: {}", e);
            }
        });
    });
}

/// The meeting ended: allow future nudges again and dismiss any open overlay.
pub fn handle_ended<R: Runtime>(app: &AppHandle<R>) {
    app.state::<NudgeState>()
        .dismissed
        .store(false, Ordering::SeqCst);
    close_overlay(app);
}

fn show_overlay<R: Runtime>(app: &AppHandle<R>, platform_label: &str) -> tauri::Result<()> {
    // Percent-encode spaces so the query survives the URL round-trip.
    let encoded = platform_label.replace(' ', "%20");
    let url = format!("{}.html?platform={}", NUDGE_LABEL, encoded);

    let win = WebviewWindowBuilder::new(app, NUDGE_LABEL, WebviewUrl::App(url.into()))
        .title("Meeting detected")
        .inner_size(430.0, 132.0)
        .resizable(false)
        .minimizable(false)
        .maximizable(false)
        .decorations(false)
        .transparent(true)
        .shadow(true)
        .always_on_top(true)
        .skip_taskbar(true)
        .visible(false)
        .build()?;

    // Position top-center of the primary monitor, just below the menu bar.
    if let Ok(Some(monitor)) = app.primary_monitor() {
        let scale = monitor.scale_factor();
        let screen_w = monitor.size().width as f64 / scale;
        let origin_x = monitor.position().x as f64 / scale;
        let x = origin_x + (screen_w - 430.0) / 2.0;
        let _ = win.set_position(tauri::LogicalPosition::new(x, 48.0));
    }

    win.show()?;
    let _ = win.set_focus();
    Ok(())
}

// ---- commands invoked from the overlay -------------------------------------

/// Start recording via the same path the tray uses, then close the overlay.
#[tauri::command]
pub fn nudge_start_recording<R: Runtime>(app: AppHandle<R>) {
    crate::tray::focus_main_window(&app);
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.eval("sessionStorage.setItem('autoStartRecording', 'true')");
        let _ = win.eval("window.location.assign('/')");
    }
    app.state::<NudgeState>()
        .dismissed
        .store(true, Ordering::SeqCst);
    close_overlay(&app);
}

/// Dismiss the nudge for the current meeting.
#[tauri::command]
pub fn nudge_dismiss<R: Runtime>(app: AppHandle<R>) {
    app.state::<NudgeState>()
        .dismissed
        .store(true, Ordering::SeqCst);
    close_overlay(&app);
}

/// Read the enabled toggle (for a settings UI).
#[tauri::command]
pub fn nudge_get_enabled<R: Runtime>(app: AppHandle<R>) -> bool {
    is_enabled(&app)
}

/// Persist the enabled toggle; closes any open overlay when turned off.
#[tauri::command]
pub fn nudge_set_enabled<R: Runtime>(app: AppHandle<R>, enabled: bool) -> Result<(), String> {
    let store = app.store(STORE_FILE).map_err(|e| e.to_string())?;
    store.set(STORE_KEY_ENABLED, serde_json::json!(enabled));
    let _ = store.save();
    if !enabled {
        close_overlay(&app);
    }
    Ok(())
}
