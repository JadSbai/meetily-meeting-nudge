//! Meeting auto-detection.
//!
//! Reliable signal: a known meeting app is *actively capturing the microphone*
//! (`Process::is_running_input()` == true). That means you have actually joined
//! a call — far more precise than "some app is producing sound". Mirrors
//! `system_detector::list_system_audio_using_apps`, swapping output → input.
//!
//! We only read CoreAudio HAL metadata (which process holds the input device);
//! we never tap or record the audio, so no extra permission is required beyond
//! what Meetily already has.

use super::system_detector::BackgroundTask;

#[cfg(target_os = "macos")]
use cidre::core_audio as ca;

/// A meeting platform inferred from the app holding the microphone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeetingPlatform {
    Zoom,
    Teams,
    GoogleMeet,
    Webex,
    Slack,
    Discord,
    /// A browser-based call we can't attribute more precisely.
    Browser,
}

impl MeetingPlatform {
    /// Human-facing label shown in the nudge ("… you're in a **Zoom** meeting").
    pub fn label(self) -> &'static str {
        match self {
            MeetingPlatform::Zoom => "Zoom",
            MeetingPlatform::Teams => "Teams",
            MeetingPlatform::GoogleMeet => "Google Meet",
            MeetingPlatform::Webex => "Webex",
            MeetingPlatform::Slack => "Slack",
            MeetingPlatform::Discord => "Discord",
            MeetingPlatform::Browser => "a video call",
        }
    }
}

/// Classify a macOS app's localized name into a meeting platform.
///
/// Native meeting apps are unambiguous. Browsers are treated as a generic call
/// *only because they hold the mic* — that is nearly always a live call
/// (Meet / Teams-web / Whereby). A false positive just means one dismissible
/// nudge, which is an acceptable trade for never missing a real meeting.
pub fn classify_app(name: &str) -> Option<MeetingPlatform> {
    let n = name.to_lowercase();

    // Native meeting clients first (most specific).
    if n.contains("zoom") {
        return Some(MeetingPlatform::Zoom);
    }
    if n.contains("microsoft teams") || n == "teams" {
        return Some(MeetingPlatform::Teams);
    }
    if n.contains("webex") || n.contains("cisco") {
        return Some(MeetingPlatform::Webex);
    }
    if n.contains("google meet") || n == "meet" {
        return Some(MeetingPlatform::GoogleMeet);
    }
    if n.contains("slack") {
        return Some(MeetingPlatform::Slack);
    }
    if n.contains("discord") {
        return Some(MeetingPlatform::Discord);
    }

    // Browsers — mic-in-use implies a browser call.
    const BROWSERS: &[&str] = &[
        "google chrome",
        "chromium",
        "brave browser",
        "microsoft edge",
        "arc",
        "safari",
        "firefox",
        "vivaldi",
        "opera",
    ];
    if BROWSERS.iter().any(|b| n.contains(b)) {
        return Some(MeetingPlatform::Browser);
    }

    None
}

/// Emitted as the meeting state transitions.
#[derive(Debug, Clone)]
pub enum MeetingEvent {
    Started {
        platform: MeetingPlatform,
        app: String,
    },
    Ended,
}

pub type MeetingCallback = std::sync::Arc<dyn Fn(MeetingEvent) + Send + Sync + 'static>;

pub fn new_meeting_callback<F>(f: F) -> MeetingCallback
where
    F: Fn(MeetingEvent) + Send + Sync + 'static,
{
    std::sync::Arc::new(f)
}

/// Snapshot of meeting apps currently holding the microphone.
#[cfg(target_os = "macos")]
fn meeting_apps_using_mic() -> Vec<(String, MeetingPlatform)> {
    let mut out = Vec::new();
    let Ok(processes) = ca::System::processes() else {
        return out;
    };
    for process in processes {
        if !process.is_running_input().unwrap_or(false) {
            continue;
        }
        let Ok(pid) = process.pid() else { continue };
        let Some(app) = cidre::ns::RunningApp::with_pid(pid) else {
            continue;
        };
        let Some(name) = app.localized_name().map(|s| s.to_string()) else {
            continue;
        };
        if let Some(platform) = classify_app(&name) {
            out.push((name, platform));
        }
    }
    out
}

/// Poll interval — cheap CoreAudio metadata read, 2s is responsive without churn.
const POLL: std::time::Duration = std::time::Duration::from_secs(2);
/// Consecutive empty polls before we declare the meeting ended (~4s of grace so
/// a momentary mic release mid-call doesn't fire a spurious "ended").
const ABSENT_POLLS_TO_END: u32 = 2;

/// Watches for meeting apps grabbing the mic and fires `MeetingEvent`s.
#[derive(Default)]
pub struct MeetingDetector {
    background: BackgroundTask,
}

impl MeetingDetector {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(target_os = "macos")]
    pub fn start(&mut self, callback: MeetingCallback) {
        self.background.start(|running, mut stop_rx| {
            Box::pin(async move {
                let mut in_meeting = false;
                let mut absent: u32 = 0;

                loop {
                    tokio::select! {
                        _ = &mut stop_rx => break,
                        _ = tokio::time::sleep(POLL) => {}
                    }
                    if !running.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }

                    // CoreAudio FFI is blocking; keep it off the async worker.
                    let apps = match tokio::task::spawn_blocking(meeting_apps_using_mic).await {
                        Ok(a) => a,
                        Err(_) => continue,
                    };

                    if let Some((name, platform)) = apps.into_iter().next() {
                        absent = 0;
                        if !in_meeting {
                            in_meeting = true;
                            tracing::info!(?platform, app = %name, "meeting detected (mic active)");
                            callback(MeetingEvent::Started {
                                platform,
                                app: name,
                            });
                        }
                    } else if in_meeting {
                        absent += 1;
                        if absent >= ABSENT_POLLS_TO_END {
                            in_meeting = false;
                            absent = 0;
                            tracing::info!("meeting ended (mic released)");
                            callback(MeetingEvent::Ended);
                        }
                    }
                }
            })
        });
    }

    #[cfg(not(target_os = "macos"))]
    pub fn start(&mut self, _callback: MeetingCallback) {
        tracing::warn!("Meeting auto-detection is only supported on macOS");
    }

    pub fn stop(&mut self) {
        self.background.stop();
    }
}

/// Managed-state wrapper so the detector outlives `setup()`.
pub type MeetingDetectorState = std::sync::Arc<std::sync::Mutex<Option<MeetingDetector>>>;

pub fn init_meeting_detector_state() -> MeetingDetectorState {
    std::sync::Arc::new(std::sync::Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_native_meeting_apps() {
        assert_eq!(classify_app("zoom.us"), Some(MeetingPlatform::Zoom));
        assert_eq!(classify_app("Microsoft Teams"), Some(MeetingPlatform::Teams));
        assert_eq!(classify_app("Webex"), Some(MeetingPlatform::Webex));
        assert_eq!(classify_app("Slack"), Some(MeetingPlatform::Slack));
    }

    #[test]
    fn classifies_browsers_as_generic_call() {
        assert_eq!(classify_app("Google Chrome"), Some(MeetingPlatform::Browser));
        assert_eq!(classify_app("Arc"), Some(MeetingPlatform::Browser));
        assert_eq!(classify_app("Safari"), Some(MeetingPlatform::Browser));
    }

    #[test]
    fn ignores_non_meeting_apps() {
        assert_eq!(classify_app("meetily"), None);
        assert_eq!(classify_app("Spotify"), None);
        assert_eq!(classify_app("Music"), None);
        assert_eq!(classify_app("QuickTime Player"), None);
    }
}
