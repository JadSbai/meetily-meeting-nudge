// Overlay logic for the meeting-detection nudge window.
// Loaded as an external file because the app CSP blocks inline scripts.
// Uses the global Tauri API (withGlobalTauri: true).

(function () {
  "use strict";

  function invoke(cmd) {
    // Rust owns closing the overlay window, so a failed invoke still needs a
    // client-side fallback close to avoid a stuck overlay.
    try {
      return window.__TAURI__.core.invoke(cmd);
    } catch (e) {
      console.error("nudge invoke failed", cmd, e);
      return Promise.reject(e);
    }
  }

  function closeSelf() {
    try {
      window.__TAURI__.window.getCurrentWindow().close();
    } catch (e) {
      /* Rust already closed us */
    }
  }

  // Fill in the detected platform from the query string.
  var platform = new URLSearchParams(window.location.search).get("platform");
  if (platform) {
    document.getElementById("platform").textContent =
      platform === "a video call" ? "a video call" : "a " + platform + " meeting";
  }

  document.getElementById("start").addEventListener("click", function () {
    invoke("nudge_start_recording").catch(closeSelf);
  });

  document.getElementById("ignore").addEventListener("click", function () {
    invoke("nudge_dismiss").catch(closeSelf);
  });

  // Esc dismisses.
  window.addEventListener("keydown", function (e) {
    if (e.key === "Escape") invoke("nudge_dismiss").catch(closeSelf);
  });

  // Auto-dismiss after 30s so a missed nudge doesn't linger on screen.
  setTimeout(function () {
    invoke("nudge_dismiss").catch(closeSelf);
  }, 30000);
})();
