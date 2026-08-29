"use client";

import { Bell, Download, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import {
  displayMode,
  enablePushNotifications,
  installPromptAvailable,
  isInstalled,
  pwaDeviceId,
  promptInstall,
  registerHexisServiceWorker,
  rememberInstallPrompt,
  type InstallPromptEvent,
} from "@/lib/pwa-client";

const DISMISSED_KEY = "hexis:pwa-prompt-dismissed";

export function PwaClient() {
  const [installAvailable, setInstallAvailable] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const [notificationAvailable, setNotificationAvailable] = useState(false);
  const [busy, setBusy] = useState<"install" | "push" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshPrompt = useCallback(() => {
    const dismissed = window.localStorage.getItem(DISMISSED_KEY) === "true";
    const canNotify =
      window.isSecureContext &&
      "Notification" in window &&
      "PushManager" in window &&
      Notification.permission === "default";
    const canInstall = !isInstalled() && installPromptAvailable();
    setNotificationAvailable(canNotify);
    setInstallAvailable(canInstall);
    setShowPrompt(!dismissed && (canInstall || canNotify));
  }, []);

  useEffect(() => {
    const onInstallPrompt = (rawEvent: Event) => {
      const event = rawEvent as InstallPromptEvent;
      event.preventDefault();
      rememberInstallPrompt(event);
      refreshPrompt();
    };
    const onInstalled = () => refreshPrompt();
    window.addEventListener("beforeinstallprompt", onInstallPrompt);
    window.addEventListener("appinstalled", onInstalled);
    window.addEventListener("hexis:install-available", refreshPrompt);
    refreshPrompt();
    if (window.isSecureContext && "serviceWorker" in navigator) {
      registerHexisServiceWorker().catch(() => {
        // The settings App tab carries the visible diagnostic and recovery.
      });
    }
    return () => {
      window.removeEventListener("beforeinstallprompt", onInstallPrompt);
      window.removeEventListener("appinstalled", onInstalled);
      window.removeEventListener("hexis:install-available", refreshPrompt);
    };
  }, [refreshPrompt]);

  useEffect(() => {
    const deviceId = pwaDeviceId();
    const report = (presence: "online" | "offline" | "idle", keepalive = false) => {
      const payload = JSON.stringify({
        device_id: deviceId,
        presence,
        display_mode: displayMode(),
        visibility: document.visibilityState === "visible" ? "visible" : "hidden",
      });
      if (keepalive && "sendBeacon" in navigator) {
        navigator.sendBeacon(
          "/api/pwa/presence",
          new Blob([payload], { type: "application/json" }),
        );
        return;
      }
      fetch("/api/pwa/presence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
        keepalive,
      }).catch(() => undefined);
    };
    const reportVisibility = () => report(document.visibilityState === "visible" ? "online" : "idle");
    const reportOffline = () => report("offline", true);
    const timer = window.setInterval(reportVisibility, 30_000);
    document.addEventListener("visibilitychange", reportVisibility);
    window.addEventListener("pagehide", reportOffline, { once: true });
    reportVisibility();
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", reportVisibility);
      window.removeEventListener("pagehide", reportOffline);
    };
  }, []);

  async function install() {
    setBusy("install");
    setError(null);
    try {
      const outcome = await promptInstall();
      if (outcome === "unavailable") {
        setError("Use your browser menu and choose Install app or Add to Home Screen.");
      } else if (outcome === "dismissed") {
        window.localStorage.setItem(DISMISSED_KEY, "true");
      }
      refreshPrompt();
    } finally {
      setBusy(null);
    }
  }

  async function enableNotifications() {
    setBusy("push");
    setError(null);
    try {
      await enablePushNotifications();
      refreshPrompt();
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Notifications could not be enabled.");
    } finally {
      setBusy(null);
    }
  }

  function dismiss() {
    window.localStorage.setItem(DISMISSED_KEY, "true");
    setShowPrompt(false);
  }

  if (!showPrompt) return null;
  return (
    <aside
      aria-label="Install Hexis app"
      className="fixed bottom-4 right-4 z-50 w-[min(24rem,calc(100vw-2rem))] rounded-xl border border-[var(--outline)] bg-white p-4 shadow-xl"
    >
      <button
        type="button"
        aria-label="Dismiss app setup"
        onClick={dismiss}
        className="absolute right-3 top-3 rounded-md p-1 text-[var(--ink-soft)] hover:bg-[var(--surface-strong)]"
      >
        <X size={16} />
      </button>
      <p className="pr-7 text-sm font-semibold">Keep Hexis close</p>
      <p className="mt-1 text-xs leading-5 text-[var(--ink-soft)]">
        Install the app or let your agent notify you. Message previews stay off by default.
      </p>
      {error ? <p role="alert" className="mt-3 text-xs text-red-700">{error}</p> : null}
      <div className="mt-3 flex flex-wrap gap-2">
        {installAvailable ? (
          <button type="button" disabled={busy !== null} onClick={install} className="inline-flex items-center gap-2 rounded-md bg-[var(--foreground)] px-3 py-2 text-xs font-semibold text-white disabled:opacity-50">
            <Download size={14} /> {busy === "install" ? "Opening..." : "Install app"}
          </button>
        ) : null}
        {notificationAvailable ? (
          <button type="button" disabled={busy !== null} onClick={enableNotifications} className="inline-flex items-center gap-2 rounded-md border border-[var(--outline)] px-3 py-2 text-xs font-semibold disabled:opacity-50">
            <Bell size={14} /> {busy === "push" ? "Enabling..." : "Enable notifications"}
          </button>
        ) : null}
      </div>
    </aside>
  );
}
