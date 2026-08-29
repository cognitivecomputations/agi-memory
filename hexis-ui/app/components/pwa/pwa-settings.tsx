"use client";

import { Bell, Download, LockKeyhole, Smartphone, WifiOff } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import {
  currentPushSubscription,
  disablePushNotifications,
  displayMode,
  enablePushNotifications,
  installPromptAvailable,
  isInstalled,
  promptInstall,
  registerHexisServiceWorker,
} from "@/lib/pwa-client";

type AppState = {
  secure: boolean;
  installed: boolean;
  displayMode: string;
  serviceWorker: boolean;
  pushSupported: boolean;
  notificationPermission: NotificationPermission | "unsupported";
  subscribed: boolean;
  ios: boolean;
};

export function PwaSettings() {
  const [state, setState] = useState<AppState | null>(null);
  const [busy, setBusy] = useState<"install" | "enable" | "disable" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const secure = window.isSecureContext;
    let serviceWorker = false;
    if (secure && "serviceWorker" in navigator) {
      try {
        await registerHexisServiceWorker();
        serviceWorker = true;
      } catch {
        serviceWorker = false;
      }
    }
    const subscribed = Boolean(await currentPushSubscription().catch(() => null));
    setState({
      secure,
      installed: isInstalled(),
      displayMode: displayMode(),
      serviceWorker,
      pushSupported: "PushManager" in window && "Notification" in window,
      notificationPermission: "Notification" in window ? Notification.permission : "unsupported",
      subscribed,
      ios: /iPad|iPhone|iPod/.test(navigator.userAgent),
    });
  }, []);

  useEffect(() => {
    refresh();
    window.addEventListener("hexis:install-available", refresh);
    return () => window.removeEventListener("hexis:install-available", refresh);
  }, [refresh]);

  async function install() {
    setBusy("install");
    setMessage(null);
    setError(null);
    try {
      const result = await promptInstall();
      if (result === "unavailable") {
        setMessage(
          state?.ios
            ? "In Safari, tap Share, then Add to Home Screen. Open the installed app before enabling notifications."
            : "Open your browser menu and choose Install app. If that option is missing, reload once after the service worker finishes installing.",
        );
      } else if (result === "accepted") {
        setMessage("Hexis was installed on this device.");
      } else {
        setMessage("Installation was cancelled. Nothing changed.");
      }
      await refresh();
    } finally {
      setBusy(null);
    }
  }

  async function enable() {
    setBusy("enable");
    setMessage(null);
    setError(null);
    try {
      await enablePushNotifications();
      setMessage("Notifications are enabled. Message previews remain hidden by default.");
      await refresh();
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Notifications could not be enabled.");
    } finally {
      setBusy(null);
    }
  }

  async function disable() {
    setBusy("disable");
    setMessage(null);
    setError(null);
    try {
      await disablePushNotifications();
      setMessage("Notifications are off for this browser.");
      await refresh();
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Notifications could not be disabled.");
      await refresh();
    } finally {
      setBusy(null);
    }
  }

  if (!state) return <p className="text-sm text-[var(--ink-soft)]">Checking app capabilities...</p>;

  return (
    <section id="app-access" className="space-y-5">
      <div className="rounded-lg border border-[var(--outline)] bg-white p-5">
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-[var(--surface-strong)] text-[var(--teal)]"><Smartphone size={18} /></div>
          <div>
            <h2 className="text-sm font-semibold">Installed app</h2>
            <p className="mt-1 max-w-2xl text-sm text-[var(--ink-soft)]">Use the same Hexis dashboard as a standalone app on desktop, Android, or iOS. No second chat client or account is created.</p>
          </div>
        </div>
        <dl className="mt-5 grid gap-3 text-sm sm:grid-cols-3">
          <Status label="Secure context" value={state.secure ? "HTTPS ready" : "HTTPS required"} good={state.secure} />
          <Status label="Service worker" value={state.serviceWorker ? "Registered" : "Unavailable"} good={state.serviceWorker} />
          <Status label="Display" value={state.installed ? state.displayMode : "Browser tab"} good={state.installed} />
        </dl>
        {!state.secure ? (
          <div className="mt-5 flex gap-3 rounded-md border border-amber-300 bg-amber-50 p-4 text-sm">
            <WifiOff size={18} className="mt-0.5 flex-none" />
            <div><p className="font-semibold">This address is not a secure context.</p><p className="mt-1 text-[var(--ink-soft)]">A phone opened over plain LAN HTTP cannot install the service worker, receive push, or use the microphone. Follow the private Tailscale HTTPS runbook, then reopen the resulting <code>https://…ts.net</code> address.</p></div>
          </div>
        ) : null}
        <button type="button" onClick={install} disabled={busy !== null || state.installed} className="mt-5 inline-flex items-center gap-2 rounded-md bg-[var(--foreground)] px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"><Download size={16} />{state.installed ? "Installed" : busy === "install" ? "Opening..." : installPromptAvailable() ? "Install Hexis" : "Show install steps"}</button>
      </div>

      <div className="rounded-lg border border-[var(--outline)] bg-white p-5">
        <div className="flex items-start gap-3"><div className="flex h-9 w-9 items-center justify-center rounded-md bg-[var(--surface-strong)] text-[var(--teal)]"><Bell size={18} /></div><div><h2 className="text-sm font-semibold">Push notifications</h2><p className="mt-1 max-w-2xl text-sm text-[var(--ink-soft)]">Automation suggestions, questions, and agent messages can arrive when the dashboard is closed. Permission is requested only when you press Enable.</p></div></div>
        <div className="mt-4 flex gap-3 rounded-md bg-[var(--surface-strong)] p-3 text-xs text-[var(--ink-soft)]"><LockKeyhole size={16} className="flex-none" /><p>Lock-screen previews are off by default: notifications say what kind of decision is waiting without showing message content.</p></div>
        <p className="mt-4 text-sm">Browser permission: <span className="font-semibold">{state.notificationPermission}</span> · Server subscription: <span className="font-semibold">{state.subscribed ? "active" : "off"}</span></p>
        {state.ios && !state.installed ? <p className="mt-3 text-sm text-amber-800">On iOS 16.4+, push is available only after Add to Home Screen and opening the installed app.</p> : null}
        {error ? <p role="alert" className="mt-4 text-sm text-red-700">{error}</p> : null}
        {message ? <p role="status" className="mt-4 text-sm text-emerald-700">{message}</p> : null}
        <div className="mt-5 flex flex-wrap gap-2">
          {state.subscribed ? <button type="button" onClick={disable} disabled={busy !== null} className="rounded-md border border-[var(--outline)] px-4 py-2 text-sm font-semibold disabled:opacity-50">{busy === "disable" ? "Turning off..." : "Turn off notifications"}</button> : <button type="button" onClick={enable} disabled={busy !== null || !state.secure || !state.pushSupported} className="rounded-md bg-[var(--foreground)] px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">{busy === "enable" ? "Enabling..." : "Enable notifications"}</button>}
        </div>
      </div>

      <div className="rounded-lg border border-[var(--outline)] bg-white p-5 text-sm text-[var(--ink-soft)]">
        <p className="font-semibold text-[var(--foreground)]">Platform limits</p>
        <p className="mt-2 leading-6">The PWA supports foreground microphone capture and push. It cannot run host commands, listen for a wake word, or record in the background. On iOS there is no Siri integration or Web Share Target, and an uninstalled site&apos;s storage may be evicted after roughly seven unused days.</p>
        <a href="https://github.com/QuixiAI/Hexis/blob/main/docs/operations/secure-remote-access.md" target="_blank" rel="noreferrer" className="mt-3 inline-block font-semibold text-[var(--teal)] hover:underline">Open the secure phone-access runbook</a>
      </div>
    </section>
  );
}

function Status({ label, value, good }: { label: string; value: string; good: boolean }) {
  return <div className="rounded-md border border-[var(--outline)] p-3"><dt className="text-xs text-[var(--ink-soft)]">{label}</dt><dd className={`mt-1 font-semibold ${good ? "text-emerald-700" : "text-amber-800"}`}>{value}</dd></div>;
}
