export type InstallPromptEvent = Event & {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed"; platform: string }>;
};

let deferredInstallPrompt: InstallPromptEvent | null = null;

export function rememberInstallPrompt(event: InstallPromptEvent): void {
  deferredInstallPrompt = event;
  window.dispatchEvent(new Event("hexis:install-available"));
}

export function installPromptAvailable(): boolean {
  return deferredInstallPrompt !== null;
}

export async function promptInstall(): Promise<"accepted" | "dismissed" | "unavailable"> {
  const prompt = deferredInstallPrompt;
  if (!prompt) return "unavailable";
  await prompt.prompt();
  const choice = await prompt.userChoice;
  deferredInstallPrompt = null;
  return choice.outcome;
}

export function displayMode(): string {
  if (typeof window === "undefined") return "browser";
  if (window.matchMedia("(display-mode: window-controls-overlay)").matches) {
    return "window-controls-overlay";
  }
  if (window.matchMedia("(display-mode: standalone)").matches) return "standalone";
  if (window.matchMedia("(display-mode: fullscreen)").matches) return "fullscreen";
  if (window.matchMedia("(display-mode: minimal-ui)").matches) return "minimal-ui";
  return "browser";
}

export function isInstalled(): boolean {
  return displayMode() !== "browser" || (navigator as Navigator & { standalone?: boolean }).standalone === true;
}

export function pwaDeviceId(): string {
  const key = "hexis:pwa-device-id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const value = crypto.randomUUID ? crypto.randomUUID() : `pwa-${Date.now()}-${Math.random()}`;
  window.localStorage.setItem(key, value);
  return value;
}

export async function registerHexisServiceWorker(): Promise<ServiceWorkerRegistration> {
  if (!("serviceWorker" in navigator)) {
    throw new Error("This browser does not support service workers.");
  }
  if (!window.isSecureContext) {
    throw new Error(
      "App install, notifications, and microphone capture require HTTPS on another device. Use the documented Tailscale HTTPS setup, then reopen this page.",
    );
  }
  return navigator.serviceWorker.register("/sw.js", { scope: "/" });
}

export async function currentPushSubscription(): Promise<PushSubscription | null> {
  if (!("serviceWorker" in navigator) || !window.isSecureContext) return null;
  const registration = await navigator.serviceWorker.getRegistration("/");
  return registration ? registration.pushManager.getSubscription() : null;
}

export async function enablePushNotifications(): Promise<PushSubscription> {
  if (!("Notification" in window) || !("PushManager" in window)) {
    throw new Error("This browser does not support Web Push notifications.");
  }
  const registration = await registerHexisServiceWorker();
  const configResponse = await fetch("/api/pwa/push/config", { cache: "no-store" });
  const config = await readJson(configResponse);
  if (!configResponse.ok) throw new Error(apiError(config, "Notification setup failed."));
  if (config.enabled !== true || typeof config.public_key !== "string") {
    throw new Error("Web Push is disabled on this Hexis host.");
  }

  const permission = await Notification.requestPermission();
  if (permission !== "granted") {
    throw new Error(
      permission === "denied"
        ? "Notifications are blocked in this browser. Allow them in the site settings, then retry."
        : "Notification permission was not granted.",
    );
  }

  let subscription = await registration.pushManager.getSubscription();
  let created = false;
  if (!subscription) {
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToArrayBuffer(config.public_key),
    });
    created = true;
  }
  const serialized = subscription.toJSON();
  const response = await fetch("/api/pwa/push/subscriptions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      endpoint: subscription.endpoint,
      expirationTime: subscription.expirationTime,
      keys: serialized.keys,
      installed: isInstalled(),
      display_mode: displayMode(),
    }),
  });
  const body = await readJson(response);
  if (!response.ok) {
    if (created) await subscription.unsubscribe().catch(() => false);
    throw new Error(apiError(body, "The browser subscription could not be saved."));
  }
  return subscription;
}

export async function disablePushNotifications(): Promise<boolean> {
  const subscription = await currentPushSubscription();
  if (!subscription) return false;
  let serverError: string | null = null;
  try {
    const response = await fetch("/api/pwa/push/subscriptions", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpoint: subscription.endpoint }),
    });
    const body = await readJson(response);
    if (!response.ok) serverError = apiError(body, "Server revocation failed.");
  } catch (error: unknown) {
    serverError = error instanceof Error ? error.message : "Server revocation failed.";
  }
  const removed = await subscription.unsubscribe();
  if (serverError) {
    throw new Error(
      `Notifications are off in this browser. Hexis could not immediately clean up the server record (${serverError}); the invalid endpoint will be revoked after its next failed delivery.`,
    );
  }
  return removed;
}

function urlBase64ToArrayBuffer(value: string): ArrayBuffer {
  const padding = "=".repeat((4 - (value.length % 4)) % 4);
  const base64 = (value + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = window.atob(base64);
  const bytes = new Uint8Array(new ArrayBuffer(raw.length));
  for (let index = 0; index < raw.length; index += 1) {
    bytes[index] = raw.charCodeAt(index);
  }
  return bytes.buffer;
}

async function readJson(response: Response): Promise<Record<string, unknown>> {
  try {
    const body = await response.json();
    return body && typeof body === "object" ? body as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

function apiError(body: Record<string, unknown>, fallback: string): string {
  return String(body.error || body.detail || fallback);
}
