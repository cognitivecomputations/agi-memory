import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import manifest from "./manifest";

describe("PWA manifest", () => {
  it("offers a standalone install with PNG and maskable icons", () => {
    const value = manifest();
    expect(value.start_url).toBe("/chat?source=pwa");
    expect(value.display).toBe("standalone");
    expect(value.display_override).toEqual(["window-controls-overlay", "standalone"]);
    expect(value.edge_side_panel.preferred_width).toBe(420);
    expect(value.icons).toEqual(expect.arrayContaining([
      expect.objectContaining({ src: "/pwa-icon/192", sizes: "192x192", type: "image/png" }),
      expect.objectContaining({ src: "/pwa-icon/512", sizes: "512x512", purpose: "maskable" }),
    ]));
  });

  it("ships push and notification-click handlers in the service worker", () => {
    const worker = readFileSync(join(process.cwd(), "public", "sw.js"), "utf8");
    expect(worker).toContain('addEventListener("push"');
    expect(worker).toContain('addEventListener("notificationclick"');
    expect(worker).toContain("showNotification");
    expect(worker).toContain("openWindow");
  });
});
