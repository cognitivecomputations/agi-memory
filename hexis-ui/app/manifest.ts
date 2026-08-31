import type { MetadataRoute } from "next";

type HexisManifest = MetadataRoute.Manifest & {
  display_override: string[];
  edge_side_panel: { preferred_width: number };
};

export default function manifest(): HexisManifest {
  return {
    id: "/",
    name: "Hexis",
    short_name: "Hexis",
    description: "A persistent AI identity, available wherever you are.",
    start_url: "/chat?source=pwa",
    scope: "/",
    display: "standalone",
    display_override: ["window-controls-overlay", "standalone"],
    edge_side_panel: { preferred_width: 420 },
    orientation: "any",
    background_color: "#f3f5f2",
    theme_color: "#18211e",
    lang: "en",
    dir: "ltr",
    categories: ["productivity", "utilities"],
    prefer_related_applications: false,
    icons: [
      {
        src: "/pwa-icon/192",
        sizes: "192x192",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/pwa-icon/512",
        sizes: "512x512",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/pwa-icon/512",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
      {
        src: "/hexis-mark.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "any",
      },
      {
        src: "/hexis-mark-maskable.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "maskable",
      },
    ],
    shortcuts: [
      {
        name: "Conversation",
        short_name: "Chat",
        description: "Open your conversation with Hexis",
        url: "/chat?source=shortcut",
        icons: [{ src: "/pwa-icon/192", sizes: "192x192", type: "image/png" }],
      },
    ],
  };
}
