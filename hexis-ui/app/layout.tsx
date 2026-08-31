import type { Metadata, Viewport } from "next";
import { Fraunces, Sora } from "next/font/google";
import "./globals.css";
import { Shell } from "./components/nav/shell";
import { PwaClient } from "./components/pwa/pwa-client";

const displayFont = Fraunces({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-display",
});

const bodyFont = Sora({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: "Hexis",
  description: "Persistent identity for AI.",
  applicationName: "Hexis",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Hexis",
  },
  icons: {
    icon: "/hexis-mark.svg",
    apple: "/pwa-icon/180",
  },
};

export const viewport: Viewport = {
  themeColor: "#18211e",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${displayFont.variable} ${bodyFont.variable} antialiased`}>
        <Shell>{children}</Shell>
        <PwaClient />
      </body>
    </html>
  );
}
