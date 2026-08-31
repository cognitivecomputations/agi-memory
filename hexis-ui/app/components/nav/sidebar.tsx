"use client";

import {
  Activity,
  BellRing,
  Brain,
  FilePlus2,
  FolderOpen,
  Layers,
  LayoutDashboard,
  MessageCircle,
  Send,
  Plug,
  Settings,
  Target,
  UserRound,
  X,
  GitCompareArrows,
  History,
  BookCheck,
  ArchiveRestore,
} from "lucide-react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useGatewayEvents } from "../../hooks/use-gateway-events";
import { ProgressBar } from "../ui/progress-bar";

type StatusData = {
  agent_name?: string;
  portrait_url?: string | null;
  energy?: number;
  max_energy?: number;
  energy_capacity?: number;
  mood?: string;
  heartbeat_active?: boolean;
  heartbeat_paused?: boolean;
};

const navItems = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/chat", label: "Conversation", icon: MessageCircle },
  { href: "/goals", label: "Goals", icon: Target },
  { href: "/responsibilities", label: "Responsibilities", icon: BellRing },
  { href: "/outbound", label: "Outbound", icon: Send },
  { href: "/memories", label: "Memory", icon: Brain },
  { href: "/memory-history", label: "Memory history", icon: History },
  { href: "/contradictions", label: "Contradictions", icon: GitCompareArrows },
  { href: "/learning-review", label: "Learning review", icon: BookCheck },
  { href: "/forgetting", label: "Forgetting", icon: ArchiveRestore },
  { href: "/user-model", label: "User Model", icon: UserRound },
  { href: "/documents", label: "Documents", icon: FolderOpen },
  { href: "/desk", label: "Desk", icon: Layers },
  { href: "/ingest", label: "Ingest", icon: FilePlus2 },
  { href: "/connections", label: "Connections", icon: Plug },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar({
  open = false,
  onClose,
}: {
  open?: boolean;
  onClose?: () => void;
}) {
  const pathname = usePathname();
  const [status, setStatus] = useState<StatusData>({});

  const loadStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (response.ok) setStatus(await response.json());
    } catch {
      // The active page owns visible connection errors.
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(loadStatus, 0);
    return () => window.clearTimeout(timer);
  }, [loadStatus]);
  useGatewayEvents(loadStatus);

  const heartbeatLabel = status.heartbeat_paused
    ? "Paused"
    : status.heartbeat_active
      ? "Running"
      : "Idle";

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex w-60 flex-col border-r border-[var(--outline)] bg-white transition-transform lg:translate-x-0 ${
        open ? "translate-x-0" : "-translate-x-full"
      }`}
    >
      <div className="flex h-16 items-center gap-3 border-b border-[var(--outline)] px-4">
        {status.portrait_url ? (
          <Image
            src={status.portrait_url}
            alt=""
            width={40}
            height={40}
            unoptimized
            className="h-10 w-10 rounded-md object-cover"
          />
        ) : (
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-[var(--foreground)] font-display text-lg text-white">
            {(status.agent_name || "H").slice(0, 1).toUpperCase()}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="text-[11px] font-semibold uppercase text-[var(--teal)]">Hexis</p>
          <p className="truncate text-sm font-semibold text-[var(--foreground)]">
            {status.agent_name || "Agent"}
          </p>
        </div>
        <button
          type="button"
          aria-label="Close navigation menu"
          onClick={onClose}
          className="flex h-9 w-9 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] lg:hidden"
        >
          <X size={18} />
        </button>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4" aria-label="Primary navigation">
        {navItems.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onClose}
              className={`flex h-10 items-center gap-3 rounded-md px-3 text-sm transition ${
                active
                  ? "bg-[var(--surface-strong)] font-semibold text-[var(--foreground)]"
                  : "text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] hover:text-[var(--foreground)]"
              }`}
            >
              <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="space-y-4 border-t border-[var(--outline)] px-4 py-4">
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-2 text-[var(--ink-soft)]">
            <Activity size={15} aria-hidden="true" />
            Heartbeat
          </span>
          <span className="font-medium text-[var(--foreground)]">{heartbeatLabel}</span>
        </div>
        {status.energy !== undefined ? (
          <ProgressBar
            value={status.energy}
            max={status.energy_capacity || status.max_energy || 20}
            label="Energy"
            color="teal"
          />
        ) : null}
        {status.mood ? (
          <div className="flex items-center justify-between text-xs">
            <span className="text-[var(--ink-soft)]">Mood</span>
            <span className="capitalize text-[var(--foreground)]">{status.mood}</span>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
