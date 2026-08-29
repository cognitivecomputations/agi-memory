"use client";

import { Check, Mic, Square, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { pwaDeviceId } from "@/lib/pwa-client";

export function VoiceRecorder({
  disabled,
  onTranscript,
}: {
  disabled?: boolean;
  onTranscript: (transcript: string) => void;
}) {
  const [state, setState] = useState<"idle" | "recording" | "transcribing">("idle");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const useRecordingRef = useRef(false);
  const meterRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationRef = useRef<number | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      useRecordingRef.current = false;
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") recorder.stop();
      cleanupMedia();
    };
  }, []);

  useEffect(() => {
    if (state !== "recording") return;
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [state]);

  async function startRecording() {
    setError(null);
    if (!window.isSecureContext) {
      setError("Microphone capture needs HTTPS on another device. Open Settings → App and follow the Tailscale HTTPS setup.");
      return;
    }
    if (!("MediaRecorder" in window) || !navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not support foreground audio recording.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const mimeType = preferredMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];
      useRecordingRef.current = false;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        cleanupMedia();
        if (mountedRef.current) {
          setState("idle");
          setError("The browser stopped recording unexpectedly. Nothing was uploaded; retry when ready.");
        }
      };
      recorder.onstop = () => finishRecording(recorder.mimeType || mimeType || "audio/webm");
      startMeter(stream);
      setElapsed(0);
      setState("recording");
      recorder.start(500);
    } catch (requestError: unknown) {
      cleanupMedia();
      const name = requestError instanceof DOMException ? requestError.name : "";
      setError(
        name === "NotAllowedError"
          ? "Microphone access was not granted. Allow it in this site's browser settings, then retry."
          : requestError instanceof Error
            ? requestError.message
            : "The microphone could not be opened.",
      );
    }
  }

  function stopRecording(useRecording: boolean) {
    useRecordingRef.current = useRecording;
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    if (useRecording) setState("transcribing");
    recorder.stop();
  }

  async function finishRecording(mimeType: string) {
    const shouldUse = useRecordingRef.current;
    const chunks = chunksRef.current;
    chunksRef.current = [];
    cleanupMedia();
    if (!shouldUse) {
      if (mountedRef.current) setState("idle");
      return;
    }
    const blob = new Blob(chunks, { type: mimeType });
    if (!blob.size) {
      if (mountedRef.current) {
        setState("idle");
        setError("The recording was empty. Retry and speak after the meter starts moving.");
      }
      return;
    }
    const extension = mimeType.includes("mp4") || mimeType.includes("m4a") ? "m4a" : mimeType.includes("ogg") ? "ogg" : "webm";
    const form = new FormData();
    form.append("file", blob, `voice-note.${extension}`);
    form.append("device_id", pwaDeviceId());
    try {
      const response = await fetch("/api/voice/transcribe", { method: "POST", body: form });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(String(body.error || body.detail || "Voice transcription failed."));
      const transcript = String(body.transcript || "").trim();
      if (!transcript) throw new Error("No speech was detected. Retry or type the message.");
      if (mountedRef.current) {
        onTranscript(transcript);
        setState("idle");
        setError(null);
      }
    } catch (requestError: unknown) {
      if (mountedRef.current) {
        setState("idle");
        setError(requestError instanceof Error ? requestError.message : "Voice transcription failed.");
      }
    }
  }

  function startMeter(stream: MediaStream) {
    try {
      const context = new AudioContext();
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.75;
      context.createMediaStreamSource(stream).connect(analyser);
      const values = new Uint8Array(analyser.fftSize);
      audioContextRef.current = context;
      const draw = () => {
        analyser.getByteTimeDomainData(values);
        let squares = 0;
        for (const value of values) {
          const normalized = (value - 128) / 128;
          squares += normalized * normalized;
        }
        const level = Math.min(1, Math.sqrt(squares / values.length) * 5);
        if (meterRef.current) meterRef.current.style.transform = `scaleX(${Math.max(0.03, level)})`;
        animationRef.current = window.requestAnimationFrame(draw);
      };
      draw();
    } catch {
      // Recording still works when the browser cannot construct a level meter.
    }
  }

  function cleanupMedia() {
    if (animationRef.current !== null) window.cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;
    if (meterRef.current) meterRef.current.style.transform = "scaleX(0.03)";
  }

  return (
    <div className="relative flex-none">
      <button
        type="button"
        aria-label={state === "recording" ? "Voice recording in progress" : "Record voice message"}
        title="Record voice message"
        disabled={disabled || state !== "idle"}
        onClick={startRecording}
        className={`flex h-10 w-10 items-center justify-center rounded-md text-[var(--ink-soft)] hover:bg-[var(--outline)] hover:text-[var(--foreground)] disabled:opacity-40 ${state === "recording" ? "text-red-700" : ""}`}
      >
        <Mic size={17} />
      </button>
      {state !== "idle" ? (
        <div className="absolute bottom-12 left-0 z-20 w-72 rounded-lg border border-[var(--outline)] bg-white p-3 shadow-xl">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold">{state === "recording" ? `Recording · ${formatElapsed(elapsed)}` : "Transcribing locally or with your chosen provider..."}</p>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[var(--surface-strong)]"><div ref={meterRef} className="h-full origin-left rounded-full bg-[var(--accent)] transition-transform" style={{ transform: "scaleX(0.03)" }} /></div>
            </div>
            {state === "recording" ? <div className="flex gap-1"><button type="button" onClick={() => stopRecording(false)} aria-label="Cancel recording" title="Cancel and discard" className="flex h-8 w-8 items-center justify-center rounded-md border border-[var(--outline)]"><X size={15} /></button><button type="button" onClick={() => stopRecording(true)} aria-label="Use recording" title="Stop and transcribe" className="flex h-8 w-8 items-center justify-center rounded-md bg-[var(--foreground)] text-white"><Check size={15} /></button></div> : <Square size={16} className="animate-pulse text-[var(--accent)]" />}
          </div>
          <p className="mt-2 text-[11px] text-[var(--ink-soft)]">Nothing is sent until you choose Use recording. Cancel discards it immediately.</p>
        </div>
      ) : null}
      {error ? (
        <div role="alert" className="absolute bottom-12 left-0 z-20 w-80 rounded-lg border border-red-200 bg-white p-3 text-xs text-red-700 shadow-xl">
          <button type="button" aria-label="Dismiss voice error" onClick={() => setError(null)} className="float-right ml-2 rounded p-1 hover:bg-red-50"><X size={14} /></button>
          {error}
        </div>
      ) : null}
    </div>
  );
}

function preferredMimeType(): string {
  for (const value of ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus", "audio/webm"]) {
    if (MediaRecorder.isTypeSupported(value)) return value;
  }
  return "";
}

function formatElapsed(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
