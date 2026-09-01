"use client";

import { Radio, Send, Square, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { pwaDeviceId } from "@/lib/pwa-client";

type TalkPhase = "idle" | "listening" | "transcribing" | "waiting" | "speaking" | "paused";

type VoiceStatus = {
  stt_enabled?: boolean;
  tts_enabled?: boolean;
  talk_enabled?: boolean;
  talk_ready?: boolean;
  provider_ready?: boolean;
  detail?: string;
  error?: string;
  max_utterance_seconds?: number;
};

export function TalkMode({
  disabled,
  onUtterance,
}: {
  disabled?: boolean;
  onUtterance: (transcript: string) => Promise<string | null>;
}) {
  const [phase, setPhase] = useState<TalkPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [heardSpeech, setHeardSpeech] = useState(false);
  const meterRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef(false);
  const mountedRef = useRef(true);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const heardSpeechRef = useRef(false);
  const useSegmentRef = useRef(false);
  const speechStartedRef = useRef(0);
  const silenceStartedRef = useRef(0);
  const listeningStartedRef = useRef(0);
  const maxUtteranceRef = useRef(60);
  const playingRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const cleanupCapture = useCallback(() => {
    if (animationRef.current !== null) window.cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    audioContextRef.current?.close().catch(() => undefined);
    audioContextRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;
    if (meterRef.current) meterRef.current.style.transform = "scaleX(0.03)";
  }, []);

  const cleanupPlayback = useCallback(() => {
    const audio = playingRef.current;
    if (audio) {
      audio.pause();
      audio.src = "";
    }
    playingRef.current = null;
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
  }, []);

  const stopMode = useCallback((message?: string) => {
    activeRef.current = false;
    useSegmentRef.current = false;
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
    cleanupCapture();
    cleanupPlayback();
    chunksRef.current = [];
    if (mountedRef.current) {
      setPhase("idle");
      if (message) setError(message);
    }
  }, [cleanupCapture, cleanupPlayback]);

  useEffect(() => {
    mountedRef.current = true;
    const onVisibility = () => {
      if (document.hidden && activeRef.current) {
        stopMode("Talk mode stopped when this page left the foreground. Press Start Talk mode when you are ready to listen again.");
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      mountedRef.current = false;
      document.removeEventListener("visibilitychange", onVisibility);
      stopMode();
    };
  }, [stopMode]);

  const playReply = useCallback(async (reply: string) => {
    setPhase("speaking");
    const spokenReply = plainTextForSpeech(reply);
    if (!spokenReply) {
      throw new Error("This response has no readable prose to speak. The written response is still available.");
    }
    const response = await fetch("/api/voice/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: spokenReply }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(String(payload.detail || payload.error || "Speech synthesis failed. The written response is still available."));
    }
    if (!activeRef.current) return;
    const blob = await response.blob();
    if (!blob.size) throw new Error("Speech synthesis returned empty audio. The written response is still available.");
    cleanupPlayback();
    const objectUrl = URL.createObjectURL(blob);
    objectUrlRef.current = objectUrl;
    const audio = new Audio(objectUrl);
    playingRef.current = audio;
    try {
      await new Promise<void>((resolve, reject) => {
        audio.onended = () => resolve();
        audio.onerror = () => reject(new Error("The browser could not play the spoken response. The written response is still available."));
        audio.play().catch(reject);
      });
    } finally {
      cleanupPlayback();
    }
  }, [cleanupPlayback]);

  const processSegment = useCallback(async (mimeType: string, chunks: Blob[]) => {
    const blob = new Blob(chunks, { type: mimeType });
    if (!blob.size) throw new Error("The recording was empty. Press Resume and speak after Listening appears.");
    const extension = mimeType.includes("mp4") || mimeType.includes("m4a") ? "m4a" : mimeType.includes("ogg") ? "ogg" : "webm";
    const form = new FormData();
    form.append("file", blob, `talk-utterance.${extension}`);
    form.append("device_id", pwaDeviceId());
    const response = await fetch("/api/voice/transcribe", { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(String(payload.detail || payload.error || "Talk-mode transcription failed."));
    const text = String(payload.transcript || "").trim();
    if (!text) throw new Error("No speech was detected. Press Resume and try again.");
    setTranscript(text);
    setPhase("waiting");
    const reply = await onUtterance(text);
    if (!reply?.trim()) throw new Error("Hexis did not finish a response. The transcript remains in the conversation; press Resume to continue.");
    if (!activeRef.current) return;
    await playReply(reply);
  }, [onUtterance, playReply]);

  const startListening = useCallback(async () => {
    if (!activeRef.current) return;
    cleanupCapture();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    if (!activeRef.current) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    const mimeType = preferredMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    streamRef.current = stream;
    recorderRef.current = recorder;
    chunksRef.current = [];
    heardSpeechRef.current = false;
    setHeardSpeech(false);
    useSegmentRef.current = false;
    speechStartedRef.current = 0;
    silenceStartedRef.current = 0;
    listeningStartedRef.current = performance.now();
    recorder.ondataavailable = (event) => {
      if (!event.data.size) return;
      chunksRef.current.push(event.data);
      if (!heardSpeechRef.current && chunksRef.current.length > 3) chunksRef.current.shift();
    };
    recorder.onerror = () => {
      cleanupCapture();
      activeRef.current = false;
      if (mountedRef.current) {
        setPhase("paused");
        setError("The browser stopped recording unexpectedly. Nothing was uploaded; press Resume when ready.");
      }
    };
    recorder.onstop = () => {
      const useSegment = useSegmentRef.current && heardSpeechRef.current;
      const chunks = chunksRef.current;
      chunksRef.current = [];
      cleanupCapture();
      if (!useSegment || !activeRef.current) return;
      setPhase("transcribing");
      void processSegment(recorder.mimeType || mimeType || "audio/webm", chunks)
        .then(async () => {
          if (activeRef.current) await startListening();
        })
        .catch((processError: unknown) => {
          if (!mountedRef.current) return;
          setError(processError instanceof Error ? processError.message : "Talk mode stopped unexpectedly.");
          setPhase("paused");
        });
    };
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.65;
    context.createMediaStreamSource(stream).connect(analyser);
    audioContextRef.current = context;
    const values = new Uint8Array(analyser.fftSize);
    let noiseFloor = 0.008;
    let calibrationSamples = 0;
    const draw = () => {
      if (!activeRef.current || recorder.state === "inactive") return;
      analyser.getByteTimeDomainData(values);
      let squares = 0;
      for (const value of values) {
        const normalized = (value - 128) / 128;
        squares += normalized * normalized;
      }
      const rms = Math.sqrt(squares / values.length);
      const now = performance.now();
      if (!heardSpeechRef.current && now - listeningStartedRef.current < 700) {
        noiseFloor = (noiseFloor * calibrationSamples + rms) / (calibrationSamples + 1);
        calibrationSamples += 1;
      }
      const threshold = Math.max(0.018, noiseFloor * 2.8);
      if (meterRef.current) meterRef.current.style.transform = `scaleX(${Math.max(0.03, Math.min(1, rms * 8))})`;
      if (rms >= threshold) {
        if (!heardSpeechRef.current) {
          speechStartedRef.current = now;
          setHeardSpeech(true);
        }
        heardSpeechRef.current = true;
        silenceStartedRef.current = 0;
      } else if (heardSpeechRef.current) {
        if (!silenceStartedRef.current) silenceStartedRef.current = now;
        const speechElapsed = now - speechStartedRef.current;
        if (speechElapsed >= 450 && now - silenceStartedRef.current >= 1100) {
          useSegmentRef.current = true;
          recorder.stop();
          return;
        }
      }
      if (heardSpeechRef.current && now - speechStartedRef.current >= maxUtteranceRef.current * 1000) {
        useSegmentRef.current = true;
        recorder.stop();
        return;
      }
      animationRef.current = window.requestAnimationFrame(draw);
    };
    setTranscript("");
    setError(null);
    setPhase("listening");
    recorder.start(400);
    draw();
  }, [cleanupCapture, processSegment]);

  async function startMode() {
    setError(null);
    if (!window.isSecureContext) {
      setError("Talk mode needs HTTPS on another device. Run hexis tunnel start, then reopen the private URL.");
      return;
    }
    if (!("MediaRecorder" in window) || !navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not support foreground Talk mode.");
      return;
    }
    try {
      const response = await fetch("/api/voice/status", { cache: "no-store" });
      const status = await response.json() as VoiceStatus;
      if (!response.ok) throw new Error(String(status.detail || status.error || "Voice status is unavailable."));
      if (!status.stt_enabled) throw new Error("Voice transcription is off. Open Settings → Voice, choose a transcription provider, and save it.");
      if (!status.tts_enabled) throw new Error("Speech output is off. Open Settings → Voice, enable local speech output, and save it.");
      if (!status.talk_enabled) throw new Error("Talk mode is off. Open Settings → Voice, allow foreground Talk mode, and save it.");
      if (!status.provider_ready) throw new Error(String(status.detail || "The local voice sidecar is unavailable. Run hexis voice setup, then retry."));
      maxUtteranceRef.current = Math.max(5, Number(status.max_utterance_seconds || 60));
      activeRef.current = true;
      await startListening();
    } catch (startError: unknown) {
      activeRef.current = false;
      cleanupCapture();
      setPhase("idle");
      setError(startError instanceof Error ? startError.message : "Talk mode could not start.");
    }
  }

  function finishNow() {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive" || !heardSpeechRef.current) return;
    useSegmentRef.current = true;
    recorder.stop();
  }

  function resumeMode() {
    setError(null);
    activeRef.current = true;
    void startListening();
  }

  const phaseLabel: Record<TalkPhase, string> = {
    idle: "Talk mode off",
    listening: "Listening in this foreground tab",
    transcribing: "Transcribing your utterance",
    waiting: "Waiting for Hexis",
    speaking: "Playing the response",
    paused: "Talk mode paused",
  };

  return (
    <div className="relative flex-none">
      {phase === "idle" ? (
        <button type="button" disabled={disabled} onClick={() => void startMode()} aria-label="Start Talk mode" title="Start foreground Talk mode" className="flex h-10 items-center gap-1.5 rounded-md px-2 text-xs font-medium text-[var(--ink-soft)] hover:bg-[var(--outline)] hover:text-[var(--foreground)] disabled:opacity-40"><Radio size={16} /> Talk</button>
      ) : (
        <button type="button" onClick={() => stopMode()} aria-label="Stop Talk mode" title="Stop Talk mode" className="flex h-10 items-center gap-1.5 rounded-md bg-red-50 px-2 text-xs font-semibold text-red-700"><Square size={14} /> Stop</button>
      )}
      {phase !== "idle" ? <div className="absolute bottom-12 left-0 z-30 w-80 rounded-lg border border-[var(--outline)] bg-white p-3 shadow-xl"><div className="flex items-center justify-between gap-3"><div className="min-w-0 flex-1"><p className="text-xs font-semibold">{phaseLabel[phase]}</p><div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[var(--surface-strong)]"><div ref={meterRef} className="h-full origin-left rounded-full bg-[var(--accent)]" style={{ transform: "scaleX(0.03)" }} /></div></div>{phase === "listening" ? <button type="button" onClick={finishNow} disabled={!heardSpeech} className="flex h-8 items-center gap-1 rounded-md border border-[var(--outline)] px-2 text-xs disabled:opacity-40"><Send size={12} /> Send now</button> : null}</div>{transcript ? <p className="mt-2 line-clamp-2 text-xs text-[var(--ink-soft)]">You said: {transcript}</p> : null}<p className="mt-2 text-[11px] text-[var(--ink-soft)]">The microphone stops while Hexis thinks and speaks. Leaving this tab stops Talk mode.</p></div> : null}
      {phase === "paused" && !error ? <button type="button" onClick={resumeMode} className="absolute bottom-12 left-0 z-40 rounded-md bg-[var(--foreground)] px-3 py-2 text-xs font-semibold text-white">Resume Talk mode</button> : null}
      {error ? <div role="alert" className="absolute bottom-12 left-0 z-50 w-80 rounded-lg border border-red-200 bg-white p-3 text-xs text-red-700 shadow-xl"><button type="button" aria-label="Dismiss Talk mode error" onClick={() => setError(null)} className="float-right ml-2 rounded p-1 hover:bg-red-50"><X size={14} /></button><p>{error}</p>{phase === "paused" ? <button type="button" onClick={resumeMode} className="mt-3 rounded-md bg-[var(--foreground)] px-3 py-2 text-xs font-semibold text-white">Resume Talk mode</button> : null}</div> : null}
    </div>
  );
}

function preferredMimeType(): string {
  for (const value of ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus", "audio/webm"]) {
    if (MediaRecorder.isTypeSupported(value)) return value;
  }
  return "";
}

function plainTextForSpeech(value: string): string {
  return value
    .replace(/```[\s\S]*?```/g, " Code omitted from spoken response. ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/[*_~]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}
