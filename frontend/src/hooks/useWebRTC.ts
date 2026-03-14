"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface UseWebRTCOptions {
  videoConstraints?: MediaTrackConstraints;
  audioConstraints?: MediaTrackConstraints | boolean;
}

interface UseWebRTCReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  stream: MediaStream | null;
  isActive: boolean;
  error: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
}

/**
 * useWebRTC — WebRTC camera + audio hook.
 *
 * Manages a local MediaStream from the user's camera / microphone.
 * The returned `videoRef` should be attached to a <video> element.
 *
 * Usage:
 *   const { videoRef, startCamera, stopCamera, isActive, error } = useWebRTC();
 *   useEffect(() => { startCamera(); return stopCamera; }, []);
 */
export function useWebRTC({
  videoConstraints = { width: 640, height: 480, facingMode: "user" },
  audioConstraints = false,
}: UseWebRTCOptions = {}): UseWebRTCReturn {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    if (streamRef.current) return; // already running

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: audioConstraints,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setIsActive(true);
      setError(null);
    } catch (err) {
      const msg =
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Camera access denied"
          : "Camera unavailable";
      setError(msg);
      setIsActive(false);
    }
  }, [videoConstraints, audioConstraints]);

  const stopCamera = useCallback(() => {
    if (!streamRef.current) return;

    streamRef.current.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsActive(false);
  }, []);

  // Auto-cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    stream: streamRef.current,
    isActive,
    error,
    startCamera,
    stopCamera,
  };
}
