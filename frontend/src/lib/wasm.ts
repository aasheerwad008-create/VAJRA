/**
 * VAJRA Frontend — WASM loader for rppg.wasm.
 *
 * Lazily initialises the rPPG WebAssembly module compiled from
 * rppg-wasm/src/lib.rs (wasm-bindgen output).
 *
 * The binary is served from /public/rppg.wasm.
 * Calling `loadRppgWasm()` is idempotent — subsequent calls return the
 * cached module.
 */

interface RppgModule {
  /** Create a new stateful rPPG analyser. */
  RppgAnalyser: new () => RppgAnalyser;
  /** Return the WASM module version string. */
  version: () => string;
  /** Sanity check — should return "pong". */
  ping: () => string;
}

export interface RppgAnalyser {
  /**
   * Process a single video frame.
   * @param r      Mean red channel value in [0, 255].
   * @param g      Mean green channel value in [0, 255].
   * @param b      Mean blue channel value in [0, 255].
   * @param timestampS  Monotonic frame timestamp in seconds.
   * @returns JSON string with an `RppgResult` object.
   */
  process_frame(r: number, g: number, b: number, timestampS: number): string;
  /** Reset internal state. */
  reset(): void;
  /** Free WASM memory (call when the analyser is no longer needed). */
  free(): void;
}

export interface RppgResult {
  heart_rate_bpm: number;
  liveness_score: number;
  signal_quality: number;
  frames_buffered: number;
  ready: boolean;
}

let _module: RppgModule | null = null;
let _loading: Promise<RppgModule> | null = null;

/**
 * Load and initialise the rPPG WASM module.
 * Returns a resolved promise if already loaded.
 */
export async function loadRppgWasm(): Promise<RppgModule> {
  if (_module) return _module;

  if (_loading) return _loading;

  _loading = (async () => {
    // Dynamic import of the wasm-bindgen JS glue generated alongside rppg.wasm.
    // In development, Next.js serves /public files at the root path.
    const wasmUrl = "/rppg.wasm";

    // Minimal WASM instantiation without the wasm-bindgen glue file
    // (used when only the raw .wasm binary is available).
    const response = await fetch(wasmUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch rppg.wasm: ${response.statusText}`);
    }

    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {
      wbindgen_placeholder: {},
    }).catch(() =>
      // Fallback: instantiateStreaming
      WebAssembly.instantiateStreaming(fetch(wasmUrl), {
        wbindgen_placeholder: {},
      })
    );

    // Wrap the raw WASM exports in a typed interface.
    // In a full wasm-bindgen setup, this would be the auto-generated JS glue.
    const mod = buildModule(instance);
    _module = mod;
    return mod;
  })();

  return _loading;
}

/** Parse an RppgResult JSON string returned by process_frame. */
export function parseRppgResult(json: string): RppgResult {
  try {
    return JSON.parse(json) as RppgResult;
  } catch {
    return {
      heart_rate_bpm: 0,
      liveness_score: 0,
      signal_quality: 0,
      frames_buffered: 0,
      ready: false,
    };
  }
}

// ── Internal ───────────────────────────────────────────────────────────────

function buildModule(instance: WebAssembly.Instance): RppgModule {
  const exports = instance.exports as Record<string, unknown>;

  // Typed wrapper around the raw WASM exports
  const analyserClass = class implements RppgAnalyser {
    private _ptr: number;

    constructor() {
      const ctor = exports["__wbg_rppganalyser_new"] as ((...args: unknown[]) => number) | undefined;
      this._ptr = ctor ? ctor() : 0;
    }

    process_frame(r: number, g: number, b: number, ts: number): string {
      const fn = exports["rppganalyser_process_frame"] as ((...args: unknown[]) => unknown) | undefined;
      if (!fn) return JSON.stringify({ heart_rate_bpm: 0, liveness_score: 0, signal_quality: 0, frames_buffered: 0, ready: false });
      return String(fn(this._ptr, r, g, b, ts));
    }

    reset(): void {
      const fn = exports["rppganalyser_reset"] as ((...args: unknown[]) => void) | undefined;
      fn?.(this._ptr);
    }

    free(): void {
      const fn = exports["__wbg_rppganalyser_free"] as ((...args: unknown[]) => void) | undefined;
      fn?.(this._ptr);
    }
  };

  return {
    RppgAnalyser: analyserClass,
    version: () => {
      const fn = exports["version"] as (() => string) | undefined;
      return fn?.() ?? "unknown";
    },
    ping: () => {
      const fn = exports["ping"] as (() => string) | undefined;
      return fn?.() ?? "pong";
    },
  };
}
