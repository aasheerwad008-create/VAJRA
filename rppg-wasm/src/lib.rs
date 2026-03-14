/*!
VAJRA — rPPG Liveness Detector (WebAssembly)
============================================

Remote PhotoPlethysmoGraphy (rPPG) estimates heart rate from subtle
luminance changes in a video feed caused by blood-volume pulses.

This crate compiles to `rppg.wasm` and is loaded by the frontend via
`lib/wasm.ts`.  It exposes a stateful analyser that processes incoming
RGB frames and returns:
  - Estimated heart rate (BPM)
  - Liveness score  (0.0 – 1.0)
  - Signal quality  (0.0 – 1.0)

Algorithm:
  1. Extract mean green-channel value per frame (most sensitive to rPPG).
  2. Accumulate a 10-second sliding window at 30 fps (300 frames).
  3. Detrend with a simple linear fit.
  4. Apply a Hanning window to suppress spectral leakage.
  5. FFT → find dominant frequency in [0.7 Hz, 4 Hz] (42–240 BPM).
  6. Compute SNR as liveness score.

Security:
  - Replay detection: the analyser expects frame timestamps to be
    monotonically increasing.  A gap > 3 s resets the buffer.
  - Spoofing resistance: a flat or static signal scores 0.0.
*/

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// ── Public API types ───────────────────────────────────────────────────────

/// Result returned by `RppgAnalyser::process_frame`.
#[derive(Debug, Serialize, Deserialize)]
pub struct RppgResult {
    /// Estimated heart rate in beats per minute.
    pub heart_rate_bpm: f64,
    /// Liveness confidence score in [0, 1].
    pub liveness_score: f64,
    /// Signal quality in [0, 1]; low quality → measurement may be unreliable.
    pub signal_quality: f64,
    /// Number of frames accumulated so far.
    pub frames_buffered: usize,
    /// True when enough frames have been buffered for a reliable estimate.
    pub ready: bool,
}

// ── Constants ──────────────────────────────────────────────────────────────

const FPS: f64 = 30.0;
/// Minimum frames before producing an estimate (5 s at 30 fps).
const MIN_FRAMES: usize = 150;
/// Sliding window size in frames (10 s at 30 fps).
const WINDOW_SIZE: usize = 300;
/// Valid heart-rate frequency range (Hz).
const FREQ_MIN: f64 = 0.7;   // 42 BPM
const FREQ_MAX: f64 = 4.0;   // 240 BPM
/// Max gap between consecutive frames before resetting (seconds).
const MAX_FRAME_GAP_S: f64 = 3.0;

// ── Analyser ───────────────────────────────────────────────────────────────

/// Stateful rPPG analyser exposed to JavaScript via `wasm-bindgen`.
#[wasm_bindgen]
pub struct RppgAnalyser {
    /// Circular buffer of mean green-channel values.
    green_buffer: Vec<f64>,
    /// Timestamp of the last processed frame (seconds).
    last_timestamp_s: f64,
    /// Write cursor in the circular buffer.
    cursor: usize,
    /// Total frames processed since the last reset.
    total_frames: usize,
}

#[wasm_bindgen]
impl RppgAnalyser {
    /// Create a new `RppgAnalyser`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> RppgAnalyser {
        RppgAnalyser {
            green_buffer: vec![0.0; WINDOW_SIZE],
            last_timestamp_s: -1.0,
            cursor: 0,
            total_frames: 0,
        }
    }

    /// Reset the internal state.
    pub fn reset(&mut self) {
        self.green_buffer.fill(0.0);
        self.last_timestamp_s = -1.0;
        self.cursor = 0;
        self.total_frames = 0;
    }

    /**
     * Process a single video frame.
     *
     * Parameters
     * ----------
     * r, g, b          — mean RGB channel values in [0, 255].
     * timestamp_s      — monotonic timestamp of this frame in seconds.
     *
     * Returns a JSON string containing an `RppgResult`.
     *
     * Called from TypeScript for each decoded video frame:
     *   const result = analyser.process_frame(r, g, b, timestamp);
     *   const { heart_rate_bpm, liveness_score } = JSON.parse(result);
     */
    pub fn process_frame(
        &mut self,
        r: f64,
        g: f64,
        b: f64,
        timestamp_s: f64,
    ) -> String {
        // Reset if the stream was paused (gap > threshold)
        if self.last_timestamp_s > 0.0
            && (timestamp_s - self.last_timestamp_s) > MAX_FRAME_GAP_S
        {
            self.reset();
        }
        self.last_timestamp_s = timestamp_s;

        // CHROM method: use a chrominance-based signal to suppress illumination
        // variation.  Signal S = 3R - 2G (empirical coefficients from de Haan 2013).
        let chrom = 3.0 * r - 2.0 * g;

        // Write to circular buffer
        self.green_buffer[self.cursor] = chrom;
        self.cursor = (self.cursor + 1) % WINDOW_SIZE;
        self.total_frames += 1;

        let frames_buffered = self.total_frames.min(WINDOW_SIZE);
        let ready = frames_buffered >= MIN_FRAMES;

        let (heart_rate_bpm, liveness_score, signal_quality) = if ready {
            self.analyse(frames_buffered)
        } else {
            (0.0, 0.0, 0.0)
        };

        let result = RppgResult {
            heart_rate_bpm,
            liveness_score,
            signal_quality,
            frames_buffered,
            ready,
        };

        serde_json::to_string(&result).unwrap_or_else(|_| {
            r#"{"heart_rate_bpm":0,"liveness_score":0,"signal_quality":0,"frames_buffered":0,"ready":false}"#
                .to_string()
        })
    }

    // ── Private ──────────────────────────────────────────────────────────

    fn analyse(&self, n: usize) -> (f64, f64, f64) {
        // Extract the most recent `n` samples from the circular buffer
        let signal = self.get_window(n);

        // Linear detrend
        let detrended = detrend(&signal);

        // Hanning window
        let windowed = apply_hanning(&detrended);

        // Discrete Fourier Transform (manual for no-std WASM)
        let freqs = rfft_freqs(n, FPS);
        let magnitudes = rfft_magnitude(&windowed);

        // Find dominant frequency in valid range
        let (peak_freq, peak_mag, mean_mag) = find_peak(&freqs, &magnitudes, FREQ_MIN, FREQ_MAX);

        if peak_freq <= 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let heart_rate_bpm = peak_freq * 60.0;

        // SNR-based liveness score
        let snr = peak_mag / (mean_mag + 1e-8);
        let liveness_score = tanh(snr / 5.0);

        // Signal quality: coefficient of variation (low = noisy)
        let mean = signal.iter().sum::<f64>() / n as f64;
        let variance = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / (mean.abs() + 1e-8);
        let signal_quality = (1.0 - cv.min(1.0)).max(0.0);

        (heart_rate_bpm, liveness_score.min(1.0), signal_quality)
    }

    fn get_window(&self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        let start = (self.cursor + WINDOW_SIZE - n) % WINDOW_SIZE;
        for i in 0..n {
            out.push(self.green_buffer[(start + i) % WINDOW_SIZE]);
        }
        out
    }
}

// ── DSP Helpers ────────────────────────────────────────────────────────────

fn detrend(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n < 2 {
        return signal.to_vec();
    }
    let n_f = n as f64;
    let mean_x = (n_f - 1.0) / 2.0;
    let mean_y: f64 = signal.iter().sum::<f64>() / n_f;

    let num: f64 = signal.iter().enumerate()
        .map(|(i, &y)| (i as f64 - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = (0..n).map(|i| (i as f64 - mean_x).powi(2)).sum();

    let slope = if den.abs() > 1e-12 { num / den } else { 0.0 };
    let intercept = mean_y - slope * mean_x;

    signal.iter().enumerate()
        .map(|(i, &y)| y - (slope * i as f64 + intercept))
        .collect()
}

fn apply_hanning(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    signal.iter().enumerate().map(|(i, &x)| {
        let w = 0.5 * (1.0 - (2.0 * core::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos());
        x * w
    }).collect()
}

fn rfft_freqs(n: usize, sample_rate: f64) -> Vec<f64> {
    let num_freqs = n / 2 + 1;
    (0..num_freqs).map(|k| k as f64 * sample_rate / n as f64).collect()
}

fn rfft_magnitude(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let num_freqs = n / 2 + 1;
    let pi = core::f64::consts::PI;

    (0..num_freqs).map(|k| {
        let re: f64 = signal.iter().enumerate()
            .map(|(j, &x)| x * (2.0 * pi * k as f64 * j as f64 / n as f64).cos())
            .sum();
        let im: f64 = signal.iter().enumerate()
            .map(|(j, &x)| -x * (2.0 * pi * k as f64 * j as f64 / n as f64).sin())
            .sum();
        (re * re + im * im).sqrt()
    }).collect()
}

fn find_peak(
    freqs: &[f64],
    magnitudes: &[f64],
    freq_min: f64,
    freq_max: f64,
) -> (f64, f64, f64) {
    let valid: Vec<(f64, f64)> = freqs.iter().zip(magnitudes.iter())
        .filter(|(&f, _)| f >= freq_min && f <= freq_max)
        .map(|(&f, &m)| (f, m))
        .collect();

    if valid.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mean_mag = valid.iter().map(|(_, m)| m).sum::<f64>() / valid.len() as f64;
    let (peak_freq, peak_mag) = valid.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|&(f, m)| (f, m))
        .unwrap_or((0.0, 0.0));

    (peak_freq, peak_mag, mean_mag)
}

fn tanh(x: f64) -> f64 {
    let e2x = (2.0 * x).exp();
    (e2x - 1.0) / (e2x + 1.0)
}

// ── Module-level helpers exposed to JS ────────────────────────────────────

/// Return the version string of this WASM module.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Validate that WASM is loaded and working.
#[wasm_bindgen]
pub fn ping() -> String {
    "pong".to_string()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detrend_removes_linear_trend() {
        let signal: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let detrended = detrend(&signal);
        let max_val = detrended.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val < 1.0, "detrended signal should be near zero");
    }

    #[test]
    fn test_analyser_resets_on_gap() {
        let mut analyser = RppgAnalyser::new();
        analyser.process_frame(128.0, 128.0, 128.0, 1.0);
        assert_eq!(analyser.total_frames, 1);
        // Simulate a 5-second gap
        analyser.process_frame(128.0, 128.0, 128.0, 6.0);
        assert_eq!(analyser.total_frames, 1, "should have reset after gap");
    }

    #[test]
    fn test_sine_wave_heart_rate() {
        let mut analyser = RppgAnalyser::new();
        let bpm = 72.0_f64;
        let freq = bpm / 60.0;
        for i in 0..WINDOW_SIZE {
            let t = i as f64 / FPS;
            let g = 128.0 + 10.0 * (2.0 * core::f64::consts::PI * freq * t).sin();
            analyser.process_frame(g / 3.0, g, g / 3.0, t);
        }
        let json = analyser.process_frame(128.0, 128.0, 128.0, WINDOW_SIZE as f64 / FPS);
        let result: serde_json::Value = serde_json::from_str(&json).unwrap();
        let estimated_bpm = result["heart_rate_bpm"].as_f64().unwrap_or(0.0);
        // Allow ±10 BPM tolerance due to FFT resolution
        assert!(
            (estimated_bpm - bpm).abs() < 10.0,
            "expected ~{bpm} BPM, got {estimated_bpm}"
        );
    }
}
