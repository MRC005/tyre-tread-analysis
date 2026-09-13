/**
 * Live framing feedback, computed in the browser before the shutter is pressed.
 *
 * The backend quality gate is honest but late: the user has already framed, tapped,
 * waited for an upload, and only then learns the photo was too dark. The fix is to give
 * feedback *during* framing, naming the specific problem rather than a generic "poor
 * quality" — the way document-scanning apps handle the same situation.
 *
 * This is **not** the quality gate, and it does not decide anything. The backend gate
 * remains the authority — it works on the full-resolution image after ROI extraction,
 * measures things this cannot (oversampling against the analysis grid, blockiness,
 * ROI coverage), and is the version that was empirically calibrated. This is a hint,
 * deliberately more forgiving, so it nudges rather than blocks.
 *
 * The three checks mirror the gate's three most common rejection reasons, measured on
 * a heavily downscaled frame so it costs almost nothing:
 *
 *   - **brightness** — mean luma, the gate's most frequent refusal after resolution
 *   - **contrast** — the 2nd-to-98th percentile spread, which separates a genuinely
 *     dark tyre from an underexposed photograph exactly as the gate does
 *   - **sharpness** — a Laplacian-style neighbour difference, standing in for the
 *     gate's Laplacian variance
 */

export type LiveGrade = "poor" | "fair" | "good" | "ready";

export interface LiveQuality {
  grade: LiveGrade;
  /** The single most useful thing to fix right now, or null when ready. */
  hint: string | null;
  metrics: { brightness: number; contrast: number; sharpness: number };
}

/** Downscale target. Small enough to be free, large enough for the statistics. */
const SAMPLE = 96;

let canvas: HTMLCanvasElement | null = null;

function sampleContext(): CanvasRenderingContext2D | null {
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.width = SAMPLE;
    canvas.height = SAMPLE;
  }
  return canvas.getContext("2d", { alpha: false, willReadFrequently: true });
}

export function assessFrame(video: HTMLVideoElement): LiveQuality | null {
  if (!video.videoWidth || !video.videoHeight) return null;
  const context = sampleContext();
  if (!context) return null;

  // Sample the centre square: that is roughly what the framing guide covers, and it
  // avoids letting dark background at the edges drag the statistics down.
  const side = Math.min(video.videoWidth, video.videoHeight);
  const sx = (video.videoWidth - side) / 2;
  const sy = (video.videoHeight - side) / 2;
  context.drawImage(video, sx, sy, side, side, 0, 0, SAMPLE, SAMPLE);

  const { data } = context.getImageData(0, 0, SAMPLE, SAMPLE);
  const luma = new Float32Array(SAMPLE * SAMPLE);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    luma[p] = 0.299 * data[i]! + 0.587 * data[i + 1]! + 0.114 * data[i + 2]!;
  }

  let sum = 0;
  for (let i = 0; i < luma.length; i += 1) sum += luma[i]!;
  const brightness = sum / luma.length;

  const sorted = Float32Array.from(luma).sort();
  const contrast =
    sorted[Math.floor(sorted.length * 0.98)]! - sorted[Math.floor(sorted.length * 0.02)]!;

  // Mean absolute 4-neighbour difference: a cheap stand-in for Laplacian energy.
  let edge = 0;
  let count = 0;
  for (let y = 1; y < SAMPLE - 1; y += 1) {
    for (let x = 1; x < SAMPLE - 1; x += 1) {
      const i = y * SAMPLE + x;
      edge += Math.abs(
        4 * luma[i]! - luma[i - 1]! - luma[i + 1]! - luma[i - SAMPLE]! - luma[i + SAMPLE]!,
      );
      count += 1;
    }
  }
  const sharpness = edge / Math.max(count, 1);

  return {
    ...gradeMetrics(brightness, contrast, sharpness),
    metrics: { brightness, contrast, sharpness },
  };
}

/**
 * Thresholds are looser than the backend gate's on purpose.
 *
 * A live hint that fires more readily than the real gate would train users to
 * distrust it, and a preview frame is noisier and lower-resolution than the captured
 * photograph, so the same numbers would not mean the same thing. These are set to
 * catch the obvious cases — a lens cap, a dark garage, a moving phone — and stay quiet
 * otherwise.
 */
export function gradeMetrics(
  brightness: number,
  contrast: number,
  sharpness: number,
): { grade: LiveGrade; hint: string | null } {
  if (brightness < 45) {
    return { grade: "poor" as const, hint: "Too dark — find more light" };
  }
  if (brightness > 225) {
    return { grade: "poor" as const, hint: "Too bright — move out of direct light" };
  }
  if (sharpness < 3) {
    return { grade: "poor" as const, hint: "Hold steady and let the camera focus" };
  }
  if (contrast < 45) {
    return { grade: "fair" as const, hint: "Move closer so the tread fills the guide" };
  }
  if (sharpness < 7) {
    return { grade: "fair" as const, hint: "Hold steady — nearly there" };
  }
  if (brightness < 70) {
    return { grade: "good" as const, hint: "A little more light would help" };
  }
  return { grade: "ready" as const, hint: null };
}

export const GRADE_LABEL: Record<LiveGrade, string> = {
  poor: "Poor",
  fair: "Fair",
  good: "Good",
  ready: "Ready",
};
