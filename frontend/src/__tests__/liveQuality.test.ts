/**
 * Live framing feedback.
 *
 * The grading thresholds are tested directly rather than through a canvas. jsdom has
 * no canvas implementation, and pulling in a native `canvas` build purely to read back
 * pixels in a unit test would be a heavy dependency for plumbing that the real browser
 * already exercises. What matters here is the decision table: which measurements
 * produce which grade, and whether every non-ready grade tells the user what to fix.
 *
 * The behavioural contract is that this *hints*. It must never be mistaken for the
 * quality gate, which is stricter, runs on the full-resolution image after ROI
 * extraction, and is the version that was empirically calibrated.
 */

import { describe, expect, it } from "vitest";
import { GRADE_LABEL, assessFrame, gradeMetrics } from "../lib/liveQuality";

// Representative of a well-lit, sharp, high-contrast tread photo.
const GOOD = { brightness: 130, contrast: 120, sharpness: 14 };

describe("gradeMetrics", () => {
  it("calls a well-framed shot ready, with nothing to fix", () => {
    const result = gradeMetrics(GOOD.brightness, GOOD.contrast, GOOD.sharpness);
    expect(result.grade).toBe("ready");
    expect(result.hint).toBeNull();
  });

  it("calls a dark frame poor and says it is too dark", () => {
    const result = gradeMetrics(20, GOOD.contrast, GOOD.sharpness);
    expect(result.grade).toBe("poor");
    expect(result.hint).toMatch(/dark/i);
  });

  it("calls a blown-out frame poor and says it is too bright", () => {
    const result = gradeMetrics(240, GOOD.contrast, GOOD.sharpness);
    expect(result.grade).toBe("poor");
    expect(result.hint).toMatch(/bright/i);
  });

  it("calls a badly blurred frame poor and says to hold steady", () => {
    const result = gradeMetrics(GOOD.brightness, GOOD.contrast, 1);
    expect(result.grade).toBe("poor");
    expect(result.hint).toMatch(/steady|focus/i);
  });

  it("asks the user to move closer when contrast is low", () => {
    const result = gradeMetrics(GOOD.brightness, 20, GOOD.sharpness);
    expect(result.hint).toMatch(/closer/i);
  });

  it("every grade below ready carries an actionable hint", () => {
    const cases = [
      [20, 120, 14],
      [240, 120, 14],
      [130, 120, 1],
      [130, 20, 14],
      [130, 120, 5],
      [60, 120, 14],
    ] as const;
    for (const [b, c, s] of cases) {
      const result = gradeMetrics(b, c, s);
      expect(result.grade).not.toBe("ready");
      expect(result.hint, `no hint for ${b}/${c}/${s}`).toBeTruthy();
    }
  });

  it("is more forgiving than the backend gate, by design", () => {
    // The gate refuses below a raw mean of 40. This should still allow that through as
    // a warning rather than blocking, because it is a hint and the gate is the
    // authority - a live hint that fires sooner than the real gate trains distrust.
    expect(gradeMetrics(50, 120, 14).grade).not.toBe("poor");
  });

  it("labels every grade", () => {
    for (const grade of ["poor", "fair", "good", "ready"] as const) {
      expect(GRADE_LABEL[grade]).toBeTruthy();
    }
  });
});

describe("assessFrame", () => {
  it("returns null before the camera reports dimensions", () => {
    const notReady = { videoWidth: 0, videoHeight: 0 } as HTMLVideoElement;
    expect(assessFrame(notReady)).toBeNull();
  });
});
