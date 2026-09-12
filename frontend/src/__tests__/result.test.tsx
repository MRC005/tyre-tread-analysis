/**
 * The result screen.
 *
 * These assert the product decisions that are easy to erode: a refusal must not read
 * as an error, a sidewall result must disclaim tread, and the screen must never claim
 * a tread-depth measurement.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Result } from "../screens/Result";
import type { Inspection } from "../api/types";

const base: Inspection = {
  result: {
    verdict: "defect_suspected",
    severity: "alert",
    label: "Defect detected",
    headline: "Visible signs of wear or damage",
    detail: "The surface resembles tyres with visible wear or damage.",
    meaning: "Something on this tyre's surface resembles tyres that are worn or damaged.",
    recommendation: "Have this tyre looked at by a qualified tyre professional.",
    probability_defect: 0.92,
    confidence: 0.71,
    decision_threshold: 0.52,
    abstain_band: 0.2,
  },
  image_quality: {
    usable: true,
    summary: "Good",
    issues: [],
    warnings: [],
    advice: [],
    warning_advice: [],
    metrics: { noise: 1.2 },
    checks: [{ name: "sensor_noise", passed: true, value: 1.2, threshold: 11 }],
  },
  measurements: {},
  model: { id: "current", artifact_format_version: 1, trained_on: "mendeley_tyres" },
  disclaimer: "This is an assistive visual screening tool. It cannot measure tread depth.",
  surface: { surface: "tread", tread_probability: 0.82, note: "This looks like the tread surface." },
  evidence_features: {
    measurements: [
      {
        feature: "edge_density",
        description: "the amount of visible groove edge structure",
        value: 0.17,
        percentile: 31,
        band: "typical",
        resembles: "tyres in good condition",
      },
    ],
    agreement: "4 of 5 individual measurements point the same way.",
    diagnostics: { legacy_tsci: 0.57 },
    note: "These are measurements taken from your photograph.",
  },
  explanation: { exact: false, method: "This model is not linear, so it cannot be decomposed.", reasons: [], contributions: [] },
  evidence: null,
};

const noop = () => {};

describe("Result", () => {
  it("leads with the verdict and its confidence", () => {
    render(<Result inspection={base} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Visible signs of wear or damage");
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "71");
  });

  it("keeps technical detail collapsed by default", async () => {
    render(<Result inspection={base} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);
    const trigger = screen.getByRole("button", { name: /technical analysis/i });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/quality checks/i)).toBeInTheDocument();
  });

  it("says plainly when the model cannot be decomposed", async () => {
    render(<Result inspection={base} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);
    await userEvent.click(screen.getByRole("button", { name: /technical analysis/i }));
    expect(screen.getByText(/Model interpretation is not available/i)).toBeInTheDocument();
  });

  it("warns that a sidewall result says nothing about tread", () => {
    const sidewall: Inspection = {
      ...base,
      surface: {
        surface: "sidewall_or_shoulder",
        tread_probability: 0.27,
        note: "This looks like the sidewall or shoulder rather than the tread. The assessment still applies to the rubber in the photograph, but it says nothing about how much tread is left.",
      },
    };
    render(<Result inspection={sidewall} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);
    expect(screen.getByText(/Sidewall \/ shoulder/)).toBeInTheDocument();
    expect(screen.getByText(/says nothing about how much tread is left/i)).toBeInTheDocument();
  });

  it("presents an unassessable image as a result, not an error", () => {
    const refused: Inspection = {
      ...base,
      result: {
        ...base.result,
        verdict: "unable_to_assess",
        severity: "unknown",
        label: "Unable to assess",
        headline: "This photo can't be assessed",
        meaning: "This says nothing about the tyre — only about the photograph.",
        probability_defect: null,
        confidence: null,
      },
      image_quality: {
        ...base.image_quality,
        usable: false,
        summary: "Unusable",
        issues: ["too_dark"],
        advice: ["Move somewhere brighter, or use your phone's flash."],
        warning_advice: [],
      },
    };
    render(<Result inspection={refused} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);

    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("can't be assessed");
    expect(screen.getByText("Photo is too dark")).toBeInTheDocument();
    expect(screen.getByText(/Move somewhere brighter/)).toBeInTheDocument();
    // No confidence is shown, because none exists.
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    // The only action offered is the one that helps.
    expect(screen.getByRole("button", { name: /retake photo/i })).toBeInTheDocument();
  });

  it("calls the retake handler from the refusal screen", async () => {
    const onRetake = vi.fn();
    const refused: Inspection = {
      ...base,
      result: { ...base.result, verdict: "unable_to_assess", severity: "unknown", label: "Unable to assess", headline: "Cannot assess", meaning: "About the photo, not the tyre.", probability_defect: null, confidence: null },
      image_quality: { ...base.image_quality, usable: false, summary: "Unusable", issues: ["too_blurry"], advice: ["Hold the phone still."], warning_advice: [] },
    };
    render(<Result inspection={refused} onNewInspection={noop} onRetake={onRetake} label="Tyre 1" />);
    await userEvent.click(screen.getByRole("button", { name: /retake photo/i }));
    expect(onRetake).toHaveBeenCalledOnce();
  });

  it("never claims a tread-depth measurement anywhere on screen", async () => {
    const { container } = render(
      <Result inspection={base} onNewInspection={noop} onRetake={noop} label="Tyre 1" />,
    );
    await userEvent.click(screen.getByRole("button", { name: /technical analysis/i }));
    const text = container.textContent?.toLowerCase() ?? "";
    for (const phrase of ["mm of tread", "tread depth is", "millimetre", "remaining tread"]) {
      expect(text).not.toContain(phrase);
    }
    expect(text).toContain("cannot measure tread depth");
  });

  it("labels the diagnostic as not used by the model", async () => {
    render(<Result inspection={base} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);
    await userEvent.click(screen.getByRole("button", { name: /technical analysis/i }));
    expect(screen.getByText(/deliberately not used by the model/i)).toBeInTheDocument();
    expect(screen.getByText("legacy_tsci")).toBeInTheDocument();
  });
});


// --------------------------------------------------------------------------
// Regressions found by real-device testing
// --------------------------------------------------------------------------

describe("Result — inconclusive", () => {
  const inconclusive: Inspection = {
    ...base,
    result: {
      ...base.result,
      verdict: "inconclusive",
      severity: "caution",
      label: "Attention recommended",
      headline: "Not conclusive",
      meaning: "A borderline surface, or a photograph that does not show quite enough.",
      recommendation: "Try another photograph from a slightly different angle.",
      confidence: 0,
      probability_defect: 0.695,
    },
    image_quality: {
      ...base.image_quality,
      summary: "Acceptable",
      warnings: ["resolution_too_low", "tread_not_located"],
      warning_advice: [
        "Take the photo closer to the tread, or use a larger image.",
        "Point the camera straight at the tread so it fills the middle of the frame.",
      ],
    },
  };

  it("does not render a 0% confidence meter", () => {
    render(<Result inspection={inconclusive} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);
    // A 0% bar under "Not conclusive" reads as a broken widget, not as honesty.
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(screen.getByText(/not confident enough to call this either way/i)).toBeInTheDocument();
  });

  it("leads the next step with the specific photo fix, not generic advice", () => {
    render(<Result inspection={inconclusive} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);
    expect(screen.getByText(/take the photo closer to the tread/i)).toBeInTheDocument();
    expect(screen.getByText(/fills the middle of the frame/i)).toBeInTheDocument();
  });

  it("still shows a confidence meter for a decided verdict", () => {
    render(<Result inspection={base} onNewInspection={noop} onRetake={noop} label="Tyre 1" />);
    expect(screen.getByRole("meter")).toBeInTheDocument();
  });
});
