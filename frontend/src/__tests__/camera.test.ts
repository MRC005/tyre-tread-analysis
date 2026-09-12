/**
 * Camera error classification.
 *
 * Every branch must produce advice, and every branch must leave the gallery-upload
 * route open - losing camera access should never leave a user with no way forward.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { describeCameraError } from "../lib/camera";

function secure(value: boolean) {
  vi.stubGlobal("window", { ...window, isSecureContext: value });
  Object.defineProperty(window, "isSecureContext", { value, configurable: true });
}

afterEach(() => {
  Object.defineProperty(window, "isSecureContext", { value: true, configurable: true });
  vi.unstubAllGlobals();
});

describe("describeCameraError", () => {
  it("flags an insecure context before anything else", () => {
    secure(false);
    const error = describeCameraError(new DOMException("x", "NotAllowedError"));
    expect(error.kind).toBe("insecure_context");
    expect(error.detail).toMatch(/HTTPS/);
  });

  it.each([
    ["NotAllowedError", "permission_denied"],
    ["SecurityError", "permission_denied"],
    ["NotFoundError", "no_camera"],
    ["OverconstrainedError", "no_camera"],
    ["NotReadableError", "in_use"],
  ])("maps %s to %s", (name, kind) => {
    secure(true);
    Object.defineProperty(navigator, "mediaDevices", {
      value: { getUserMedia: () => {} },
      configurable: true,
    });
    expect(describeCameraError(new DOMException("x", name)).kind).toBe(kind);
  });

  it("always leaves gallery upload available", () => {
    secure(true);
    Object.defineProperty(navigator, "mediaDevices", {
      value: { getUserMedia: () => {} },
      configurable: true,
    });
    for (const name of ["NotAllowedError", "NotFoundError", "NotReadableError", "Whatever"]) {
      const error = describeCameraError(new DOMException("x", name));
      expect(error.canUploadInstead).toBe(true);
      expect(error.title.length).toBeGreaterThan(0);
      expect(error.detail.length).toBeGreaterThan(0);
    }
  });
});
