/**
 * Session history.
 *
 * The privacy properties are the point of these tests: nothing leaves the device, and
 * nothing outlives the session.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { addToHistory, clearHistory, listHistory, nextInspectionLabel } from "../lib/history";

const entry = {
  label: "Tyre 1",
  verdict: "defect_suspected" as const,
  severity: "alert" as const,
  headline: "Possible wear or damage",
  confidence: 0.71,
  surface: "tread",
  quality: "Good",
  thumbnail: null,
};

beforeEach(() => sessionStorage.clear());

describe("history", () => {
  it("starts empty and records an inspection", () => {
    expect(listHistory()).toEqual([]);
    addToHistory(entry);
    expect(listHistory()).toHaveLength(1);
  });

  it("puts the newest entry first", () => {
    addToHistory({ ...entry, label: "Tyre 1" });
    addToHistory({ ...entry, label: "Tyre 2" });
    expect(listHistory()[0]?.label).toBe("Tyre 2");
  });

  it("caps the number of entries so storage cannot grow without bound", () => {
    for (let i = 0; i < 30; i += 1) addToHistory({ ...entry, label: `Tyre ${i}` });
    expect(listHistory().length).toBeLessThanOrEqual(12);
  });

  it("uses sessionStorage, so nothing outlives the session", () => {
    addToHistory(entry);
    expect(sessionStorage.getItem("tyretread.history.v1")).toBeTruthy();
    expect(localStorage.getItem("tyretread.history.v1")).toBeNull();
  });

  it("clears on request", () => {
    addToHistory(entry);
    expect(clearHistory()).toEqual([]);
    expect(listHistory()).toEqual([]);
  });

  it("survives corrupt stored data rather than throwing", () => {
    sessionStorage.setItem("tyretread.history.v1", "{not json");
    expect(listHistory()).toEqual([]);
  });

  it("numbers the next inspection from the count", () => {
    expect(nextInspectionLabel([])).toBe("Tyre 1");
    addToHistory(entry);
    expect(nextInspectionLabel(listHistory())).toBe("Tyre 2");
  });
});
