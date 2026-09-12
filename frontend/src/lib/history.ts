/**
 * Local inspection history.
 *
 * Deliberately minimal, and deliberately local. A user checking four tyres wants to
 * compare them without re-photographing, which needs nothing more than the verdict and
 * a thumbnail.
 *
 * Privacy decisions, made once and recorded here:
 *
 * - **No account, no upload, no server-side record.** Photographs of a vehicle are
 *   personal data; the backend already keeps nothing, and this keeps nothing either.
 * - **Stored in `sessionStorage`, not `localStorage`.** History is useful within one
 *   visit and is not worth leaving on a shared or borrowed phone afterwards.
 * - **A small thumbnail, not the photograph.** Enough to recognise which tyre, far too
 *   little to be a record of the vehicle, and small enough not to exhaust the quota.
 */

import type { Inspection } from "../api/types";

const KEY = "tyretread.history.v1";
const MAX_ENTRIES = 12;
const THUMB_EDGE = 96;

export interface HistoryEntry {
  id: string;
  at: number;
  label: string;
  verdict: Inspection["result"]["verdict"];
  severity: Inspection["result"]["severity"];
  headline: string;
  confidence: number | null;
  surface: string | null;
  quality: string;
  thumbnail: string | null;
}

function read(): HistoryEntry[] {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    // Private mode, disabled storage, or corrupt data: history is a convenience, so
    // failing to read it must never break an inspection.
    return [];
  }
}

function write(entries: HistoryEntry[]): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(entries));
  } catch {
    /* quota or disabled storage - ignore */
  }
}

/** Shrink a preview to a thumbnail small enough to store many of. */
export async function makeThumbnail(previewUrl: string): Promise<string | null> {
  try {
    const image = new Image();
    image.decoding = "async";
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("thumbnail failed"));
      image.src = previewUrl;
    });

    const scale = THUMB_EDGE / Math.max(image.naturalWidth, image.naturalHeight);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return null;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.6);
  } catch {
    return null;
  }
}

export function listHistory(): HistoryEntry[] {
  return read();
}

export function addToHistory(entry: Omit<HistoryEntry, "id" | "at">): HistoryEntry[] {
  const full: HistoryEntry = {
    ...entry,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    at: Date.now(),
  };
  const entries = [full, ...read()].slice(0, MAX_ENTRIES);
  write(entries);
  return entries;
}

export function clearHistory(): HistoryEntry[] {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
  return [];
}

export function nextInspectionLabel(entries: HistoryEntry[]): string {
  return `Tyre ${entries.length + 1}`;
}
