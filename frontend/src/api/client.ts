/**
 * Backend client.
 *
 * Every failure the user can encounter is turned into an `ApiError` carrying a stable
 * code and a sentence written for a person rather than a developer. The screens
 * branch on the code; they never parse a message.
 */

import type { ApiErrorCode, Health, Inspection } from "./types";

/**
 * Where the backend lives.
 *
 * Unset means "same origin, behind /api" - which is how development works, because the
 * Vite dev server proxies /api to the backend. That keeps phone testing to a single
 * HTTPS tunnel with no CORS involved.
 *
 * In production this is set to the Render URL at build time. It is never hardcoded:
 * the same source has to be able to point at a local backend or at production.
 */
const CONFIGURED = import.meta.env.VITE_API_BASE_URL?.trim();
const BASE_URL: string = CONFIGURED ? CONFIGURED.replace(/\/+$/, "") : "/api";

/** Generous, because a large photo on a slow mobile uplink is the normal case. */
const UPLOAD_TIMEOUT_MS = 45_000;
const HEALTH_TIMEOUT_MS = 6_000;

export class ApiError extends Error {
  readonly code: ApiErrorCode;
  readonly status?: number;
  /** Whether trying the same request again could plausibly succeed. */
  readonly retryable: boolean;

  constructor(code: ApiErrorCode, message: string, status?: number, retryable = false) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.retryable = retryable;
  }
}

/** Messages for the cases the user can actually do something about. */
const MESSAGES: Record<number, { code: ApiErrorCode; message: string; retryable: boolean }> = {
  400: {
    code: "undecodable_image",
    message: "That file could not be read as an image. Try a JPEG or PNG photo.",
    retryable: false,
  },
  413: {
    code: "file_too_large",
    message: "That photo is too large. Try taking it again — it will be resized automatically.",
    retryable: false,
  },
  415: {
    code: "unsupported_media_type",
    message: "That file type is not supported. Please use a JPEG or PNG photo.",
    retryable: false,
  },
  503: {
    code: "model_unavailable",
    message: "The analysis service is starting up. Please try again in a moment.",
    retryable: true,
  },
  504: {
    code: "inspection_timeout",
    message: "Analysis took too long. Please try again.",
    retryable: true,
  },
};

function timeoutSignal(ms: number): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(new DOMException("timeout", "TimeoutError")), ms);
  return { signal: controller.signal, cancel: () => clearTimeout(id) };
}

async function toApiError(response: Response): Promise<ApiError> {
  const known = MESSAGES[response.status];
  if (known) {
    return new ApiError(known.code, known.message, response.status, known.retryable);
  }
  // Anything else: try the body's message, but never show a raw stack to a user.
  try {
    const body = await response.json();
    if (typeof body?.message === "string" && body.message.length < 200) {
      return new ApiError(body.error ?? "unknown", body.message, response.status, response.status >= 500);
    }
  } catch {
    /* body was not JSON; fall through */
  }
  return new ApiError(
    "unknown",
    "Something went wrong while analysing the photo. Please try again.",
    response.status,
    response.status >= 500,
  );
}

export interface InspectOptions {
  includeEvidence?: boolean;
  includeFeatures?: boolean;
  /** Lets a screen cancel an in-flight upload when the user backs out. */
  signal?: AbortSignal;
}

export async function inspect(file: Blob, options: InspectOptions = {}): Promise<Inspection> {
  const { includeEvidence = true, includeFeatures = false, signal } = options;

  const params = new URLSearchParams({
    include_evidence: String(includeEvidence),
    include_features: String(includeFeatures),
  });

  const body = new FormData();
  // The filename matters only for the content-type sniff on the server side.
  body.append("image", file, "tyre.jpg");

  const timeout = timeoutSignal(UPLOAD_TIMEOUT_MS);
  // Combine the caller's cancellation with our timeout.
  const combined = signal
    ? AbortSignal.any
      ? AbortSignal.any([signal, timeout.signal])
      : timeout.signal
    : timeout.signal;

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/v1/inspect?${params}`, {
      method: "POST",
      body,
      signal: combined,
    });
  } catch (error) {
    timeout.cancel();
    if (signal?.aborted) {
      throw new ApiError("aborted", "Analysis cancelled.", undefined, false);
    }
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new ApiError(
        "inspection_timeout",
        "The upload timed out. Check your connection and try again.",
        undefined,
        true,
      );
    }
    throw new ApiError(
      "network",
      "Could not reach the analysis service. Check your connection and try again.",
      undefined,
      true,
    );
  } finally {
    timeout.cancel();
  }

  if (!response.ok) {
    throw await toApiError(response);
  }
  return (await response.json()) as Inspection;
}

export async function checkHealth(): Promise<Health> {
  const timeout = timeoutSignal(HEALTH_TIMEOUT_MS);
  try {
    const response = await fetch(`${BASE_URL}/health`, { signal: timeout.signal });
    if (!response.ok) throw await toApiError(response);
    return (await response.json()) as Health;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      "network",
      "Could not reach the analysis service.",
      undefined,
      true,
    );
  } finally {
    timeout.cancel();
  }
}

export const apiBaseUrl = BASE_URL;
