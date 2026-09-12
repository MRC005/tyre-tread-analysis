/**
 * API client error mapping.
 *
 * The screens branch on `code` and show `message` verbatim, so both are part of the
 * contract with the user. These tests pin that every failure the backend can return
 * becomes something a person can act on, rather than a status number.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, inspect } from "../api/client";

function respond(status: number, body: unknown = {}) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("inspect", () => {
  it("returns the parsed inspection on success", async () => {
    const payload = { result: { verdict: "likely_serviceable" } };
    vi.stubGlobal("fetch", vi.fn(() => respond(200, payload)));
    await expect(inspect(new Blob(["x"]))).resolves.toMatchObject(payload);
  });

  it.each([
    [413, "file_too_large", false],
    [415, "unsupported_media_type", false],
    [400, "undecodable_image", false],
    [503, "model_unavailable", true],
    [504, "inspection_timeout", true],
  ])("maps HTTP %i to %s", async (status, code, retryable) => {
    vi.stubGlobal("fetch", vi.fn(() => respond(status)));
    const error = await inspect(new Blob(["x"])).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe(code);
    expect(error.retryable).toBe(retryable);
  });

  it("turns a network failure into a retryable, human message", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    const error = await inspect(new Blob(["x"])).catch((e) => e);
    expect(error.code).toBe("network");
    expect(error.retryable).toBe(true);
    expect(error.message).toMatch(/connection/i);
  });

  it("never surfaces a raw server message to the user", async () => {
    vi.stubGlobal("fetch", vi.fn(() => respond(500, { message: "x".repeat(500) })));
    const error = await inspect(new Blob(["x"])).catch((e) => e);
    expect(error.message.length).toBeLessThan(200);
    expect(error.message).not.toContain("xxxx");
  });

  it("passes the evidence and features flags as query parameters", async () => {
    const fetchMock = vi.fn(() => respond(200, {}));
    vi.stubGlobal("fetch", fetchMock);
    await inspect(new Blob(["x"]), { includeEvidence: false, includeFeatures: true });
    const url = String(fetchMock.mock.calls.at(0)?.at(0));
    expect(url).toContain("include_evidence=false");
    expect(url).toContain("include_features=true");
  });

  it("reports a caller cancellation distinctly from a failure", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn(() => {
        controller.abort();
        return Promise.reject(new DOMException("aborted", "AbortError"));
      }),
    );
    const error = await inspect(new Blob(["x"]), { signal: controller.signal }).catch((e) => e);
    expect(error.code).toBe("aborted");
    expect(error.retryable).toBe(false);
  });

  it("sends the image as multipart form data", async () => {
    const fetchMock = vi.fn(() => respond(200, {}));
    vi.stubGlobal("fetch", fetchMock);
    await inspect(new Blob(["x"], { type: "image/jpeg" }));
    const init = fetchMock.mock.calls.at(0)?.at(1) as unknown as RequestInit;
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("image")).toBeTruthy();
  });
});
