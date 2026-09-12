import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, checkHealth, inspect } from "./api/client";
import type { Inspection } from "./api/types";
import { Analyzing } from "./screens/Analyzing";
import { Capture } from "./screens/Capture";
import { Failure } from "./screens/Failure";
import { History } from "./screens/History";
import { Home } from "./screens/Home";
import { Preview } from "./screens/Preview";
import { Result } from "./screens/Result";
import { prepareImage, type PreparedImage } from "./lib/image";
import {
  addToHistory,
  clearHistory,
  listHistory,
  makeThumbnail,
  nextInspectionLabel,
  type HistoryEntry,
} from "./lib/history";

/**
 * The application shell and its state machine.
 *
 * Screens are driven by an explicit union rather than a router. The flow is linear and
 * short, every transition is deliberate, and there is no meaningful deep link into the
 * middle of it - a URL pointing at "analysing" or at a result held only in memory
 * would be a broken promise. That keeps the bundle free of a routing dependency.
 */
type Screen =
  | { name: "home" }
  | { name: "capture" }
  | { name: "preview"; image: PreparedImage }
  | { name: "analyzing"; image: PreparedImage }
  | { name: "result"; image: PreparedImage; inspection: Inspection; label: string }
  | { name: "failure"; image: PreparedImage; error: ApiError }
  | { name: "history" };

export function App() {
  const [screen, setScreen] = useState<Screen>({ name: "home" });
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [backendDown, setBackendDown] = useState(false);
  const [saveData, setSaveData] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const objectUrls = useRef<Set<string>>(new Set());

  useEffect(() => {
    setHistory(listHistory());

    // Default the data saver on when the browser reports a metered or slow connection.
    const connection = (navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string } })
      .connection;
    if (connection?.saveData || /^(slow-)?2g$/.test(connection?.effectiveType ?? "")) {
      setSaveData(true);
    }
  }, []);

  // A health probe on load, so the home screen can warn before the user spends time
  // photographing a tyre for a service that is down.
  useEffect(() => {
    let cancelled = false;
    checkHealth()
      .then((health) => {
        if (!cancelled) setBackendDown(health.status !== "ok" || !health.model_loaded);
      })
      .catch(() => {
        if (!cancelled) setBackendDown(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Object URLs are not garbage collected; releasing them avoids leaking a few MB per
  // inspection on a long session.
  const trackUrl = useCallback((url: string) => {
    objectUrls.current.add(url);
  }, []);
  useEffect(() => {
    const urls = objectUrls.current;
    return () => urls.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  // Scroll to the top on every screen change, and move focus to the new heading so a
  // screen reader announces where it has landed.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [screen.name]);

  async function handleCaptured(file: Blob) {
    try {
      const image = await prepareImage(file);
      trackUrl(image.previewUrl);
      setScreen({ name: "preview", image });
    } catch {
      setScreen({
        name: "failure",
        image: { blob: file, previewUrl: "", width: 0, height: 0, originalBytes: file.size, bytes: file.size },
        error: new ApiError(
          "undecodable_image",
          "That photo could not be read. Try taking it again, or choose a different file.",
        ),
      });
    }
  }

  async function runAnalysis(image: PreparedImage) {
    setScreen({ name: "analyzing", image });
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const inspection = await inspect(image.blob, {
        includeEvidence: !saveData,
        includeFeatures: false,
        signal: controller.signal,
      });

      const label = nextInspectionLabel(history);
      setScreen({ name: "result", image, inspection, label });
      setBackendDown(false);

      // Record everything except unassessable attempts: a refused photo is not an
      // inspection of a tyre, and listing it would clutter the session view.
      if (inspection.result.verdict !== "unable_to_assess") {
        const thumbnail = await makeThumbnail(image.previewUrl);
        setHistory(
          addToHistory({
            label,
            verdict: inspection.result.verdict,
            severity: inspection.result.severity,
            headline: inspection.result.headline,
            confidence: inspection.result.confidence,
            surface: inspection.surface?.surface ?? null,
            quality: inspection.image_quality.summary,
            thumbnail,
          }),
        );
      }
    } catch (error) {
      if (error instanceof ApiError) {
        if (error.code === "aborted") return;
        if (error.code === "model_unavailable" || error.code === "network") {
          setBackendDown(true);
        }
        setScreen({ name: "failure", image, error });
      } else {
        setScreen({
          name: "failure",
          image,
          error: new ApiError("unknown", "Something went wrong. Please try again.", undefined, true),
        });
      }
    } finally {
      abortRef.current = null;
    }
  }

  function cancelAnalysis(image: PreparedImage) {
    abortRef.current?.abort();
    setScreen({ name: "preview", image });
  }

  return (
    <div className="app">
      <a href="#main" className="skiplink">
        Skip to content
      </a>

      <header className="app__bar">
        <div className="container app__bar-inner">
          <span className="wordmark">
            {/* A tread-groove glyph: three slots, which is what the analysis looks at.
                Inline SVG so there is no icon-font or sprite request. */}
            <svg
              className="wordmark__mark"
              viewBox="0 0 20 20"
              aria-hidden="true"
              focusable="false"
            >
              <rect x="2" y="3" width="3.2" height="14" rx="1.4" />
              <rect x="8.4" y="3" width="3.2" height="14" rx="1.4" />
              <rect x="14.8" y="3" width="3.2" height="14" rx="1.4" />
            </svg>
            <span className="wordmark__text">
              Tread<span className="wordmark__accent">Check</span>
            </span>
          </span>
          {screen.name !== "home" && (
            <button
              type="button"
              className="app__bar-action"
              onClick={() => setScreen({ name: "home" })}
            >
              Home
            </button>
          )}
        </div>
      </header>

      <main id="main" className="app__main">
        {screen.name === "home" && (
          <Home
            onStart={() => setScreen({ name: "capture" })}
            history={history}
            onOpenHistory={() => setScreen({ name: "history" })}
            backendDown={backendDown}
          />
        )}

        {screen.name === "capture" && (
          <Capture onCaptured={handleCaptured} onBack={() => setScreen({ name: "home" })} />
        )}

        {screen.name === "preview" && (
          <Preview
            image={screen.image}
            onConfirm={() => runAnalysis(screen.image)}
            onRetake={() => setScreen({ name: "capture" })}
            saveData={saveData}
            onToggleSaveData={setSaveData}
          />
        )}

        {screen.name === "analyzing" && <Analyzing onCancel={() => cancelAnalysis(screen.image)} />}

        {screen.name === "result" && (
          <Result
            inspection={screen.inspection}
            label={screen.label}
            onNewInspection={() => setScreen({ name: "capture" })}
            onRetake={() => setScreen({ name: "capture" })}
          />
        )}

        {screen.name === "failure" && (
          <Failure
            error={screen.error}
            onRetry={() => runAnalysis(screen.image)}
            onBack={() => setScreen({ name: "capture" })}
          />
        )}

        {screen.name === "history" && (
          <History
            entries={history}
            onBack={() => setScreen({ name: "home" })}
            onClear={() => setHistory(clearHistory())}
            onNew={() => setScreen({ name: "capture" })}
          />
        )}
      </main>

      <footer className="app__footer">
        <div className="container">
          <p>
            Assistive screening only · cannot measure tread depth · not a certified
            inspection
          </p>
        </div>
      </footer>
    </div>
  );
}
