import { useCallback, useEffect, useRef, useState } from "react";
import { Button, LiveRegion } from "../components/ui";
import {
  captureFrame,
  cameraSupported,
  describeCameraError,
  openCamera,
  secureContext,
  stopCamera,
  type CameraError,
} from "../lib/camera";
import { GRADE_LABEL, assessFrame, type LiveQuality } from "../lib/liveQuality";

/**
 * Capture: live camera, or a file from the device.
 *
 * The camera is not started until the user asks for it. Requesting `getUserMedia` on
 * mount would fire the browser's permission prompt before the user has any context for
 * what it is for, which is both worse UX and more likely to be declined - and a
 * declined permission is sticky.
 *
 * Guidance is shown as a short overlay on the viewfinder rather than a wall of text
 * before it. People read instructions while framing a shot, not before.
 */
export function Capture({
  onCaptured,
  onBack,
}: {
  onCaptured: (file: Blob) => void;
  onBack: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [streaming, setStreaming] = useState(false);
  const [starting, setStarting] = useState(false);
  const [cameraError, setCameraError] = useState<CameraError | null>(null);
  const [dragging, setDragging] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const [live, setLive] = useState<LiveQuality | null>(null);

  const cameraPossible = cameraSupported() && secureContext();

  const stop = useCallback(() => {
    stopCamera(streamRef.current);
    streamRef.current = null;
    setStreaming(false);
    setLive(null);
  }, []);

  // Sample the preview a few times a second for framing feedback. 350 ms is frequent
  // enough to feel responsive while the user moves the phone, and slow enough that the
  // label does not flicker between grades as they settle.
  useEffect(() => {
    if (!streaming) return;
    const id = setInterval(() => {
      if (videoRef.current) setLive(assessFrame(videoRef.current));
    }, 350);
    return () => clearInterval(id);
  }, [streaming]);

  // Release the camera when the screen unmounts. A live track keeps the phone's
  // camera indicator lit and drains battery.
  useEffect(() => stop, [stop]);

  // Also release it when the tab is hidden: iOS suspends the stream anyway and
  // resuming a suspended track is unreliable.
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) stop();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [stop]);

  async function start() {
    setStarting(true);
    setCameraError(null);
    try {
      const stream = await openCamera();
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // `playsInline` is set on the element; without it iOS Safari takes the video
        // fullscreen and the capture UI disappears.
        await videoRef.current.play();
      }
      setStreaming(true);
      setAnnouncement("Camera ready. Frame the tyre tread and take a photo.");
    } catch (error) {
      setCameraError(describeCameraError(error));
      setAnnouncement("The camera could not be started.");
    } finally {
      setStarting(false);
    }
  }

  async function shoot() {
    if (!videoRef.current) return;
    try {
      const blob = await captureFrame(videoRef.current);
      stop();
      onCaptured(blob);
    } catch {
      setCameraError(describeCameraError(new Error("capture failed")));
    }
  }

  function pickFile(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    stop();
    onCaptured(file);
  }

  return (
    <div className="capture">
      <LiveRegion message={announcement} />

      <header className="screen-header container">
        <Button variant="ghost" onClick={onBack}>
          ← Back
        </Button>
        <h1 className="screen-header__title">Take a photo</h1>
        <span className="screen-header__spacer" />
      </header>

      <div className="container stack">
        <div
          className={`viewfinder ${streaming ? "is-live" : ""} ${dragging ? "is-dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            pickFile(event.dataTransfer.files);
          }}
        >
          <video
            ref={videoRef}
            className="viewfinder__video"
            playsInline
            muted
            aria-label="Camera preview"
          />

          {streaming && (
            <>
              {/* A framing guide, not decoration: the analysis uses the middle band of
                  the frame, so showing where that is materially improves the photo. */}
              <div className="viewfinder__guide" aria-hidden="true">
                <span
                  className={`viewfinder__guide-band viewfinder__guide-band--${live?.grade ?? "fair"}`}
                />
              </div>

              {/* Live framing feedback. A hint only: the backend gate is the authority
                  and is stricter. See lib/liveQuality.ts. */}
              <div className={`livegrade livegrade--${live?.grade ?? "fair"}`} role="status">
                <span className="livegrade__dot" aria-hidden="true" />
                <span className="livegrade__label">
                  {GRADE_LABEL[live?.grade ?? "fair"]}
                </span>
              </div>

              <p className="viewfinder__hint">
                {live?.hint ?? "Fill the guide with the tread"}
              </p>
            </>
          )}

          {!streaming && !cameraError && (
            <div className="viewfinder__idle">
              {/* The idle state shows the same framing band the live camera does, so
                  the user learns what is expected before the permission prompt rather
                  than staring at an empty rectangle. */}
              <div className="framehint" aria-hidden="true">
                <svg viewBox="0 0 120 80" className="framehint__svg">
                  <rect
                    x="6" y="22" width="108" height="36" rx="4"
                    className="framehint__band"
                  />
                  {[16, 31, 46, 61, 76, 91].map((x) => (
                    <rect key={x} x={x} y="28" width="7" height="24" rx="2.5"
                          className="framehint__groove" />
                  ))}
                  <path d="M6 12 L6 6 L18 6 M114 12 L114 6 L102 6
                           M6 68 L6 74 L18 74 M114 68 L114 74 L102 74"
                        className="framehint__corners" />
                </svg>
                <span className="framehint__caption">Fill this band with the tread</span>
              </div>

              {cameraPossible ? (
                <p className="viewfinder__idle-text">
                  We'll ask for camera permission next.
                </p>
              ) : (
                <>
                  <p className="viewfinder__idle-title">
                    {secureContext() ? "Camera not available here" : "Camera needs HTTPS"}
                  </p>
                  <p className="viewfinder__idle-text">
                    Choose a photo from this device instead.
                  </p>
                </>
              )}
            </div>
          )}

          {cameraError && (
            <div className="viewfinder__error" role="alert">
              <p className="viewfinder__error-title">{cameraError.title}</p>
              <p className="viewfinder__error-text">{cameraError.detail}</p>
            </div>
          )}
        </div>

        <div className="capture__actions">
          {streaming ? (
            <>
              <button
                type="button"
                className={`shutter shutter--${live?.grade ?? "fair"}`}
                onClick={shoot}
                aria-label={
                  live?.hint
                    ? `Take photo. Framing: ${GRADE_LABEL[live.grade]}. ${live.hint}`
                    : "Take photo. Framing is ready."
                }
              >
                <span className="shutter__ring" aria-hidden="true" />
              </button>
              <Button variant="ghost" block onClick={stop}>
                Stop camera
              </Button>
            </>
          ) : (
            <>
              {cameraPossible && (
                <Button
                  variant="primary"
                  size="lg"
                  block
                  busy={starting}
                  onClick={start}
                >
                  {cameraError ? "Try camera again" : "Open camera"}
                </Button>
              )}
              <Button
                variant={cameraPossible ? "secondary" : "primary"}
                size="lg"
                block
                onClick={() => fileInputRef.current?.click()}
              >
                Choose a photo
              </Button>
              <p className="capture__droptip">or drag a photo here</p>
            </>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={(event) => pickFile(event.target.files)}
        />

        {!streaming && (
          <section className="tips" aria-label="Photo tips">
            <h2 className="tips__title">For a good photo</h2>
            <ul className="tips__list">
              <li>Get close — the tread should fill most of the frame</li>
              <li>Hold the phone square to the tyre, not at an angle</li>
              <li>Good light, but avoid direct glare on the rubber</li>
              <li>Hold still and let the camera focus</li>
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}
