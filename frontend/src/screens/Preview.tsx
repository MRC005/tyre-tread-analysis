import { Button } from "../components/ui";
import { formatBytes, type PreparedImage } from "../lib/image";

/** Confirm or retake, before spending the user's bandwidth on an upload. */
export function Preview({
  image,
  onConfirm,
  onRetake,
  saveData,
  onToggleSaveData,
}: {
  image: PreparedImage;
  onConfirm: () => void;
  onRetake: () => void;
  saveData: boolean;
  onToggleSaveData: (value: boolean) => void;
}) {
  const shrank = image.bytes < image.originalBytes;
  return (
    <div className="preview">
      <header className="screen-header container">
        <Button variant="ghost" onClick={onRetake}>
          ← Retake
        </Button>
        <h1 className="screen-header__title">Check the photo</h1>
        <span className="screen-header__spacer" />
      </header>

      <div className="container stack">
        <figure className="preview__figure">
          <img src={image.previewUrl} alt="The photo you just took" className="preview__image" />
          <figcaption className="preview__meta">
            {image.width}×{image.height}
            {shrank && (
              <>
                {" · "}
                {formatBytes(image.bytes)}{" "}
                <span className="preview__meta-dim">
                  (resized from {formatBytes(image.originalBytes)})
                </span>
              </>
            )}
          </figcaption>
        </figure>

        <div className="preview__check">
          <p className="preview__check-title">Before you continue</p>
          <ul className="preview__check-list">
            <li>Is the tread surface clearly visible?</li>
            <li>Is it sharp rather than blurred?</li>
            <li>Does it fill most of the frame?</li>
          </ul>
        </div>

        <label className="toggle">
          <input
            type="checkbox"
            checked={saveData}
            onChange={(event) => onToggleSaveData(event.target.checked)}
          />
          <span className="toggle__body">
            <span className="toggle__label">Data saver</span>
            <span className="toggle__hint">
              Skip the visual analysis images. Faster on a slow connection.
            </span>
          </span>
        </label>

        <div className="stack-sm">
          <Button variant="primary" size="lg" block onClick={onConfirm}>
            Analyse this photo
          </Button>
          <Button variant="secondary" block onClick={onRetake}>
            Take another
          </Button>
        </div>
      </div>
    </div>
  );
}
