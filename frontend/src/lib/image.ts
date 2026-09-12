/**
 * Preparing a photograph for upload.
 *
 * A modern phone produces a 12-megapixel, 4-6 MB JPEG. Uploading that over mobile data
 * is slow, and the backend reduces it to 1600 px on the long edge anyway, so shipping
 * the original wastes the user's time and data for no analytical gain.
 *
 * Two constraints shape what this does and does not do:
 *
 * **Do not over-compress.** The quality gate now refuses heavily compressed images,
 * because compression destroys the fine texture the model reads. Re-encoding at
 * quality 0.9 keeps JPEG blockiness around 1.2 against a rejection threshold of 2.2.
 * Being clever about bandwidth here would get the user's photo rejected.
 *
 * **Do not resize below what the analysis needs.** The backend refuses an ROI that
 * cannot fill the 256x128 analysis grid without upsampling. The ROI is the middle 60%
 * of the frame, so a 1600 px long edge leaves comfortable headroom; going much smaller
 * would start producing "image resolution too low" refusals caused by this code rather
 * than by the photograph.
 */

/** Matches the backend's own input reduction. No point sending more. */
const MAX_EDGE = 1600;
/** High enough to stay well clear of the blockiness threshold. */
const JPEG_QUALITY = 0.9;
/** Below this, re-encoding costs quality for no meaningful saving. */
const SKIP_BELOW_BYTES = 400 * 1024;

export interface PreparedImage {
  blob: Blob;
  previewUrl: string;
  width: number;
  height: number;
  originalBytes: number;
  bytes: number;
}

/**
 * Decode a file, applying EXIF orientation.
 *
 * `createImageBitmap` with `imageOrientation: "from-image"` applies the EXIF rotation
 * during decode, which matters because phones record orientation in metadata rather
 * than rotating pixels. Safari has historically lagged on the option, so a plain
 * `<img>` decode is used as a fallback - browsers apply EXIF to `<img>` rendering
 * anyway, so drawing one to a canvas yields upright pixels.
 */
async function decode(file: Blob): Promise<ImageBitmap | HTMLImageElement> {
  if (typeof createImageBitmap === "function") {
    try {
      return await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch {
      /* fall through to the <img> path */
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = "async";
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("could not decode image"));
      image.src = url;
    });
    return image;
  } finally {
    // Revoked after load; the decoded pixels are retained by the element.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

function dimensions(source: ImageBitmap | HTMLImageElement): { width: number; height: number } {
  if ("naturalWidth" in source) {
    return { width: source.naturalWidth, height: source.naturalHeight };
  }
  return { width: source.width, height: source.height };
}

export async function prepareImage(file: Blob): Promise<PreparedImage> {
  const source = await decode(file);
  const { width, height } = dimensions(source);

  if (width === 0 || height === 0) {
    throw new Error("could not read the image dimensions");
  }

  const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
  const needsResize = scale < 1;
  const needsReencode = needsResize || file.size > SKIP_BELOW_BYTES;

  if (!needsReencode) {
    if ("close" in source) source.close();
    return {
      blob: file,
      previewUrl: URL.createObjectURL(file),
      width,
      height,
      originalBytes: file.size,
      bytes: file.size,
    };
  }

  const targetWidth = Math.round(width * scale);
  const targetHeight = Math.round(height * scale);

  const canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("could not prepare the image for upload");

  // Browsers' default downscaling is a box filter; asking for high quality gets a
  // better resample, which matters because the analysis reads fine texture.
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source as CanvasImageSource, 0, 0, targetWidth, targetHeight);
  if ("close" in source) source.close();

  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY),
  );
  if (!blob) throw new Error("could not prepare the image for upload");

  return {
    blob,
    previewUrl: URL.createObjectURL(blob),
    width: targetWidth,
    height: targetHeight,
    originalBytes: file.size,
    bytes: blob.size,
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
