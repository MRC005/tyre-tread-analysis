/**
 * Camera access.
 *
 * The failure modes here are what separate a demo from something usable on a phone in
 * a car park, so each one is named and turned into advice rather than surfacing as a
 * raw DOMException.
 *
 * The secure-context requirement catches people out: `getUserMedia` is unavailable on
 * plain HTTP except on `localhost`. Testing from a phone against a laptop dev server
 * over a LAN IP therefore fails no matter how correct the code is, and the app says so
 * explicitly instead of appearing broken.
 */

export type CameraErrorKind =
  | "insecure_context"
  | "unsupported"
  | "permission_denied"
  | "no_camera"
  | "in_use"
  | "unknown";

export interface CameraError {
  kind: CameraErrorKind;
  title: string;
  detail: string;
  /** Whether falling back to choosing a file from the gallery still makes sense. */
  canUploadInstead: boolean;
}

export function cameraSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getUserMedia === "function"
  );
}

/** True on https, localhost, or a file:// page - the contexts getUserMedia allows. */
export function secureContext(): boolean {
  if (typeof window === "undefined") return false;
  return window.isSecureContext === true;
}

const ERRORS: Record<CameraErrorKind, Omit<CameraError, "kind">> = {
  insecure_context: {
    title: "Camera needs a secure connection",
    detail:
      "Browsers only allow camera access over HTTPS. Open this page over HTTPS, or choose a photo from your gallery instead.",
    canUploadInstead: true,
  },
  unsupported: {
    title: "Camera not available in this browser",
    detail:
      "This browser does not support in-page camera capture. You can still choose a photo from your gallery.",
    canUploadInstead: true,
  },
  permission_denied: {
    title: "Camera permission was declined",
    detail:
      "To use the camera, allow access in your browser's address bar or site settings, then try again. You can also choose a photo from your gallery.",
    canUploadInstead: true,
  },
  no_camera: {
    title: "No camera found",
    detail: "This device does not appear to have a camera available. Choose a photo instead.",
    canUploadInstead: true,
  },
  in_use: {
    title: "Camera is busy",
    detail:
      "Another app or tab seems to be using the camera. Close it and try again, or choose a photo instead.",
    canUploadInstead: true,
  },
  unknown: {
    title: "Could not start the camera",
    detail: "Something prevented the camera from starting. You can choose a photo instead.",
    canUploadInstead: true,
  },
};

export function describeCameraError(error: unknown): CameraError {
  let kind: CameraErrorKind = "unknown";

  if (!secureContext()) {
    kind = "insecure_context";
  } else if (!cameraSupported()) {
    kind = "unsupported";
  } else if (error instanceof DOMException) {
    switch (error.name) {
      case "NotAllowedError":
      case "SecurityError":
        kind = "permission_denied";
        break;
      case "NotFoundError":
      case "OverconstrainedError":
        kind = "no_camera";
        break;
      case "NotReadableError":
      case "AbortError":
        kind = "in_use";
        break;
      default:
        kind = "unknown";
    }
  }

  return { kind, ...ERRORS[kind] };
}

/**
 * Open the rear camera at a resolution high enough for the analysis.
 *
 * `facingMode: "environment"` as an *ideal* rather than an exact constraint: on a
 * laptop there is no rear camera, and an exact constraint would fail outright instead
 * of falling back to the only camera present.
 */
export async function openCamera(): Promise<MediaStream> {
  if (!secureContext()) throw new DOMException("insecure context", "SecurityError");
  if (!cameraSupported()) throw new DOMException("unsupported", "NotSupportedError");

  return navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "environment" },
      // Ask for plenty of pixels: the backend refuses images whose tread band cannot
      // fill the analysis grid, and a 640x480 stream would routinely be refused.
      width: { ideal: 1920 },
      height: { ideal: 1440 },
    },
    audio: false,
  });
}

export function stopCamera(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop());
}

/** Grab the current video frame as a JPEG blob. */
export async function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) throw new Error("the camera is not ready yet");

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("could not capture the frame");
  context.drawImage(video, 0, 0, width, height);

  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.92),
  );
  if (!blob) throw new Error("could not capture the frame");
  return blob;
}
