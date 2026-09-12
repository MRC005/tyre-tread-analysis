import { Button, Card } from "../components/ui";
import type { ApiError } from "../api/client";

/**
 * A request that failed for reasons unrelated to the photograph.
 *
 * Distinct from "unable to assess": that is a valid result about the image, this is
 * the service not working. Conflating them would tell a user to retake a photograph
 * that was perfectly good.
 */
export function Failure({
  error,
  onRetry,
  onBack,
}: {
  error: ApiError;
  onRetry: () => void;
  onBack: () => void;
}) {
  const offline = typeof navigator !== "undefined" && navigator.onLine === false;

  return (
    <div className="container stack">
      <Card className="verdict verdict--unknown">
        <h1 className="verdict__headline">
          {offline ? "You appear to be offline" : "Couldn't complete the analysis"}
        </h1>
        <p className="verdict__detail">
          {offline
            ? "Check your connection and try again — your photo is still here."
            : error.message}
        </p>
      </Card>

      {error.code === "model_unavailable" && (
        <Card>
          <p className="muted-note">
            The service is running but has no analysis model loaded. If you are running
            this locally, check that a model artifact is present.
          </p>
        </Card>
      )}

      <div className="stack-sm">
        {(error.retryable || offline) && (
          <Button variant="primary" size="lg" block onClick={onRetry}>
            Try again
          </Button>
        )}
        <Button variant="secondary" block onClick={onBack}>
          Take a different photo
        </Button>
      </div>
    </div>
  );
}
