import { Button, Card, DataRow, Disclosure, Meter, StatusBadge } from "../components/ui";
import type { Inspection } from "../api/types";

/** Plain-language names for the quality issues the backend can report. */
const ISSUE_LABEL: Record<string, string> = {
  too_blurry: "Photo is blurred",
  too_dark: "Photo is too dark",
  too_bright: "Photo is too bright",
  low_contrast: "Not enough contrast",
  glare: "Glare on the rubber",
  noisy: "Too much image noise",
  over_compressed: "Image quality too low",
  resolution_too_low: "Taken from too far away",
  extreme_aspect: "Unusual framing",
  tread_not_located: "Tread area not clearly found",
  insufficient_texture: "Surface detail not visible",
};

const SURFACE_LABEL: Record<string, string> = {
  tread: "Tread",
  sidewall_or_shoulder: "Sidewall / shoulder",
  unclear: "Unclear",
};

function issueLabel(issue: string): string {
  return ISSUE_LABEL[issue] ?? issue.replace(/_/g, " ");
}

/**
 * The unassessable state.
 *
 * Deliberately not styled as an error. The system worked correctly and is telling the
 * user something true and actionable; presenting it in red with a warning triangle
 * would teach people that a careful refusal is a malfunction.
 */
function UnableToAssess({
  inspection,
  onRetake,
}: {
  inspection: Inspection;
  onRetake: () => void;
}) {
  const { image_quality: quality } = inspection;
  return (
    <div className="container stack">
      <Card className="verdict verdict--unknown">
        <div className="verdict__status">
          <StatusBadge severity="unknown" label={inspection.result.label} />
        </div>
        <h1 className="verdict__headline">{inspection.result.headline}</h1>
        <p className="verdict__detail">{inspection.result.meaning}</p>
      </Card>

      <Card>
        <h2 className="section-title">What's wrong</h2>
        <ul className="reasons">
          {quality.issues.map((issue, index) => (
            <li key={issue} className="reasons__item">
              <span className="reasons__mark" aria-hidden="true">
                •
              </span>
              <div>
                <strong>{issueLabel(issue)}</strong>
                {quality.advice[index] && <p>{quality.advice[index]}</p>}
              </div>
            </li>
          ))}
        </ul>
      </Card>

      <Button variant="primary" size="lg" block onClick={onRetake}>
        Retake photo
      </Button>
    </div>
  );
}

export function Result({
  inspection,
  onNewInspection,
  onRetake,
  label,
}: {
  inspection: Inspection;
  onNewInspection: () => void;
  onRetake: () => void;
  label: string;
}) {
  const { result, image_quality: quality, surface, evidence_features: evidence } = inspection;

  if (result.verdict === "unable_to_assess") {
    return <UnableToAssess inspection={inspection} onRetake={onRetake} />;
  }

  const confidence = result.confidence ?? 0;

  return (
    <div className="container stack">
      <Card className={`verdict verdict--${result.severity}`}>
        <div className="verdict__top">
          <span className="verdict__eyebrow">Tyre condition</span>
          <span className="verdict__tag">{label}</span>
        </div>
        <div className="verdict__status">
          <StatusBadge severity={result.severity} label={result.label} />
        </div>
        <h1 className="verdict__headline">{result.headline}</h1>
        <p className="verdict__detail">{result.detail}</p>

        {/* Surface and image quality as facts; confidence gets the meter below, so it
            is not stated twice. */}
        <dl className="verdict__facts">
          <div className="fact">
            <dt>Surface assessed</dt>
            <dd>{surface ? SURFACE_LABEL[surface.surface] ?? surface.surface : "—"}</dd>
          </div>
          <div className="fact">
            <dt>Image quality</dt>
            <dd>{quality.summary}</dd>
          </div>
        </dl>

        {/* An inconclusive result has a confidence of zero by definition - it sits
            inside the abstention band. Rendering that as "Confidence 0%" with an empty
            bar reads as a broken widget rather than as the honest statement it is, so
            the meter is replaced by the statement itself. Real-device testing showed
            the 0% bar first. */}
        {result.verdict === "inconclusive" ? (
          <p className="verdict__nomeasure">
            Not confident enough to call this either way.
          </p>
        ) : (
          result.confidence !== null && (
            <div className="verdict__meter">
              <Meter value={confidence} label="Confidence" severity={result.severity} />
            </div>
          )
        )}
      </Card>

      <Card>
        <h2 className="section-title">What this means</h2>
        <p className="prose">{result.meaning}</p>
        {surface && surface.surface !== "tread" && (
          <p className="callout callout--caution">{surface.note}</p>
        )}
        {quality.warnings.length > 0 && (
          <p className="callout callout--neutral">
            Worth noting: {quality.warnings.map(issueLabel).join(", ").toLowerCase()}.
          </p>
        )}
      </Card>

      <Card className="recommendation">
        <h2 className="section-title">What to do next</h2>
        {/* When the photo itself is a likely cause of a weak result, name the specific
            fix before the generic advice. "Get closer" is far more useful than "try a
            different angle" when the photograph was a whole-wheel shot. */}
        {quality.warning_advice.length > 0 && (
          <ul className="fixlist">
            {quality.warning_advice.map((advice) => (
              <li key={advice}>{advice}</li>
            ))}
          </ul>
        )}
        <p className="prose">{result.recommendation}</p>
      </Card>

      {evidence && (
        <Card>
          <h2 className="section-title">Why this result</h2>
          <p className="muted-note">
            These are measurements from your photo, shown against the range seen across
            the tyres this system was built from.
          </p>
          <ul className="evidence">
            {evidence.measurements.slice(0, 5).map((measurement) => (
              <li key={measurement.feature} className="evidence__item">
                <div className="evidence__head">
                  <span className="evidence__desc">{measurement.description}</span>
                  <span className={`evidence__band evidence__band--${measurement.band.replace(" ", "-")}`}>
                    {measurement.band}
                  </span>
                </div>
                <div className="evidence__track" aria-hidden="true">
                  <span
                    className="evidence__marker"
                    style={{ left: `${Math.min(98, Math.max(2, measurement.percentile))}%` }}
                  />
                </div>
                <span className="sr-only">
                  {measurement.description}: {measurement.band}, {Math.round(measurement.percentile)}th
                  percentile
                </span>
              </li>
            ))}
          </ul>
          {evidence.agreement && <p className="callout callout--neutral">{evidence.agreement}</p>}
        </Card>
      )}

      <Disclosure
        title="Technical analysis"
        subtitle="Processing stages, measurements and model details"
      >
        <TechnicalAnalysis inspection={inspection} />
      </Disclosure>

      <p className="disclaimer">{inspection.disclaimer}</p>

      <div className="stack-sm result__actions">
        <Button variant="primary" size="lg" block onClick={onNewInspection}>
          Inspect another tyre
        </Button>
        <Button variant="secondary" block onClick={onRetake}>
          Retake this photo
        </Button>
      </div>
    </div>
  );
}

/** Everything a technically-minded user might want, behind one disclosure. */
function TechnicalAnalysis({ inspection }: { inspection: Inspection }) {
  const { evidence, evidence_features: features, model, image_quality: quality, result } = inspection;

  const panels: Array<[string, string | null | undefined, string]> = [
    ["Original with detected area", evidence?.original, "The green or amber box is the region that was analysed."],
    ["Contrast-equalised", evidence?.enhanced, "After CLAHE, which normalises uneven lighting."],
    ["Analysed region", evidence?.roi, "Resampled to the fixed analysis grid."],
    ["Edge map", evidence?.edges, "Groove and edge structure found by the edge detector."],
    ["Frequency spectrum", evidence?.spectrum, "How texture energy is distributed across scales and directions."],
  ];

  return (
    <div className="tech stack">
      <section>
        <h3 className="tech__title">Processing stages</h3>
        <div className="tech__panels">
          {panels
            .filter(([, src]) => Boolean(src))
            .map(([title, src, caption]) => (
              <figure key={title} className="tech__panel">
                <img src={src as string} alt={title} loading="lazy" />
                <figcaption>
                  <strong>{title}</strong>
                  <span>{caption}</span>
                </figcaption>
              </figure>
            ))}
        </div>
        {!evidence && (
          <p className="muted-note">
            Visual analysis was skipped for this inspection (data saver was on).
          </p>
        )}
      </section>

      <section>
        <h3 className="tech__title">Quality checks</h3>
        <ul className="checks">
          {quality.checks.map((check) => (
            <li key={check.name} className={`checks__item ${check.passed ? "is-pass" : "is-fail"}`}>
              <span className="checks__glyph" aria-hidden="true">
                {check.passed ? "✓" : "✕"}
              </span>
              <span className="checks__name">{check.name.replace(/_/g, " ")}</span>
              <span className="checks__value">
                {check.value.toFixed(2)}
                <span className="checks__threshold"> / {check.threshold}</span>
              </span>
            </li>
          ))}
        </ul>
      </section>

      {features && features.measurements.length > 0 && (
        <section>
          <h3 className="tech__title">Feature measurements</h3>
          <dl className="datalist">
            {features.measurements.map((measurement) => (
              <DataRow
                key={measurement.feature}
                label={measurement.feature}
                hint={measurement.description}
                value={`${measurement.value.toFixed(4)} · p${Math.round(measurement.percentile)}`}
              />
            ))}
          </dl>
          {Object.keys(features.diagnostics).length > 0 && (
            <>
              <h3 className="tech__title tech__title--sub">Diagnostics</h3>
              <p className="muted-note">
                Computed and reported, but deliberately not used by the model.
              </p>
              <dl className="datalist">
                {Object.entries(features.diagnostics).map(([name, value]) => (
                  <DataRow key={name} label={name} value={value.toFixed(4)} />
                ))}
              </dl>
            </>
          )}
          <p className="muted-note">{features.note}</p>
        </section>
      )}

      <section>
        <h3 className="tech__title">Model</h3>
        <dl className="datalist">
          <DataRow label="Artifact" value={model.id} />
          {model.trained_on && <DataRow label="Trained on" value={model.trained_on} />}
          {model.validation && <DataRow label="Validation" value={model.validation} />}
          {model.metrics?.balanced_accuracy != null && (
            <DataRow
              label="Balanced accuracy"
              hint="cross-validated"
              value={`${(model.metrics.balanced_accuracy * 100).toFixed(1)}%`}
            />
          )}
          {model.metrics?.brier_score != null && (
            <DataRow label="Brier score" hint="lower is better" value={model.metrics.brier_score.toFixed(3)} />
          )}
          {model.metrics?.abstention_rate != null && (
            <DataRow
              label="Abstention rate"
              value={`${(model.metrics.abstention_rate * 100).toFixed(1)}%`}
            />
          )}
          <DataRow label="Decision threshold" value={result.decision_threshold.toFixed(2)} />
          <DataRow label="Abstention band" value={`±${result.abstain_band.toFixed(2)}`} />
          {result.probability_defect != null && (
            <DataRow
              label="Probability of a defect"
              hint="calibrated; not a tread depth"
              value={result.probability_defect.toFixed(3)}
            />
          )}
          {inspection.elapsed_ms != null && (
            <DataRow label="Processing time" value={`${Math.round(inspection.elapsed_ms)} ms`} />
          )}
        </dl>
      </section>

      {inspection.explanation && !inspection.explanation.exact && (
        <p className="muted-note">
          <strong>Model interpretation is not available.</strong>{" "}
          {inspection.explanation.method}
        </p>
      )}
    </div>
  );
}
