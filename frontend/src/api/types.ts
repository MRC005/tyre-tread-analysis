/**
 * Types mirroring the FastAPI response contract.
 *
 * The backend's Pydantic schemas are the source of truth; these follow them. When the
 * contract changes, backend schema, these types, the tests and the docs change
 * together.
 */

export type Verdict =
  | "likely_serviceable"
  | "defect_suspected"
  | "inconclusive"
  | "unable_to_assess";

export type Severity = "ok" | "caution" | "alert" | "unknown";

export interface Result {
  verdict: Verdict;
  severity: Severity;
  /** Short status label shown as the taxonomy, e.g. "Healthy". */
  label: string;
  headline: string;
  detail: string;
  /** What the result means in practice — the "so what" for a driver. */
  meaning: string;
  recommendation: string;
  /** Calibrated probability of a visible defect. Null when unassessable. Not a depth. */
  probability_defect: number | null;
  confidence: number | null;
  decision_threshold: number;
  abstain_band: number;
}

export interface QualityCheck {
  name: string;
  passed: boolean;
  value: number;
  threshold: number;
}

export interface ImageQuality {
  usable: boolean;
  summary: "Good" | "Acceptable" | "Unusable";
  issues: string[];
  warnings: string[];
  /** One actionable sentence per blocking issue. */
  advice: string[];
  /** Guidance for non-blocking caveats that may still explain a weak result. */
  warning_advice: string[];
  metrics: Record<string, number>;
  checks: QualityCheck[];
}

export interface SurfaceInfo {
  surface: "tread" | "sidewall_or_shoulder" | "unclear";
  tread_probability: number;
  note: string;
}

export interface FeatureEvidence {
  feature: string;
  description: string;
  value: number;
  percentile: number;
  band: "very low" | "low" | "typical" | "high" | "very high";
  resembles: string | null;
}

export interface EvidenceFeatures {
  measurements: FeatureEvidence[];
  agreement: string | null;
  diagnostics: Record<string, number>;
  note: string;
}

export interface Contribution {
  feature: string;
  value: number;
  z_score: number;
  contribution: number;
  direction: string;
  description: string;
}

export interface Explanation {
  /** False means no causal breakdown is available and none was invented. */
  exact: boolean;
  method: string;
  reasons: string[];
  contributions: Contribution[];
}

export interface Evidence {
  original?: string | null;
  enhanced?: string | null;
  roi?: string | null;
  edges?: string | null;
  spectrum?: string | null;
}

export interface ModelMetrics {
  balanced_accuracy?: number | null;
  balanced_accuracy_std?: number | null;
  roc_auc?: number | null;
  brier_score?: number | null;
  expected_calibration_error?: number | null;
  abstention_rate?: number | null;
  balanced_accuracy_on_decided?: number | null;
  quality_gate_pass_rate?: number | null;
}

export interface ModelInfo {
  id: string;
  artifact_format_version: number;
  trained_on?: string | null;
  n_training_samples?: number | null;
  validation?: string | null;
  metrics?: ModelMetrics | null;
}

export interface Inspection {
  result: Result;
  image_quality: ImageQuality;
  measurements: Record<string, number>;
  model: ModelInfo;
  disclaimer: string;
  surface?: SurfaceInfo | null;
  explanation?: Explanation | null;
  evidence_features?: EvidenceFeatures | null;
  evidence?: Evidence | null;
  features?: Record<string, number> | null;
  elapsed_ms?: number | null;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  model_loaded: boolean;
  model_id?: string | null;
  detail?: string | null;
}

/** Stable machine-readable codes the backend returns. */
export type ApiErrorCode =
  | "file_too_large"
  | "unsupported_media_type"
  | "undecodable_image"
  | "empty_upload"
  | "model_unavailable"
  | "inspection_timeout"
  | "inspection_failed"
  | "network"
  | "aborted"
  | "unknown";
