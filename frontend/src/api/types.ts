// Mirrors the Pydantic response models in src/yieldloop/api/routes/.
// A contract test asserts these stay in step with the live OpenAPI schema, so a
// backend field rename fails the build rather than surfacing as undefined in a
// reviewer's browser.

export type DefectPattern =
  | "center"
  | "donut"
  | "edge_loc"
  | "edge_ring"
  | "loc"
  | "near_full"
  | "random"
  | "scratch"
  | "none";

export type RoutingBand = "auto_commit" | "uncertainty_band" | "below_floor";
export type TaskGate = "label" | "confirm" | "escalation";
export type DecisionAction = "accept" | "edit" | "reject";

export const DEFECT_PATTERNS: DefectPattern[] = [
  "center",
  "donut",
  "edge_loc",
  "edge_ring",
  "loc",
  "near_full",
  "random",
  "scratch",
  "none",
];

export interface QueueItem {
  task_id: string;
  wafer_id: string;
  lot_name: string;
  gate: TaskGate;
  priority: number;
  /**
   * False below the confidence floor. When false the prediction fields are not
   * merely hidden here, they are absent from the response entirely.
   */
  show_prediction: boolean;
  grid_height: number;
  grid_width: number;
  die_total: number;
  die_fail: number;
  failure_rate: number;
  predicted_label?: string | null;
  confidence?: number | null;
  routing_band?: RoutingBand | null;
}

export interface QueueResponse {
  items: QueueItem[];
  depth: number;
}

export interface TriageQueueResponse extends QueueResponse {
  confidence_floor: number;
  auto_commit_threshold: number;
}

export interface ReasonCode {
  code: string;
  label: string;
  description: string;
}

export interface DecisionPayload {
  task_id: string;
  action: DecisionAction;
  chosen_label?: DefectPattern | null;
  reason_code?: string | null;
  note?: string | null;
  decision_ms: number;
}

export interface DecisionResponse {
  decision_id: string;
  task_id: string;
  chosen_label: string | null;
  is_override: boolean;
  prediction_was_shown: boolean;
}

export interface WaferDetail {
  wafer_id: string;
  lot_name: string;
  split: string;
  grid_height: number;
  grid_width: number;
  grid: number[];
  die_total: number;
  die_fail: number;
  failure_rate: number;
  dataset_label: string | null;
  predicted_label: string | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  routing_band: RoutingBand | null;
  show_prediction: boolean;
}

export interface Citation {
  evidence_id: string;
  kind: string;
  summary: string;
}

export interface Hypothesis {
  rank: number;
  cause_category: string;
  statement: string;
  evidence_ids: string[];
  supporting_signal: string;
  contradicting_signal: string | null;
  confidence: number;
  confirming_query: string;
  eliminating_query: string;
}

export interface HypothesisEnvelope {
  request_id: string;
  lot_id: string;
  lot_name: string;
  /** A first-class outcome, not an error. Rendered as "insufficient evidence". */
  abstained: boolean;
  abstention_reason: string | null;
  hypotheses: Hypothesis[];
  citations: Record<string, Citation>;
  injection_suspected: boolean;
  context_gaps: string[];
  degraded: boolean;
  returned_count: number;
  grounded_count: number;
  evidence_ids: string[];
}

export interface ModelHealth {
  active_classifier: string | null;
  label_count: number | null;
  metrics: Record<string, unknown>;
  routing: { confidence_floor: number; auto_commit_threshold: number };
  queue_depth: number;
  decisions: number;
  override_rate: number;
  override_rate_when_shown: number;
  latest_drift: Record<string, unknown> | null;
}

export interface AuditRecord {
  sequence: number;
  occurred_at: string;
  event_type: string;
  actor: string;
  subject_type: string;
  subject_id: string;
  request_id: string;
  payload: Record<string, unknown>;
  digest: string;
}

export interface AuditPage {
  records: AuditRecord[];
  total: number;
}

export interface GuardrailAction {
  id: string;
  created_at: string;
  request_id: string;
  stage: string;
  outcome: string;
  reason: string;
  detail: Record<string, unknown>;
}

export interface ChainVerification {
  checked: number;
  intact: boolean;
  broken_at: number[];
}

export interface ApiErrorBody {
  reason: string;
  message: string;
  detail: Record<string, unknown>;
  request_id: string | null;
}
