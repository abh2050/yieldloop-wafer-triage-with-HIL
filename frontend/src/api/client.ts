import type {
  ApiErrorBody,
  AuditPage,
  ChainVerification,
  DecisionPayload,
  DecisionResponse,
  GuardrailAction,
  HypothesisEnvelope,
  ModelHealth,
  QueueResponse,
  ReasonCode,
  TriageQueueResponse,
  WaferDetail,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

/**
 * An error the API reported deliberately, carrying its stable reason code.
 *
 * The code is what the UI branches on. Matching on message text would break the
 * moment a guardrail reworded its explanation.
 */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly reason: string,
    message: string,
    readonly detail: Record<string, unknown> = {},
    readonly requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function reviewerId(): string {
  // Injected by the fab SSO proxy in a real deployment. Locally it is settable
  // so a developer's decisions are attributed to them and not to "unknown".
  return localStorage.getItem("yieldloop.reviewer") ?? "local-reviewer";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-Reviewer-Id": reviewerId(),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      // A non-JSON error body means something upstream of the app failed.
    }
    throw new ApiError(
      response.status,
      body?.reason ?? "http_error",
      body?.message ?? `${response.status} ${response.statusText}`,
      body?.detail ?? {},
      body?.request_id ?? null,
    );
  }
  return (await response.json()) as T;
}

export const api = {
  labelQueue: (limit = 25) =>
    request<QueueResponse>(`/label/queue?limit=${limit}`),

  reasonCodes: (gate: string, action: string) =>
    request<ReasonCode[]>(`/label/reason-codes?gate=${gate}&action=${action}`),

  waferGrid: (waferId: string) =>
    request<number[]>(`/label/wafer/${encodeURIComponent(waferId)}/grid`),

  submitDecision: (payload: DecisionPayload) =>
    request<DecisionResponse>("/label/decision", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  triageQueue: (limit = 25) =>
    request<TriageQueueResponse>(`/triage/queue?limit=${limit}`),

  wafer: (waferId: string) =>
    request<WaferDetail>(`/triage/wafer/${encodeURIComponent(waferId)}`),

  hypothesis: (lotName: string, reviewerNote?: string) =>
    request<HypothesisEnvelope>("/hypothesis", {
      method: "POST",
      body: JSON.stringify({ lot_name: lotName, reviewer_note: reviewerNote ?? null }),
    }),

  modelHealth: () => request<ModelHealth>("/health/model"),

  audit: (limit = 50) => request<AuditPage>(`/audit?limit=${limit}`),

  guardrailActions: (limit = 50) =>
    request<GuardrailAction[]>(`/audit/guardrails?limit=${limit}`),

  verifyChain: () => request<ChainVerification>("/audit/verify"),

  setReviewer: (id: string) => localStorage.setItem("yieldloop.reviewer", id),
  reviewer: reviewerId,
};
