import { useCallback, useRef, useState } from "react";
import { api, ApiError } from "@/api/client";
import type { DecisionAction, DefectPattern } from "@/api/types";

interface SubmitArgs {
  taskId: string;
  action: DecisionAction;
  label?: DefectPattern | null;
  reasonCode?: string | null;
  note?: string | null;
}

/**
 * Submits decisions and measures how long each one took.
 *
 * The timer starts when a task is *presented*, not when the component mounts,
 * and the elapsed value is sent with the decision. The sub-four-second claim is
 * measured from that column rather than estimated, so the measurement has to be
 * taken here, at the only point that knows when the reviewer actually saw the
 * wafer.
 */
export function useDecision() {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const presentedAt = useRef<number>(performance.now());

  const markPresented = useCallback(() => {
    presentedAt.current = performance.now();
  }, []);

  const submit = useCallback(async (args: SubmitArgs) => {
    setSubmitting(true);
    setError(null);
    const elapsed = Math.max(0, Math.round(performance.now() - presentedAt.current));
    try {
      const response = await api.submitDecision({
        task_id: args.taskId,
        action: args.action,
        chosen_label: args.label ?? null,
        reason_code: args.reasonCode ?? null,
        note: args.note ?? null,
        decision_ms: elapsed,
      });
      presentedAt.current = performance.now();
      return response;
    } catch (caught) {
      const message =
        caught instanceof ApiError ? caught.message : "Could not record the decision.";
      setError(message);
      throw caught;
    } finally {
      setSubmitting(false);
    }
  }, []);

  return { submit, submitting, error, markPresented, clearError: () => setError(null) };
}
