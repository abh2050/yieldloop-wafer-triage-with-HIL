import { useState } from "react";
import { api, ApiError } from "@/api/client";
import { GroundingCitations } from "@/components/GroundingCitations";
import { ReasonCodePicker } from "@/components/ReasonCodePicker";
import type { HypothesisEnvelope } from "@/api/types";

/**
 * The escalation gate.
 *
 * This screen renders only hypotheses that passed the guardrail layer. The API
 * returns grounded ones alone, and no client path bypasses it.
 *
 * It renders an abstention as a full answer and shows the reason. Naming the
 * missing evidence tells the reviewer something. A blank panel reads as a
 * broken feature and trains reviewers to ignore the screen.
 *
 * It states the degraded mode plainly when the breaker is open. The rest of the
 * console keeps working, and the reviewer needs to tell "no hypothesis" apart
 * from "hypotheses are unavailable".
 */
export function HypothesisCard() {
  const [lotName, setLotName] = useState("");
  const [note, setNote] = useState("");
  const [result, setResult] = useState<HypothesisEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [rejecting, setRejecting] = useState<number | null>(null);
  const [reasonCode, setReasonCode] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.hypothesis(lotName.trim(), note.trim() || undefined));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-5 p-6" data-testid="hypothesis-card">
      <header>
        <h1 className="text-xl font-semibold">Root cause hypotheses</h1>
        <p className="text-sm text-slate-400">
          Every claim must cite retrieved evidence. Unsupported claims are dropped before
          they reach this screen.
        </p>
      </header>

      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Lot
          <input
            data-testid="lot-input"
            value={lotName}
            onChange={(event) => setLotName(event.target.value)}
            placeholder="lot00891"
            className="w-48 rounded border border-slate-700 bg-slate-900 px-2 py-1.5 font-mono text-sm text-slate-100"
          />
        </label>
        <label className="flex flex-1 flex-col gap-1 text-xs text-slate-400">
          Reviewer note (optional)
          <input
            data-testid="note-input"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-100"
          />
        </label>
        <button
          data-testid="generate"
          disabled={!lotName.trim() || loading}
          onClick={() => void run()}
          className="rounded bg-sky-600 px-3 py-1.5 text-sm font-medium hover:bg-sky-500 disabled:opacity-50"
        >
          {loading ? "Working…" : "Generate"}
        </button>
      </div>

      {error && (
        <p role="alert" data-testid="hypothesis-error" className="rounded border border-rose-700 bg-rose-950/50 px-3 py-2 text-sm text-rose-200">
          {error}
        </p>
      )}

      {result?.degraded && (
        <p data-testid="degraded-banner" className="rounded border border-amber-700 bg-amber-950/40 px-3 py-2 text-sm text-amber-200">
          Hypothesis generation is unavailable and the console is running in
          classifier-only mode. Wafer maps, predictions, and the review queue are
          unaffected.
        </p>
      )}

      {result?.injection_suspected && (
        <p data-testid="injection-banner" className="rounded border border-amber-700 bg-amber-950/40 px-3 py-2 text-sm text-amber-200">
          Instruction-shaped content was detected in the text supplied to this request. It
          was isolated as data and never treated as instructions. The detection is recorded
          in the audit trail.
        </p>
      )}

      {result?.abstained && (
        <section data-testid="abstention" className="rounded border border-slate-700 bg-slate-900/60 p-5">
          <h2 className="text-sm font-medium text-slate-200">Insufficient evidence</h2>
          <p className="mt-1 text-sm text-slate-400" data-testid="abstention-reason">
            {result.abstention_reason}
          </p>
          {result.returned_count > 0 && (
            <p className="mt-2 font-mono text-xs text-slate-500">
              {result.returned_count} hypothes{result.returned_count === 1 ? "is" : "es"} proposed,{" "}
              {result.grounded_count} survived grounding.
            </p>
          )}
        </section>
      )}

      {result && !result.abstained && (
        <ol className="flex flex-col gap-4" data-testid="hypothesis-list">
          {result.hypotheses.map((hypothesis) => (
            <li
              key={hypothesis.rank}
              data-testid="hypothesis"
              data-rank={hypothesis.rank}
              className="rounded border border-slate-700 bg-slate-900/50 p-4"
            >
              <div className="flex items-baseline justify-between gap-3">
                <h3 className="font-medium text-slate-100">
                  <span className="mr-2 font-mono text-slate-500">#{hypothesis.rank}</span>
                  {hypothesis.cause_category.replace(/_/g, " ")}
                </h3>
                <span className="font-mono text-xs text-slate-400">
                  {(hypothesis.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <p className="mt-2 text-sm text-slate-200">{hypothesis.statement}</p>

              <dl className="mt-3 grid gap-1 text-xs">
                <dt className="text-slate-500">Supporting</dt>
                <dd className="text-slate-300">{hypothesis.supporting_signal}</dd>
                {hypothesis.contradicting_signal && (
                  <>
                    <dt className="text-slate-500">Contradicting</dt>
                    <dd className="text-slate-300">{hypothesis.contradicting_signal}</dd>
                  </>
                )}
                <dt className="text-slate-500">Would confirm</dt>
                <dd className="text-slate-300">{hypothesis.confirming_query}</dd>
                <dt className="text-slate-500">Would eliminate</dt>
                <dd className="text-slate-300">{hypothesis.eliminating_query}</dd>
              </dl>

              <div className="mt-3">
                <p className="mb-1 text-xs text-slate-500">Evidence</p>
                <GroundingCitations
                  evidenceIds={hypothesis.evidence_ids}
                  citations={result.citations}
                />
              </div>

              <div className="mt-3 flex gap-2">
                <button
                  data-testid={`reject-${hypothesis.rank}`}
                  onClick={() =>
                    setRejecting(rejecting === hypothesis.rank ? null : hypothesis.rank)
                  }
                  className="rounded border border-slate-600 px-2 py-1 text-xs hover:border-rose-500"
                >
                  Reject
                </button>
              </div>

              {rejecting === hypothesis.rank && (
                <div className="mt-3 rounded border border-slate-800 p-3" data-testid="reject-panel">
                  <ReasonCodePicker
                    gate="escalation"
                    action="reject"
                    value={reasonCode}
                    onChange={setReasonCode}
                  />
                  <p className="mt-2 text-[11px] text-slate-500">
                    A reason is required. It is what makes the rejection usable as signal.
                  </p>
                </div>
              )}
            </li>
          ))}
        </ol>
      )}

      {result && result.context_gaps.length > 0 && (
        <section data-testid="context-gaps" className="text-xs text-slate-400">
          <h3 className="mb-1 font-medium text-slate-300">Context gaps reported</h3>
          <ul className="list-inside list-disc">
            {result.context_gaps.map((gap) => (
              <li key={gap}>{gap}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
