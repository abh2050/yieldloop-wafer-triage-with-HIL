import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "@/api/client";
import { ConfidenceBar } from "@/components/ConfidenceBar";
import type { TriageQueueResponse } from "@/api/types";

/**
 * The confirm gate: uncertainty-band predictions awaiting a human.
 *
 * Rows below the confidence floor arrive with no prediction at all and are
 * rendered as withheld. That is not a display choice made here -- the API omits
 * the fields -- and the explicit "withheld" state exists so a reviewer
 * understands the model has an opinion that is deliberately not being shown,
 * rather than assuming it had none.
 */
export function TriageQueue() {
  const [data, setData] = useState<TriageQueueResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api
      .triageQueue(50)
      .then(setData)
      .catch(() => setError("Could not load the triage queue."));
  }, []);

  if (error) {
    return (
      <p role="alert" className="p-6 text-rose-300">
        {error}
      </p>
    );
  }
  if (!data) return <p className="p-6 text-slate-400">Loading…</p>;

  return (
    <div className="flex flex-col gap-4 p-6" data-testid="triage-queue">
      <header>
        <h1 className="text-xl font-semibold">Triage queue</h1>
        <p className="text-sm text-slate-400">
          Predictions inside the uncertainty band. Auto-commit at{" "}
          <span className="font-mono">{data.auto_commit_threshold}</span>, floor at{" "}
          <span className="font-mono">{data.confidence_floor}</span>.
        </p>
      </header>

      {data.items.length === 0 ? (
        <div data-testid="empty-triage" className="rounded border border-slate-700 p-8 text-center text-slate-400">
          Nothing awaiting confirmation.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-2">Wafer</th>
              <th>Lot</th>
              <th>Failure rate</th>
              <th>Prediction</th>
              <th className="w-56">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((item) => (
              <tr key={item.task_id} data-testid="triage-row" className="border-t border-slate-800">
                <td className="py-2 font-mono text-xs">
                  <Link className="text-sky-400 hover:underline" to={`/wafer/${item.wafer_id}`}>
                    {item.wafer_id}
                  </Link>
                </td>
                <td className="font-mono text-xs text-slate-400">{item.lot_name}</td>
                <td className="font-mono text-xs">{(item.failure_rate * 100).toFixed(1)}%</td>
                <td>
                  {item.show_prediction && item.predicted_label ? (
                    <span data-testid="prediction-shown">{item.predicted_label.replace(/_/g, " ")}</span>
                  ) : (
                    <span
                      data-testid="prediction-withheld"
                      title="Below the confidence floor: withheld so it cannot anchor the reviewer"
                      className="text-slate-500"
                    >
                      withheld
                    </span>
                  )}
                </td>
                <td>
                  {item.show_prediction && item.confidence != null ? (
                    <ConfidenceBar
                      confidence={item.confidence}
                      confidenceFloor={data.confidence_floor}
                      autoCommitThreshold={data.auto_commit_threshold}
                      band={item.routing_band}
                    />
                  ) : (
                    <span className="font-mono text-xs text-slate-600">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
