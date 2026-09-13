import { useEffect, useState } from "react";
import { api } from "@/api/client";
import type { AuditRecord, ChainVerification, GuardrailAction } from "@/api/types";

/**
 * The audit trail, with its chain verification shown at the top.
 *
 * Verification is surfaced rather than buried because an audit log nobody checks
 * is a filing cabinet, not a control.
 */
export function AuditTrail() {
  const [records, setRecords] = useState<AuditRecord[]>([]);
  const [actions, setActions] = useState<GuardrailAction[]>([]);
  const [chain, setChain] = useState<ChainVerification | null>(null);
  const [tab, setTab] = useState<"records" | "guardrails">("records");

  useEffect(() => {
    void api.audit(100).then((page) => setRecords(page.records)).catch(() => setRecords([]));
    void api.guardrailActions(100).then(setActions).catch(() => setActions([]));
    void api.verifyChain().then(setChain).catch(() => setChain(null));
  }, []);

  return (
    <div className="flex flex-col gap-4 p-6" data-testid="audit-trail">
      <header>
        <h1 className="text-xl font-semibold">Audit trail</h1>
        <p className="text-sm text-slate-400">
          Append-only. Every model output, human decision, guardrail action, and threshold
          change.
        </p>
      </header>

      {chain && (
        <div
          data-testid="chain-status"
          className={`rounded border px-3 py-2 text-sm ${
            chain.intact
              ? "border-emerald-800 bg-emerald-950/40 text-emerald-200"
              : "border-rose-700 bg-rose-950/50 text-rose-200"
          }`}
        >
          {chain.intact ? (
            <>Hash chain intact across {chain.checked.toLocaleString()} records.</>
          ) : (
            <>
              Chain broken at sequence {chain.broken_at.join(", ")}. Records have been
              altered outside the application.
            </>
          )}
        </div>
      )}

      <div className="flex gap-2 text-sm">
        {(["records", "guardrails"] as const).map((name) => (
          <button
            key={name}
            data-testid={`tab-${name}`}
            onClick={() => setTab(name)}
            className={`rounded px-3 py-1 ${
              tab === name ? "bg-slate-800 text-slate-100" : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {name === "records" ? "Audit records" : "Guardrail actions"}
          </button>
        ))}
      </div>

      {tab === "records" ? (
        <table className="w-full text-xs" data-testid="audit-records">
          <thead className="text-left uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-1 pr-4">Seq</th>
              <th className="pr-4">When</th>
              <th className="pr-4">Event</th>
              <th className="pr-4">Actor</th>
              <th className="pr-4">Subject</th>
              <th>Digest</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              <tr key={record.sequence} data-testid="audit-row" className="border-t border-slate-800">
                <td className="py-1 pr-4 font-mono">{record.sequence}</td>
                <td className="pr-4 font-mono text-slate-400">
                  {new Date(record.occurred_at).toISOString().replace("T", " ").slice(0, 19)}
                </td>
                <td className="pr-4">{record.event_type.replace(/_/g, " ")}</td>
                <td className="pr-4 font-mono text-slate-400">{record.actor}</td>
                <td className="pr-4 font-mono text-slate-400">
                  {record.subject_type}/{record.subject_id.slice(0, 8)}
                </td>
                <td className="font-mono text-slate-600">{record.digest.slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <table className="w-full text-xs" data-testid="guardrail-actions">
          <thead className="text-left uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-1 pr-4">When</th>
              <th className="pr-4">Stage</th>
              <th className="pr-4">Outcome</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {actions.map((action) => (
              <tr key={action.id} data-testid="guardrail-row" className="border-t border-slate-800">
                <td className="py-1 pr-4 font-mono text-slate-400">
                  {new Date(action.created_at).toISOString().replace("T", " ").slice(0, 19)}
                </td>
                <td className="pr-4">{action.stage.replace(/_/g, " ")}</td>
                <td className="pr-4">
                  <span
                    className={
                      action.outcome === "blocked"
                        ? "text-rose-300"
                        : action.outcome === "modified"
                          ? "text-amber-300"
                          : "text-slate-400"
                    }
                  >
                    {action.outcome}
                  </span>
                </td>
                <td className="font-mono text-slate-400">{action.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
