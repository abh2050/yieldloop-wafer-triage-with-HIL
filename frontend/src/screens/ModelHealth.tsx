import { useEffect, useState } from "react";
import { api } from "@/api/client";
import type { ModelHealth as ModelHealthData } from "@/api/types";

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-900/50 p-3">
      <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-lg text-slate-100">{value}</p>
      {hint && <p className="mt-1 text-[11px] text-slate-500">{hint}</p>}
    </div>
  );
}

/**
 * Model health.
 *
 * Shows override rate twice: overall, and restricted to decisions where the
 * prediction was visible. The gap between them is the anchoring effect, and
 * reporting only the first would hide it.
 */
export function ModelHealth() {
  const [health, setHealth] = useState<ModelHealthData | null>(null);

  useEffect(() => {
    void api.modelHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  if (!health) return <p className="p-6 text-slate-400">Loading…</p>;

  const metrics = health.metrics as Record<string, number | string | undefined>;
  const recall = (health.metrics.val_per_class_recall ?? {}) as Record<string, number>;
  const support = (health.metrics.val_support ?? {}) as Record<string, number>;

  return (
    <div className="flex flex-col gap-6 p-6" data-testid="model-health">
      <header>
        <h1 className="text-xl font-semibold">Model health</h1>
        <p className="text-sm text-slate-400">
          Active classifier{" "}
          <span className="font-mono">
            {health.active_classifier ? health.active_classifier.slice(0, 12) : "none"}
          </span>
          {health.label_count != null && <> · trained on {health.label_count.toLocaleString()} labels</>}
        </p>
      </header>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat
          label="Val accuracy"
          value={metrics.val_accuracy != null ? `${(Number(metrics.val_accuracy) * 100).toFixed(1)}%` : "—"}
        />
        <Stat
          label="Macro F1"
          value={metrics.val_macro_f1 != null ? Number(metrics.val_macro_f1).toFixed(3) : "—"}
          hint="Unweighted across classes"
        />
        <Stat
          label="ECE (calibrated)"
          value={metrics.val_ece_calibrated != null ? Number(metrics.val_ece_calibrated).toFixed(4) : "—"}
          hint={
            metrics.val_ece_uncalibrated != null
              ? `was ${Number(metrics.val_ece_uncalibrated).toFixed(4)}`
              : undefined
          }
        />
        <Stat
          label="Temperature"
          value={metrics.temperature != null ? Number(metrics.temperature).toFixed(3) : "—"}
          hint={metrics.temperature_at_bound ? "pinned at bound, so confidences are not trustworthy" : undefined}
        />
        <Stat label="Queue depth" value={String(health.queue_depth)} />
        <Stat label="Decisions" value={String(health.decisions)} />
        <Stat
          label="Override rate"
          value={`${(health.override_rate * 100).toFixed(1)}%`}
          hint="All decisions"
        />
        <Stat
          label="Override (shown)"
          value={`${(health.override_rate_when_shown * 100).toFixed(1)}%`}
          hint="The gap against the overall rate measures anchoring"
        />
      </div>

      <section>
        <h2 className="mb-2 text-sm font-medium text-slate-300">
          Per-class recall on validation
        </h2>
        <p className="mb-3 text-xs text-slate-500">
          Reported per class because the distribution is extreme: a majority-class
          predictor scores about 85% accuracy on this dataset.
        </p>
        <table className="text-sm" data-testid="per-class-recall">
          <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-1 pr-6">Class</th>
              <th className="pr-6">Recall</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(recall)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, value]) => (
                <tr key={cls} className="border-t border-slate-800">
                  <td className="py-1 pr-6">{cls.replace(/_/g, " ")}</td>
                  <td className="pr-6 font-mono text-xs">{value.toFixed(3)}</td>
                  <td className="font-mono text-xs text-slate-400">
                    {(support[cls] ?? 0).toLocaleString()}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>

      <section className="text-xs text-slate-400">
        <h2 className="mb-1 text-sm font-medium text-slate-300">Routing</h2>
        <p className="font-mono">
          floor {health.routing.confidence_floor} · auto-commit{" "}
          {health.routing.auto_commit_threshold}
        </p>
      </section>
    </div>
  );
}
