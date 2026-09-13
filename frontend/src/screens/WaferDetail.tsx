import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "@/api/client";
import { ConfidenceBar } from "@/components/ConfidenceBar";
import { WaferMap, WaferMapLegend } from "@/components/WaferMap";
import type { WaferDetail as WaferDetailData } from "@/api/types";

export function WaferDetail() {
  const { waferId = "" } = useParams();
  const [wafer, setWafer] = useState<WaferDetailData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api
      .wafer(waferId)
      .then(setWafer)
      .catch(() => setError(`No wafer ${waferId}.`));
  }, [waferId]);

  if (error) return <p role="alert" className="p-6 text-rose-300">{error}</p>;
  if (!wafer) return <p className="p-6 text-slate-400">Loading…</p>;

  return (
    <div className="flex flex-col gap-6 p-6" data-testid="wafer-detail">
      <header>
        <h1 className="font-mono text-xl font-semibold" data-testid="detail-wafer-id">
          {wafer.wafer_id}
        </h1>
        <p className="text-sm text-slate-400">
          Lot <span className="font-mono">{wafer.lot_name}</span> · split {wafer.split}
        </p>
      </header>

      <div className="flex flex-wrap gap-8">
        <div className="flex flex-col gap-3">
          <WaferMap
            grid={wafer.grid}
            height={wafer.grid_height}
            width={wafer.grid_width}
            size={340}
            label={wafer.wafer_id}
          />
          <WaferMapLegend />
        </div>

        <div className="flex min-w-[280px] flex-col gap-5">
          <section>
            <h2 className="mb-2 text-sm font-medium text-slate-300">Measured</h2>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs">
              <dt className="text-slate-500">die total</dt>
              <dd>{wafer.die_total}</dd>
              <dt className="text-slate-500">die failed</dt>
              <dd>{wafer.die_fail}</dd>
              <dt className="text-slate-500">failure rate</dt>
              <dd>{(wafer.failure_rate * 100).toFixed(2)}%</dd>
              <dt className="text-slate-500">dataset label</dt>
              <dd>{wafer.dataset_label ?? "unlabeled"}</dd>
            </dl>
          </section>

          <section>
            <h2 className="mb-2 text-sm font-medium text-slate-300">Model</h2>
            {wafer.show_prediction && wafer.predicted_label && wafer.confidence != null ? (
              <div className="flex flex-col gap-2" data-testid="detail-prediction">
                <p className="text-sm">{wafer.predicted_label.replace(/_/g, " ")}</p>
                <ConfidenceBar
                  confidence={wafer.confidence}
                  confidenceFloor={0.55}
                  autoCommitThreshold={0.95}
                  band={wafer.routing_band}
                />
              </div>
            ) : (
              <p
                data-testid="detail-prediction-withheld"
                className="rounded border border-slate-800 bg-slate-900/60 px-3 py-2 text-xs text-slate-400"
              >
                The prediction is withheld because its confidence is below the configured
                floor. It is hidden deliberately so that a weak model opinion cannot anchor
                your judgement.
              </p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
