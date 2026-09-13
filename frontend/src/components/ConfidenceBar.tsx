import type { RoutingBand } from "@/api/types";

interface ConfidenceBarProps {
  confidence: number;
  confidenceFloor: number;
  autoCommitThreshold: number;
  band?: RoutingBand | null;
}

/**
 * Shows a calibrated confidence against the routing bands.
 *
 * The thresholds are drawn as markers rather than baked into the colour scale,
 * because they are configuration the eval harness sweeps. A reviewer seeing
 * "just below the line" is seeing the actual configured line.
 */
export function ConfidenceBar({
  confidence,
  confidenceFloor,
  autoCommitThreshold,
  band,
}: ConfidenceBarProps) {
  const percent = Math.max(0, Math.min(1, confidence)) * 100;
  const tone =
    band === "auto_commit"
      ? "bg-emerald-500"
      : band === "below_floor"
        ? "bg-rose-500"
        : "bg-amber-400";

  return (
    <div className="w-full" data-testid="confidence-bar">
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full ${tone}`} style={{ width: `${percent}%` }} />
        <span
          className="absolute top-0 h-full w-px bg-slate-400"
          style={{ left: `${confidenceFloor * 100}%` }}
          title={`Confidence floor ${confidenceFloor}`}
        />
        <span
          className="absolute top-0 h-full w-px bg-slate-300"
          style={{ left: `${autoCommitThreshold * 100}%` }}
          title={`Auto-commit ${autoCommitThreshold}`}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[11px] text-slate-400">
        <span data-testid="confidence-value">{(confidence * 100).toFixed(1)}%</span>
        <span>{band?.replace(/_/g, " ") ?? ""}</span>
      </div>
    </div>
  );
}
