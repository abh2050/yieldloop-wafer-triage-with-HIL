import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/api/client";
import { WaferMap, WaferMapLegend } from "@/components/WaferMap";
import { KeyboardHints } from "@/components/KeyboardHints";
import { CLASS_KEYS, useKeyboard } from "@/hooks/useKeyboard";
import { useDecision } from "@/hooks/useDecision";
import { useQueue } from "@/hooks/useQueue";
import type { DefectPattern } from "@/api/types";

/**
 * The label gate.
 *
 * Built first and optimized for keyboard-only operation. One wafer is focused
 * at a time and a digit key both labels it and advances, so a decision is a
 * single keystroke.
 *
 * No model prediction is shown here at any confidence. The point of this gate is
 * an unanchored human label on a wafer the sampler found informative; showing a
 * guess would turn the label into agreement with the model and make it useless
 * as training signal.
 */
export function LabelGrid() {
  const { items, depth, loading, error, reload, remove } = useQueue("label", 24);
  const { submit, submitting, error: decisionError, markPresented } = useDecision();
  const [focused, setFocused] = useState(0);
  const [grids, setGrids] = useState<Record<string, number[]>>({});
  const [decided, setDecided] = useState(0);
  const [lastDecisionMs, setLastDecisionMs] = useState<number | null>(null);
  const [startedAt] = useState(() => performance.now());

  const current = items[focused];

  useEffect(() => {
    if (current) markPresented();
  }, [current?.task_id, markPresented, current]);

  // Prefetch grids for what is on screen so a keystroke never waits on a fetch.
  useEffect(() => {
    let cancelled = false;
    const missing = items.filter((item) => !(item.wafer_id in grids)).slice(0, 24);
    if (missing.length === 0) return;
    void Promise.all(
      missing.map(async (item) => {
        try {
          const grid = await api.waferGrid(item.wafer_id);
          if (!cancelled) setGrids((prev) => ({ ...prev, [item.wafer_id]: grid }));
        } catch {
          // A missing grid leaves a placeholder rather than blocking the queue.
        }
      }),
    );
    return () => {
      cancelled = true;
    };
  }, [items, grids]);

  const label = useCallback(
    async (pattern: DefectPattern) => {
      if (!current || submitting) return;
      const began = performance.now();
      try {
        await submit({ taskId: current.task_id, action: "edit", label: pattern, reasonCode: "wrong_class" });
        setLastDecisionMs(Math.round(performance.now() - began));
        setDecided((n) => n + 1);
        remove(current.task_id);
        setFocused((index) => Math.min(index, Math.max(0, items.length - 2)));
      } catch {
        // useDecision surfaces the message; the item stays for a retry.
      }
    },
    [current, submit, submitting, remove, items.length],
  );

  const handlers = useMemo(
    () => ({
      onClass: label,
      onSkip: () => setFocused((i) => Math.min(i + 1, items.length - 1)),
      onNext: () => setFocused((i) => Math.min(i + 1, items.length - 1)),
      onPrev: () => setFocused((i) => Math.max(i - 1, 0)),
    }),
    [label, items.length],
  );

  useKeyboard(handlers, !loading && items.length > 0);

  const rate = decided > 0 ? (performance.now() - startedAt) / decided / 1000 : null;

  return (
    <div className="flex h-full flex-col gap-4 p-6" data-testid="label-grid">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Label queue</h1>
          <p className="text-sm text-slate-400">
            Most informative unlabeled wafers. No model prediction is shown here.
          </p>
        </div>
        <div className="flex items-center gap-5 font-mono text-sm">
          <span data-testid="queue-depth">
            <span className="text-slate-400">queue </span>
            {depth}
          </span>
          <span data-testid="decided-count">
            <span className="text-slate-400">decided </span>
            {decided}
          </span>
          {rate !== null && (
            <span data-testid="seconds-per-decision">
              <span className="text-slate-400">s/decision </span>
              {rate.toFixed(1)}
            </span>
          )}
        </div>
      </header>

      <KeyboardHints
        hints={[
          { keys: "1-9", action: "assign class" },
          { keys: "space", action: "skip" },
          { keys: "← →", action: "move focus" },
        ]}
      />

      {decisionError && (
        <div role="alert" data-testid="decision-error" className="rounded border border-rose-700 bg-rose-950/50 px-3 py-2 text-sm text-rose-200">
          {decisionError}
        </div>
      )}
      {error && (
        <div role="alert" data-testid="queue-error" className="rounded border border-rose-700 bg-rose-950/50 px-3 py-2 text-sm text-rose-200">
          {error}{" "}
          <button className="underline" onClick={() => void reload()}>
            retry
          </button>
        </div>
      )}

      {loading && <p className="text-slate-400">Loading queue…</p>}

      {!loading && items.length === 0 && !error && (
        <div data-testid="empty-queue" className="rounded border border-slate-700 p-8 text-center text-slate-400">
          <p className="font-medium text-slate-200">Nothing to label</p>
          <p className="mt-1 text-sm">
            The label queue is empty. Run an active learning round to select the next batch.
          </p>
        </div>
      )}

      <div className="grid flex-1 grid-cols-2 gap-6 overflow-hidden lg:grid-cols-[320px_1fr]">
        {current && (
          <section className="flex flex-col gap-3" data-testid="focused-wafer">
            <WaferMap
              grid={grids[current.wafer_id] ?? []}
              height={current.grid_height}
              width={current.grid_width}
              size={300}
              label={current.wafer_id}
            />
            <WaferMapLegend />
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs text-slate-300">
              <dt className="text-slate-500">wafer</dt>
              <dd data-testid="focused-wafer-id">{current.wafer_id}</dd>
              <dt className="text-slate-500">lot</dt>
              <dd>{current.lot_name}</dd>
              <dt className="text-slate-500">die failed</dt>
              <dd>
                {current.die_fail} / {current.die_total} ({(current.failure_rate * 100).toFixed(1)}%)
              </dd>
            </dl>
            <div className="grid grid-cols-3 gap-1.5">
              {Object.entries(CLASS_KEYS).map(([key, pattern]) => (
                <button
                  key={key}
                  data-testid={`class-${pattern}`}
                  disabled={submitting}
                  onClick={() => void label(pattern)}
                  className="flex items-center justify-between rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-left text-xs hover:border-slate-500 disabled:opacity-50"
                >
                  <span>{pattern.replace(/_/g, " ")}</span>
                  <kbd className="ml-2 rounded bg-slate-800 px-1 font-mono text-slate-400">{key}</kbd>
                </button>
              ))}
            </div>
            {lastDecisionMs !== null && (
              <p className="font-mono text-[11px] text-slate-500" data-testid="last-decision-ms">
                last decision {lastDecisionMs} ms
              </p>
            )}
          </section>
        )}

        <section className="overflow-y-auto" aria-label="Queue">
          <div className="grid grid-cols-[repeat(auto-fill,minmax(104px,1fr))] gap-2">
            {items.map((item, index) => (
              <button
                key={item.task_id}
                data-testid="queue-tile"
                data-wafer-id={item.wafer_id}
                onClick={() => setFocused(index)}
                className={`rounded border p-1 transition ${
                  index === focused ? "border-sky-400 ring-1 ring-sky-400" : "border-slate-800 hover:border-slate-600"
                }`}
              >
                <WaferMap
                  grid={grids[item.wafer_id] ?? []}
                  height={item.grid_height}
                  width={item.grid_width}
                  size={92}
                  label={item.wafer_id}
                />
                <span className="mt-1 block truncate font-mono text-[10px] text-slate-400">
                  {item.wafer_id}
                </span>
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
