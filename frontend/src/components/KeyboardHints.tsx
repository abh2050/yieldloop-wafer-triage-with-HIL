interface Hint {
  keys: string;
  action: string;
}

/**
 * The shortcut legend, always visible.
 *
 * Kept on screen rather than behind a help modal: a reviewer who has to
 * remember nine class keys will use the mouse instead, and the throughput claim
 * dies there.
 */
export function KeyboardHints({ hints }: { hints: Hint[] }) {
  return (
    <div
      data-testid="keyboard-hints"
      className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400"
    >
      {hints.map((hint) => (
        <span key={hint.keys} className="flex items-center gap-1.5">
          <kbd className="rounded border border-slate-600 bg-slate-800 px-1.5 py-0.5 font-mono text-slate-200">
            {hint.keys}
          </kbd>
          {hint.action}
        </span>
      ))}
    </div>
  );
}
