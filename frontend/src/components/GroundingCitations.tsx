import type { Citation } from "@/api/types";

interface Props {
  evidenceIds: string[];
  citations: Record<string, Citation>;
  onFocus?: (evidenceId: string) => void;
}

/**
 * Renders a hypothesis's evidence references as clickable citations.
 *
 * Every id here resolved against the context bundle: the grounding gate dropped
 * any hypothesis whose citations did not, so this component never has to handle
 * a dead reference. If an id is nonetheless missing from the description map it
 * is rendered as unresolved rather than silently omitted, because a citation
 * quietly disappearing from a claim is precisely the failure the gate exists to
 * make impossible.
 */
export function GroundingCitations({ evidenceIds, citations, onFocus }: Props) {
  return (
    <ul className="flex flex-col gap-1" data-testid="citations">
      {evidenceIds.map((id) => {
        const citation = citations[id];
        return (
          <li key={id}>
            <button
              data-testid="citation"
              data-evidence-id={id}
              onClick={() => onFocus?.(id)}
              className="group flex w-full gap-2 rounded border border-slate-800 bg-slate-900/60 px-2 py-1.5 text-left hover:border-sky-600"
            >
              <code className="shrink-0 font-mono text-[11px] text-sky-400">{id}</code>
              <span className="text-xs text-slate-300">
                {citation ? (
                  citation.summary
                ) : (
                  <em data-testid="unresolved-citation" className="text-amber-400">
                    reference not present in the context bundle
                  </em>
                )}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
