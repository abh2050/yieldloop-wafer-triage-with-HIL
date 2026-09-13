import { useEffect } from "react";
import type { DefectPattern } from "@/api/types";

/**
 * Digit keys to defect classes.
 *
 * Digits rather than mnemonic letters: the nine classes have overlapping first
 * letters (none/near-full, edge-loc/edge-ring, center/scratch) and a reviewer
 * pressing the wrong key writes a wrong label into the training set. A fixed
 * numeric row is unambiguous and is always in the same physical place.
 */
export const CLASS_KEYS: Record<string, DefectPattern> = {
  "1": "center",
  "2": "donut",
  "3": "edge_loc",
  "4": "edge_ring",
  "5": "loc",
  "6": "near_full",
  "7": "random",
  "8": "scratch",
  "9": "none",
};

export const CLASS_KEY_BY_PATTERN: Record<DefectPattern, string> = Object.entries(
  CLASS_KEYS,
).reduce(
  (acc, [key, pattern]) => ({ ...acc, [pattern]: key }),
  {} as Record<DefectPattern, string>,
);

export interface KeyboardHandlers {
  onClass?: (pattern: DefectPattern) => void;
  onSkip?: () => void;
  onUndo?: () => void;
  onAccept?: () => void;
  onReject?: () => void;
  onNext?: () => void;
  onPrev?: () => void;
}

/**
 * Binds the review shortcuts to the document.
 *
 * Typing in a field never triggers a shortcut: a reviewer writing "1 scratch
 * near the notch" in a note must not have that labelled as centre nine times.
 */
export function useKeyboard(handlers: KeyboardHandlers, enabled = true): void {
  useEffect(() => {
    if (!enabled) return;

    function isTextEntry(target: EventTarget | null): boolean {
      if (!(target instanceof HTMLElement)) return false;
      const tag = target.tagName.toLowerCase();
      return tag === "input" || tag === "textarea" || target.isContentEditable;
    }

    function onKeyDown(event: KeyboardEvent) {
      if (isTextEntry(event.target)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      const pattern = CLASS_KEYS[event.key];
      if (pattern && handlers.onClass) {
        event.preventDefault();
        handlers.onClass(pattern);
        return;
      }

      switch (event.key) {
        case " ":
          if (handlers.onSkip) {
            event.preventDefault();
            handlers.onSkip();
          }
          break;
        case "u":
        case "U":
          if (handlers.onUndo) {
            event.preventDefault();
            handlers.onUndo();
          }
          break;
        case "a":
        case "A":
          if (handlers.onAccept) {
            event.preventDefault();
            handlers.onAccept();
          }
          break;
        case "r":
        case "R":
          if (handlers.onReject) {
            event.preventDefault();
            handlers.onReject();
          }
          break;
        case "ArrowRight":
        case "j":
          if (handlers.onNext) {
            event.preventDefault();
            handlers.onNext();
          }
          break;
        case "ArrowLeft":
        case "k":
          if (handlers.onPrev) {
            event.preventDefault();
            handlers.onPrev();
          }
          break;
        default:
          break;
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [handlers, enabled]);
}
