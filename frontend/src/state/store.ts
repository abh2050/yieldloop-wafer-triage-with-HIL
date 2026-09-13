import { create } from "zustand";

interface ConsoleState {
  reviewerId: string;
  setReviewerId: (id: string) => void;
  /** Set when the breaker is open and hypothesis generation is unavailable. */
  degraded: boolean;
  setDegraded: (value: boolean) => void;
}

/**
 * Console-wide state.
 *
 * Deliberately small. Queue contents and decisions are server state and are not
 * mirrored here: a client-side copy of the queue is a copy that can disagree
 * with the database about what has already been decided.
 */
export const useConsoleStore = create<ConsoleState>((set) => ({
  reviewerId: localStorage.getItem("yieldloop.reviewer") ?? "local-reviewer",
  setReviewerId: (id) => {
    localStorage.setItem("yieldloop.reviewer", id);
    set({ reviewerId: id });
  },
  degraded: false,
  setDegraded: (value) => set({ degraded: value }),
}));
