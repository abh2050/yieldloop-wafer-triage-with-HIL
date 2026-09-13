import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "@/api/client";
import type { QueueItem } from "@/api/types";

interface QueueState {
  items: QueueItem[];
  depth: number;
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
  /** Drop one item locally after it is decided, without a refetch. */
  remove: (taskId: string) => void;
}

/**
 * Loads a review queue.
 *
 * Decided items are removed locally rather than triggering a refetch. Waiting
 * for a round trip between every decision is exactly the latency the keyboard
 * flow exists to avoid; the queue is refilled in the background instead.
 */
export function useQueue(gate: "label" | "confirm", limit = 25): QueueState {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [depth, setDepth] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response =
        gate === "label" ? await api.labelQueue(limit) : await api.triageQueue(limit);
      setItems(response.items);
      setDepth(response.depth);
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught.message : "Could not load the review queue.",
      );
    } finally {
      setLoading(false);
    }
  }, [gate, limit]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const remove = useCallback((taskId: string) => {
    setItems((current) => current.filter((item) => item.task_id !== taskId));
    setDepth((current) => Math.max(0, current - 1));
  }, []);

  return { items, depth, loading, error, reload, remove };
}
