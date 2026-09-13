import { useEffect, useState } from "react";
import { api } from "@/api/client";
import type { ReasonCode } from "@/api/types";

interface Props {
  gate: string;
  action: string;
  value: string | null;
  onChange: (code: string) => void;
}

/**
 * The controlled vocabulary a reviewer must pick from.
 *
 * Required on every edit and reject, and enforced by the API rather than here:
 * a client-side-only requirement is one a direct API call skips, and the
 * resulting decision would be unusable as training signal.
 */
export function ReasonCodePicker({ gate, action, value, onChange }: Props) {
  const [codes, setCodes] = useState<ReasonCode[]>([]);

  useEffect(() => {
    let cancelled = false;
    void api
      .reasonCodes(gate, action)
      .then((loaded) => {
        if (!cancelled) setCodes(loaded);
      })
      .catch(() => {
        if (!cancelled) setCodes([]);
      });
    return () => {
      cancelled = true;
    };
  }, [gate, action]);

  return (
    <fieldset className="flex flex-col gap-1.5" data-testid="reason-code-picker">
      <legend className="mb-1 text-xs font-medium text-slate-300">
        Reason <span className="text-rose-400">*</span>
      </legend>
      {codes.map((code) => (
        <label
          key={code.code}
          data-testid={`reason-${code.code}`}
          className={`flex cursor-pointer gap-2 rounded border px-2 py-1.5 text-xs ${
            value === code.code
              ? "border-sky-500 bg-sky-950/40"
              : "border-slate-800 hover:border-slate-600"
          }`}
        >
          <input
            type="radio"
            name="reason-code"
            className="mt-0.5"
            checked={value === code.code}
            onChange={() => onChange(code.code)}
          />
          <span>
            <span className="block font-medium text-slate-200">{code.label}</span>
            <span className="block text-slate-400">{code.description}</span>
          </span>
        </label>
      ))}
    </fieldset>
  );
}
