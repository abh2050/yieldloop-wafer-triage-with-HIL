import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { LabelGrid } from "@/screens/LabelGrid";
import { TriageQueue } from "@/screens/TriageQueue";
import { WaferDetail } from "@/screens/WaferDetail";
import { HypothesisCard } from "@/screens/HypothesisCard";
import { ModelHealth } from "@/screens/ModelHealth";
import { AuditTrail } from "@/screens/AuditTrail";

const NAV = [
  { to: "/label", label: "Label" },
  { to: "/triage", label: "Triage" },
  { to: "/hypotheses", label: "Hypotheses" },
  { to: "/health", label: "Model health" },
  { to: "/audit", label: "Audit" },
];

export function App() {
  return (
    <div className="flex h-full flex-col">
      <nav className="flex items-center gap-1 border-b border-slate-800 px-4 py-2">
        <span className="mr-4 font-mono text-sm font-semibold tracking-tight text-sky-400">
          yieldloop
        </span>
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            data-testid={`nav-${item.to.slice(1)}`}
            className={({ isActive }) =>
              `rounded px-3 py-1 text-sm ${
                isActive ? "bg-slate-800 text-slate-100" : "text-slate-400 hover:text-slate-200"
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/label" replace />} />
          <Route path="/label" element={<LabelGrid />} />
          <Route path="/triage" element={<TriageQueue />} />
          <Route path="/wafer/:waferId" element={<WaferDetail />} />
          <Route path="/hypotheses" element={<HypothesisCard />} />
          <Route path="/health" element={<ModelHealth />} />
          <Route path="/audit" element={<AuditTrail />} />
        </Routes>
      </main>
    </div>
  );
}
