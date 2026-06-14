import { useEffect, useState } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import { api } from "@/lib/api";
import { Card } from "@/components/ui/card";
import {
  ArrowLeft, Activity, Brain, Loader2, Trash2, Percent
} from "lucide-react";

const RISK_COLOR = { High: "#E2725B", Medium: "#CC7722", Low: "#71A6D2" };
const RISK_BG = { High: "#E2725B14", Medium: "#CC77221A", Low: "#71A6D214" };
const PROB_PAL = ["#71A6D2", "#E2725B", "#CC7722", "#7EC8A4"];

function ProbBar({ label, value, color, idx }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-semibold tabular-nums">{value}%</span>
      </div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value}%`, backgroundColor: color, transitionDelay: `${idx * 100}ms` }}
        />
      </div>
    </div>
  );
}

export default function HistoryDetail() {
  const { id } = useParams();
  const { state } = useLocation();
  const navigate = useNavigate();

  const [item, setItem] = useState(state?.item || null);
  const [loading, setLoading] = useState(!state?.item);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (item) return;
    api.getHistoryItem(Number(id))
      .then(setItem)
      .catch((e) => setError(e?.error || "Record not found"))
      .finally(() => setLoading(false));
  }, [id]);

  const handleDelete = async () => {
    if (!window.confirm("Delete this record permanently?")) return;
    setDeleting(true);
    try {
      await api.deleteHistoryItem(Number(id));
      navigate("/dashboard", { replace: true });
    } catch (e) {
      alert(e?.error || "Delete failed");
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground">
        <Loader2 className="h-6 w-6 animate-spin" />
      </div>
    );
  }

  if (error || !item) {
    return (
      <div className="flex h-64 flex-col items-center justify-center gap-3">
        <p className="text-muted-foreground">{error || "Record not found."}</p>
        <button onClick={() => navigate("/dashboard")} className="text-sm text-primary underline underline-offset-4">
          Back to Dashboard
        </button>
      </div>
    );
  }

  const color = RISK_COLOR[item.risk_level] || RISK_COLOR.Medium;
  const bg = RISK_BG[item.risk_level] || RISK_BG.Medium;
  const probs = Object.entries(item.probabilities || {});
  const inputs = Object.entries(item.inputs || {});
  const isDemo = item.model_type === "diabetes";

  return (
    <div className="flex flex-col h-full min-h-[calc(100vh-65px)]">

      {/* ── Top bar ──────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-8 py-4 border-b border-border bg-background/60 backdrop-blur-sm sticky top-[65px] z-10">
        <button
          onClick={() => navigate("/dashboard")}
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </button>

        <div className="flex items-center gap-3">
          {/* Type badge */}
          <span className="flex items-center gap-2 text-sm text-muted-foreground">
            {isDemo
              ? <Activity className="h-4 w-4 text-primary" />
              : <Brain className="h-4 w-4 text-primary" />}
            {isDemo ? "Diabetes Readmission" : "Dementia Assessment"}
          </span>

          {/* Risk badge — no icon */}
          <span
            className="rounded-full px-3 py-1 text-xs font-semibold"
            style={{ backgroundColor: bg, color }}
          >
            {item.risk_level} Risk
          </span>

          {/* Delete — icon only */}
          <button
            onClick={handleDelete}
            disabled={deleting}
            title="Delete record"
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-red-500/40 text-red-500 hover:bg-red-500/10 transition-colors disabled:opacity-40"
          >
            {deleting
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : <Trash2 className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* ── Main content — two columns ────────────────────────────────── */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-0 divide-y lg:divide-y-0 lg:divide-x divide-border">

        {/* ── LEFT: Result ──────────────────────────────────────────────── */}
        <div className="flex flex-col justify-center px-10 py-12 space-y-10">

          {/* Prediction */}
          <div>
            <p className="text-xs uppercase tracking-widest text-muted-foreground mb-2">Prediction</p>
            <p className="font-display text-5xl font-semibold leading-tight" style={{ color }}>
              {item.prediction}
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              {new Date(item.created_at).toLocaleString()}
            </p>
          </div>

          {/* Confidence */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs uppercase tracking-widest text-muted-foreground flex items-center gap-1">
                <Percent className="h-3 w-3" /> Model Confidence
              </p>
              <p className="font-display text-3xl font-semibold tabular-nums">{item.confidence}%</p>
            </div>
            <div className="h-3 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full transition-all duration-700 delay-100"
                style={{ width: `${item.confidence}%`, backgroundColor: color }}
              />
            </div>
          </div>

          {/* Probabilities */}
          {probs.length > 0 && (
            <div className="space-y-4">
              <p className="text-xs uppercase tracking-widest text-muted-foreground">Probability Breakdown</p>
              {probs.map(([k, v], i) => (
                <ProbBar key={k} label={k} value={v} color={PROB_PAL[i % PROB_PAL.length]} idx={i} />
              ))}
            </div>
          )}
        </div>

        {/* ── RIGHT: Inputs ─────────────────────────────────────────────── */}
        <div className="flex flex-col px-10 py-12">
          <p className="text-xs uppercase tracking-widest text-muted-foreground mb-6">Input Parameters</p>
          {inputs.length > 0 ? (
            <div className="grid grid-cols-2 gap-x-8 gap-y-6 xl:grid-cols-3">
              {inputs.map(([k, v]) => (
                <div key={k} className="flex flex-col gap-1">
                  <span className="text-xs uppercase tracking-wide text-muted-foreground/80">
                    {k.replace(/_/g, " ")}
                  </span>
                  <span className="text-base font-medium">{String(v)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No input data recorded.</p>
          )}
        </div>
      </div>
    </div>
  );
}
