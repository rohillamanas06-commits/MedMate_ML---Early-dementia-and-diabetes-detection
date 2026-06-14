import { Card } from "@/components/ui/card";
import { ShieldAlert, ShieldCheck, ShieldQuestion, Sparkles } from "lucide-react";

const RISK_STYLES = {
  High: { color: "#E2725B", bg: "bg-[#E2725B]/10", text: "text-[#b6452f]", icon: ShieldAlert, ring: "ring-[#E2725B]/30" },
  Medium: { color: "#CC7722", bg: "bg-[#CC7722]/10", text: "text-[#9a5a14]", icon: ShieldQuestion, ring: "ring-[#CC7722]/30" },
  Low: { color: "#71A6D2", bg: "bg-[#71A6D2]/12", text: "text-[#3f76a3]", icon: ShieldCheck, ring: "ring-[#71A6D2]/30" },
};

function Bar({ label, value, color, idx }) {
  return (
    <div className="flex flex-col gap-1" data-testid={`prob-bar-${label.replace(/[^a-z0-9]/gi, "-").toLowerCase()}`}>
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-semibold tabular-nums">{value}%</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className="mm-bar-grow h-full rounded-full"
          style={{ width: `${value}%`, backgroundColor: color, animationDelay: `${idx * 120}ms` }}
        />
      </div>
    </div>
  );
}

export default function ResultCard({ result, type }) {
  if (!result) return null;
  const risk = RISK_STYLES[result.risk_level] || RISK_STYLES.Medium;
  const Icon = risk.icon;
  const probs = Object.entries(result.probabilities || {});
  const palette = ["#71A6D2", "#E2725B"];

  return (
    <Card
      data-testid="result-card"
      className={`mm-fade-up overflow-hidden rounded-2xl border-border ring-1 ${risk.ring} bg-card`}
    >
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          {type === "diabetes" ? "Readmission Assessment" : "Dementia Assessment"}
        </div>
        <span data-testid="result-risk-badge" className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${risk.bg} ${risk.text}`}>
          {result.risk_level} Risk
        </span>
      </div>

      <div className="grid gap-6 p-6 sm:grid-cols-2">
        <div className="flex flex-col justify-center">
          <p className="text-xs uppercase tracking-widest text-muted-foreground">Prediction</p>
          <p data-testid="result-prediction" className="mt-1 font-display text-3xl font-medium leading-tight" style={{ color: risk.color }}>
            {result.prediction}
          </p>

          <div className="mt-5">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Model Confidence</span>
              <span data-testid="result-confidence" className="font-display text-xl font-semibold tabular-nums">{result.confidence}%</span>
            </div>
            <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-secondary">
              <div className="mm-bar-grow h-full rounded-full" style={{ width: `${result.confidence}%`, backgroundColor: risk.color }} />
            </div>
          </div>
        </div>

        <div className="flex flex-col justify-center gap-3 rounded-xl bg-secondary/40 p-4">
          <p className="text-xs uppercase tracking-widest text-muted-foreground">Probability Breakdown</p>
          {probs.map(([k, v], i) => (
            <Bar key={k} label={k} value={v} color={palette[i % palette.length]} idx={i} />
          ))}
        </div>
      </div>
    </Card>
  );
}
