import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, AlertCircle } from "lucide-react";

interface ResultProps {
  prediction: string;
  confidence: number;
  riskLevel: string;
  probabilities: Record<string, number>;
}

export const ResultCard = ({ prediction, confidence, riskLevel, probabilities }: ResultProps) => {
  const riskIcon = {
    High: <AlertTriangle className="w-5 h-5" />,
    Medium: <AlertCircle className="w-5 h-5" />,
    Low: <CheckCircle className="w-5 h-5" />,
  }[riskLevel];

  const riskColor = {
    High: "risk-high",
    Medium: "risk-medium",
    Low: "risk-low",
  }[riskLevel] || "";

  return (
    <div className="mt-8 p-6 rounded-2xl bg-card shadow-layered">
      <div className="flex items-center gap-3 mb-4">
        <span className={riskColor}>{riskIcon}</span>
        <h3 className="text-xl font-semibold">Result: {prediction}</h3>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-4 rounded-xl bg-muted">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Confidence</p>
          <p className="text-2xl font-semibold font-mono">{confidence}%</p>
        </div>
        <div className="p-4 rounded-xl bg-muted">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Risk Level</p>
          <p className={`text-2xl font-semibold ${riskColor}`}>{riskLevel}</p>
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Probabilities</p>
        {Object.entries(probabilities).map(([label, value]) => (
          <div key={label} className="flex items-center gap-3">
            <span className="text-sm w-28 text-muted-foreground">{label}</span>
            <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
              <div
                style={{ width: `${value}%` }}
                className="h-full rounded-full bg-primary"
              />
      </div>
            <span className="text-sm font-mono w-14 text-right">{value}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};
