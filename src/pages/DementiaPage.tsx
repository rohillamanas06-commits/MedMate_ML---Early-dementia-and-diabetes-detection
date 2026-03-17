import { useState } from "react";
import { motion } from "framer-motion";
import { Brain, Loader2 } from "lucide-react";
import { FormField } from "@/components/FormField";
import { ResultCard } from "@/components/ResultCard";

interface FieldDef {
  key: string;
  label: string;
  description: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
}

const FIELDS: FieldDef[] = [
  { key: "Visit", label: "Visit Number", description: "Number of visits", min: 1, max: 5, step: 1, defaultValue: 1 },
  { key: "MR Delay", label: "MR Delay", description: "Days between visits", min: 0, max: 2000, step: 1, defaultValue: 0 },
  { key: "Age", label: "Age", description: "Patient age in years", min: 40, max: 100, step: 1, defaultValue: 70 },
  { key: "EDUC", label: "Education (years)", description: "Years of formal education", min: 1, max: 25, step: 1, defaultValue: 14 },
  { key: "SES", label: "SES", description: "Socioeconomic status (1=high, 5=low)", min: 1, max: 5, step: 0.5, defaultValue: 2 },
  { key: "MMSE", label: "MMSE Score", description: "Mini-Mental State Exam (0–30)", min: 0, max: 30, step: 1, defaultValue: 27 },
  { key: "CDR", label: "CDR", description: "Clinical Dementia Rating (0, 0.5, 1, 2)", min: 0, max: 3, step: 0.5, defaultValue: 0 },
  { key: "eTIV", label: "eTIV", description: "Estimated total intracranial volume (mm³)", min: 1000, max: 2500, step: 1, defaultValue: 1500 },
  { key: "nWBV", label: "nWBV", description: "Normalized whole brain volume", min: 0.5, max: 0.9, step: 0.001, defaultValue: 0.72 },
  { key: "ASF", label: "ASF", description: "Atlas scaling factor", min: 0.5, max: 1.5, step: 0.001, defaultValue: 1.0 },
];

interface Result {
  prediction: string;
  confidence: number;
  risk_level: string;
  probabilities: Record<string, number>;
}

const DementiaPage = () => {
  const [gender, setGender] = useState<"M" | "F">("M");
  const [values, setValues] = useState<Record<string, number>>(
    Object.fromEntries(FIELDS.map((f) => [f.key, f.defaultValue]))
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const payload: Record<string, unknown> = { "M/F": gender, ...values };

    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:5000";
      const res = await fetch(`${apiUrl}/predict/dementia`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Could not reach the prediction server. Make sure the Flask backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-16">
      <div className="max-w-2xl mx-auto px-6 py-16">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-muted flex items-center justify-center">
              <Brain className="w-5 h-5" />
            </div>
            <h1 className="text-3xl font-bold tracking-tight">Dementia Screening</h1>
          </div>
          <p className="text-muted-foreground mb-8">
            Enter MRI-derived biomarkers and cognitive assessment scores.
          </p>
        </motion.div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <FormField label="Sex">
            <div className="flex gap-2">
              {(["M", "F"] as const).map((g) => (
                <button
                  key={g}
                  type="button"
                  onClick={() => setGender(g)}
                  className={`flex-1 h-11 rounded-xl text-sm font-medium transition-all duration-200 ${
                    gender === g
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground hover:bg-accent"
                  }`}
                >
                  {g === "M" ? "Male" : "Female"}
                </button>
              ))}
            </div>
          </FormField>

          <div className="grid grid-cols-2 gap-4">
            {FIELDS.map((f) => (
              <FormField key={f.key} label={f.label} description={f.description}>
                <input
                  type="number"
                  value={values[f.key]}
                  onChange={(e) =>
                    setValues((prev) => ({ ...prev, [f.key]: Number(e.target.value) }))
                  }
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  className="w-full h-11 px-4 rounded-xl bg-muted text-foreground font-mono text-sm outline-none focus:ring-2 focus:ring-ring transition-all"
                />
              </FormField>
            ))}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full h-12 rounded-xl bg-primary text-primary-foreground font-medium flex items-center justify-center gap-2 hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-150 disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Run Prediction"}
          </button>
        </form>

        {error && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 text-sm text-destructive text-center"
          >
            {error}
          </motion.p>
        )}

        {result && (
          <ResultCard
            prediction={result.prediction}
            confidence={result.confidence}
            riskLevel={result.risk_level}
            probabilities={result.probabilities}
          />
        )}
      </div>
    </div>
  );
};

export default DementiaPage;
