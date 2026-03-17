import { useState } from "react";
import { motion } from "framer-motion";
import { Droplets, Loader2 } from "lucide-react";
import { FormField, ToggleOption } from "@/components/FormField";
import { ResultCard } from "@/components/ResultCard";

const SYMPTOMS = [
  "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
  "Polyphagia", "Genital thrush", "visual blurring", "Itching",
  "Irritability", "delayed healing", "partial paresis",
  "muscle stiffness", "Alopecia", "Obesity",
];

const SYMPTOM_LABELS: Record<string, string> = {
  Polyuria: "Excessive Urination",
  Polydipsia: "Excessive Thirst",
  "sudden weight loss": "Sudden Weight Loss",
  weakness: "Weakness",
  Polyphagia: "Excessive Hunger",
  "Genital thrush": "Genital Thrush",
  "visual blurring": "Visual Blurring",
  Itching: "Itching",
  Irritability: "Irritability",
  "delayed healing": "Delayed Healing",
  "partial paresis": "Partial Paresis",
  "muscle stiffness": "Muscle Stiffness",
  Alopecia: "Hair Loss (Alopecia)",
  Obesity: "Obesity",
};

interface Result {
  prediction: string;
  confidence: number;
  risk_level: string;
  probabilities: Record<string, number>;
}

const DiabetesPage = () => {
  const [age, setAge] = useState(45);
  const [gender, setGender] = useState<"Male" | "Female">("Male");
  const [symptoms, setSymptoms] = useState<Record<string, boolean>>(
    Object.fromEntries(SYMPTOMS.map((s) => [s, false]))
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const payload: Record<string, unknown> = {
      Age: age,
      Gender: gender,
    };
    SYMPTOMS.forEach((s) => {
      payload[s] = symptoms[s] ? "Yes" : "No";
    });

    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:5000";
      const res = await fetch(`${apiUrl}/predict/diabetes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      // Add artificial delay for smoother UI
      await new Promise((resolve) => setTimeout(resolve, 500));
      setResult(data);
    } catch {
      setError("Could not reach the prediction server. Make sure the Flask backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-14 sm:pt-16">
      <div className="max-w-2xl mx-auto px-4 sm:px-6 py-8 sm:py-12 md:py-16">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-muted flex items-center justify-center">
              <Droplets className="w-4 h-4 sm:w-5 sm:h-5" />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Diabetes Screening</h1>
          </div>
          <p className="text-sm sm:text-base text-muted-foreground mb-6 sm:mb-8">
            Enter patient details and symptom profile. All 16 features are used for prediction.
          </p>
        </motion.div>

        <form onSubmit={handleSubmit} className="space-y-5 sm:space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            <FormField label="Age">
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(Number(e.target.value))}
                min={1}
                max={120}
                className="w-full h-10 sm:h-11 px-3 sm:px-4 rounded-xl bg-muted text-foreground font-mono text-sm outline-none focus:ring-2 focus:ring-ring transition-all"
              />
            </FormField>
            <FormField label="Gender">
              <div className="flex gap-2">
                {(["Male", "Female"] as const).map((g) => (
                  <button
                    key={g}
                    type="button"
                    onClick={() => setGender(g)}
                    className={`flex-1 h-10 sm:h-11 rounded-xl text-sm font-medium transition-all duration-200 ${
                      gender === g
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-muted-foreground hover:bg-accent"
                    }`}
                  >
                    {g}
                  </button>
                ))}
              </div>
            </FormField>
          </div>

          <div>
            <h2 className="text-sm font-medium text-foreground mb-3">Symptoms</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {SYMPTOMS.map((s) => (
                <ToggleOption
                  key={s}
                  label={SYMPTOM_LABELS[s]}
                  value={symptoms[s]}
                  onChange={(v) => setSymptoms((prev) => ({ ...prev, [s]: v }))}
                />
              ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full h-11 sm:h-12 rounded-xl bg-primary text-primary-foreground font-medium flex items-center justify-center gap-2 hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-150 disabled:opacity-50 text-sm sm:text-base"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Run Prediction"}
          </button>
        </form>

        {error && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 text-xs sm:text-sm text-destructive text-center px-4"
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

export default DiabetesPage;