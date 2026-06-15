import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Loader2, RotateCcw, Play, User, Activity, History, FlaskConical, Stethoscope, Brain, HeartPulse } from "lucide-react";
import Field from "@/components/medmate/Field";
import ResultCard from "@/components/medmate/ResultCard";
import { buildDefaults } from "@/lib/medmateConfig";
import { useToast } from "@/hooks/use-toast";

const ICONS = { user: User, activity: Activity, history: History, flask: FlaskConical, stethoscope: Stethoscope, brain: Brain, heart: HeartPulse };

export default function PredictionForm({ type, groups, predictFn, onResult }) {
  const defaults = useMemo(() => buildDefaults(groups), [groups]);
  const [values, setValues] = useState(defaults);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const { toast } = useToast();

  const onChange = (name, val) => setValues((v) => ({ ...v, [name]: val }));
  const reset = () => { setValues(defaults); setResult(null); };

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await predictFn(values);
      setResult(res);
      onResult?.(res);
      setTimeout(() => document.getElementById("result-anchor")?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 80);
    } catch (e) {
      toast({ title: "Prediction failed", description: e.error || "Something went wrong", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="flex flex-col gap-8 w-full">
      <div className="w-full space-y-5">
        {groups.map((group) => {
          const GIcon = ICONS[group.icon] || Activity;
          return (
            <Card key={group.title} className="rounded-2xl border-border bg-card p-5">
              <div className="mb-4 flex items-center gap-2">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <GIcon className="h-4 w-4" />
                </span>
                <h3 className="font-display text-base font-medium">{group.title}</h3>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                {group.fields.map((f) => (
                  <Field key={f.name} field={f} value={values[f.name]} onChange={onChange} />
                ))}
              </div>
            </Card>
          );
        })}

        <div className="flex items-center gap-3">
          <Button
            type="submit"
            disabled={loading}
            data-testid={`predict-${type}-btn`}
            className="rounded-xl bg-primary px-6 hover:bg-primary/90 transition-transform hover:-translate-y-px"
          >
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Run Assessment
          </Button>
          <Button variant="outline" onClick={reset} disabled={loading || !result} className="rounded-xl" data-testid={`reset-${type}-btn`}>
            <RotateCcw className="mr-2 h-4 w-4" /> Reset
          </Button>
        </div>
      </div>

      {result && (
        <div className="w-full mt-4">
          <div id="result-anchor">
            <ResultCard result={result} type={type} />
          </div>
        </div>
      )}
    </form>
  );
}
