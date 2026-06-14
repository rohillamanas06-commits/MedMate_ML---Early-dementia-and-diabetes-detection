import { useNavigate } from "react-router-dom";
import { api } from "@/lib/api";
import { DIABETES_GROUPS } from "@/lib/medmateConfig";
import PredictionForm from "@/components/medmate/PredictionForm";

export default function Diabetes() {
  const navigate = useNavigate();

  return (
    <div className="w-full px-6 py-8" data-testid="diabetes-page">
      <div className="mm-fade-up">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">
          Diabetes Readmission
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">Run a clinical assessment for 30-day hospital readmission risk.</p>
      </div>

      <div className="mt-8">
        <PredictionForm 
          type="diabetes" 
          groups={DIABETES_GROUPS} 
          predictFn={api.predictDiabetes} 
          onResult={() => {}} 
        />
      </div>
    </div>
  );
}
