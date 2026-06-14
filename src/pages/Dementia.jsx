import { useNavigate } from "react-router-dom";
import { api } from "@/lib/api";
import { DEMENTIA_GROUPS } from "@/lib/medmateConfig";
import PredictionForm from "@/components/medmate/PredictionForm";

export default function Dementia() {
  const navigate = useNavigate();

  return (
    <div className="w-full px-6 py-8" data-testid="dementia-page">
      <div className="mm-fade-up">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">
          Dementia Prediction
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">Run a clinical assessment for early dementia detection based on NACC criteria.</p>
      </div>

      <div className="mt-8">
        <PredictionForm 
          type="dementia" 
          groups={DEMENTIA_GROUPS} 
          predictFn={api.predictDementia} 
          onResult={() => {}} 
        />
      </div>
    </div>
  );
}
