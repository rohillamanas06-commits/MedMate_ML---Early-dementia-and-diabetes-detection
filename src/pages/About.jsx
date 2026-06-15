import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";

export default function About() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background flex flex-col font-sans">
      <header className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur-md px-6 py-4">
        <Button variant="ghost" onClick={() => navigate(-1)} className="-ml-6 sm:-ml-4 gap-2 hover:bg-secondary hover:text-secondary-foreground transition-colors">
          <ChevronLeft className="h-4 w-4" /> Back
        </Button>
      </header>
      <main className="flex-1 mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-4xl font-serif font-semibold tracking-tight mb-8">About MedMate</h1>
        <div className="prose dark:prose-invert max-w-none text-muted-foreground">
          <p className="mb-4">
            MedMate is a state-of-the-art predictive analytics tool designed for clinical assessments.
            By leveraging advanced machine learning models, MedMate assists healthcare professionals in evaluating patient risk levels for readmission and dementia.
          </p>
          <p>
            Our mission is to empower decision-makers with data-driven insights, ensuring a more proactive and preventative approach to healthcare.
          </p>
        </div>
      </main>
    </div>
  );
}
