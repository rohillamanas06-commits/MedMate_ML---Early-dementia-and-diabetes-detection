import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";

export default function FAQ() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background flex flex-col font-sans">
      <header className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur-md px-6 py-4">
        <Button variant="ghost" onClick={() => navigate(-1)} className="-ml-8 gap-2 hover:bg-foreground hover:text-background transition-colors">
          <ChevronLeft className="h-4 w-4" /> Back
        </Button>
      </header>
      <main className="flex-1 mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-4xl font-serif font-semibold tracking-tight mb-8">Frequently Asked Questions</h1>
        <div className="prose dark:prose-invert max-w-none text-muted-foreground space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-foreground">Is MedMate free to use?</h3>
            <p>Yes, MedMate offers a fully functional demo mode for trial purposes.</p>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">How accurate are the models?</h3>
            <p>Our models are trained on extensive clinical datasets. However, they should be used as an assistive tool, not as a definitive diagnostic metric.</p>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Is my data secure?</h3>
            <p>We prioritize privacy. All clinical assessments run locally in the browser interface, and no sensitive PHI is persisted permanently without authorization.</p>
          </div>
        </div>
      </main>
    </div>
  );
}
