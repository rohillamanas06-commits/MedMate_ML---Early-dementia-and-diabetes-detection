import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";

export default function Terms() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background flex flex-col font-sans">
      <header className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur-md px-6 py-4">
        <Button variant="ghost" onClick={() => navigate(-1)} className="-ml-6 sm:-ml-4 gap-2 hover:bg-secondary hover:text-secondary-foreground transition-colors">
          <ChevronLeft className="h-4 w-4" /> Back
        </Button>
      </header>
      <main className="flex-1 mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-4xl font-serif font-semibold tracking-tight mb-8">Terms & Conditions</h1>
        <div className="prose dark:prose-invert max-w-none text-muted-foreground">
          <p className="mb-4">Last updated: June 2026</p>
          <p className="mb-4">
            By accessing or using MedMate, you agree to be bound by these Terms.
            If you disagree with any part of the terms, then you may not access the service.
          </p>
          <h2 className="text-xl font-semibold text-foreground mt-6 mb-2">Usage Limitations</h2>
          <p className="mb-4">
            MedMate is provided for informational and clinical assistance purposes only. It is not intended to substitute professional medical advice, diagnosis, or treatment.
          </p>
          <h2 className="text-xl font-semibold text-foreground mt-6 mb-2">Liability</h2>
          <p>
            In no event shall MedMate, nor its directors, employees, partners, agents, suppliers, or affiliates, be liable for any indirect, incidental, special, consequential or punitive damages resulting from your access to or use of the service.
          </p>
        </div>
      </main>
    </div>
  );
}
