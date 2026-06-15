import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";

export default function Privacy() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background flex flex-col font-sans">
      <header className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur-md px-6 py-4">
        <Button variant="ghost" onClick={() => navigate(-1)} className="-ml-4 gap-2 hover:bg-secondary hover:text-secondary-foreground transition-colors">
          <ChevronLeft className="h-4 w-4" /> Back
        </Button>
      </header>
      <main className="flex-1 mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-4xl font-serif font-semibold tracking-tight mb-8">Privacy Policy</h1>
        <div className="prose dark:prose-invert max-w-none text-muted-foreground">
          <p className="mb-4">Last updated: June 2026</p>
          <p className="mb-4">
            At MedMate, we take your privacy seriously. This Privacy Policy describes how your personal information is collected, used, and shared when you visit or use the application.
          </p>
          <h2 className="text-xl font-semibold text-foreground mt-6 mb-2">Data Collection</h2>
          <p className="mb-4">
            We collect the basic information required for account creation, such as email and name. Clinical assessment data is processed ephemerally to generate prediction results.
          </p>
          <h2 className="text-xl font-semibold text-foreground mt-6 mb-2">Data Protection</h2>
          <p>
            All network communication is encrypted via TLS. We implement industry-standard safeguards to secure any stored data against unauthorized access.
          </p>
        </div>
      </main>
    </div>
  );
}
