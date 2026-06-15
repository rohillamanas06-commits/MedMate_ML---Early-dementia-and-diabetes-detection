import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";

export default function Cookies() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background flex flex-col font-sans">
      <header className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur-md px-6 py-4">
        <Button variant="ghost" onClick={() => navigate(-1)} className="-ml-4 gap-2 hover:bg-secondary hover:text-secondary-foreground transition-colors">
          <ChevronLeft className="h-4 w-4" /> Back
        </Button>
      </header>
      <main className="flex-1 mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-4xl font-serif font-semibold tracking-tight mb-8">Cookies Policy</h1>
        <div className="prose dark:prose-invert max-w-none text-muted-foreground">
          <p className="mb-4">Last updated: June 2026</p>
          <p className="mb-4">
            MedMate uses cookies to improve your experience on our site, analyze site usage, and assist in our marketing efforts.
          </p>
          <h2 className="text-xl font-semibold text-foreground mt-6 mb-2">Essential Cookies</h2>
          <p className="mb-4">
            These cookies are necessary for the website to function properly, including saving your session and theme preferences (e.g. Light/Dark mode).
          </p>
          <h2 className="text-xl font-semibold text-foreground mt-6 mb-2">Analytics</h2>
          <p>
            We may use third-party analytics cookies to track how you use our application to help us improve its features and performance.
          </p>
        </div>
      </main>
    </div>
  );
}
