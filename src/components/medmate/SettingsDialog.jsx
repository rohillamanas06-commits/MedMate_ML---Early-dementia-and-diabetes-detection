import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { getConfig, setConfig, api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { Loader2, PlugZap, CheckCircle2, XCircle } from "lucide-react";

export default function SettingsDialog({ open, onOpenChange }) {
  const initial = getConfig();
  const [baseUrl, setBaseUrl] = useState(initial.baseUrl);
  const [demoMode, setDemoMode] = useState(initial.demoMode);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const { toast } = useToast();

  const handleSave = () => {
    setConfig({ baseUrl: baseUrl.trim(), demoMode });
    toast({ title: "Settings saved", description: demoMode ? "Demo Mode is ON — using built-in simulator." : "Connected to your backend URL." });
    onOpenChange(false);
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    setConfig({ baseUrl: baseUrl.trim(), demoMode });
    try {
      const res = await api.health();
      setTestResult({ ok: true, msg: `Reachable — ${res.version || "ok"}` });
    } catch (e) {
      setTestResult({ ok: false, msg: e.error || "Could not reach backend" });
    } finally {
      setTesting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg rounded-2xl" data-testid="settings-dialog">
        <DialogHeader>
          <DialogTitle className="font-display text-2xl">Backend Connection</DialogTitle>
          <DialogDescription>
            Point MedMate at your Flask API, or enable Demo Mode to explore with the built-in simulator (no backend needed).
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center justify-between rounded-xl border border-border bg-secondary/40 p-4">
          <div>
            <p className="font-medium text-sm">Demo Mode</p>
            <p className="text-xs text-muted-foreground">Simulate predictions & auth locally (no backend needed).</p>
          </div>
          <Switch checked={demoMode} onCheckedChange={setDemoMode} data-testid="settings-demo-toggle" />
        </div>

        <div className="flex flex-col gap-2">
          <Label htmlFor="api-url" className="text-sm">Flask API Base URL</Label>
          <Input
            id="api-url"
            data-testid="settings-api-url"
            placeholder="https://your-medmate-api.com"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            disabled={demoMode}
            className="rounded-xl"
          />
          <p className="text-[11px] text-muted-foreground">
            No trailing slash. Endpoints like <code>/auth/login</code> &amp; <code>/predict/diabetes</code> are appended automatically.
          </p>
        </div>

        {testResult && (
          <div
            data-testid="settings-test-result"
            className={`flex items-center gap-2 rounded-xl px-3 py-2 text-sm ${testResult.ok ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-600"}`}
          >
            {testResult.ok ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
            {testResult.msg}
          </div>
        )}

        <DialogFooter className="flex-col gap-2 sm:flex-row sm:justify-between">
          <Button variant="outline" onClick={handleTest} disabled={testing || demoMode} className="rounded-xl" data-testid="settings-test-btn">
            {testing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <PlugZap className="mr-2 h-4 w-4" />}
            Test Connection
          </Button>
          <Button onClick={handleSave} className="rounded-xl bg-primary hover:bg-primary/90" data-testid="settings-save-btn">
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
