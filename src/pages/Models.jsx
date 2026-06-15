import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Activity, Brain, ChevronDown, ChevronUp, Database, BarChart3, Layers, Target } from "lucide-react";

/* ────────── data ────────── */

const DIABETES_MODELS = [
  { name: "Random Forest",       acc: "78.25%", f1: "0.5575", minF1: "0.2420", auc: "0.6307", prauc: "0.1825", mcc: "0.1266" },
  { name: "Extra Trees",         acc: "72.59%", f1: "0.5430", minF1: "0.2538", auc: "0.6260", prauc: "0.1782", mcc: "0.1317" },
  { name: "Logistic Regression", acc: "76.90%", f1: "0.5569", minF1: "0.2504", auc: "0.6393", prauc: "0.1956", mcc: "0.1330", best: true },
  { name: "LightGBM",            acc: "77.38%", f1: "0.5579", minF1: "0.2489", auc: "0.6309", prauc: "0.1794", mcc: "0.1323" },
  { name: "Voting Ensemble",     acc: "73.96%", f1: "0.5493", minF1: "0.2565", auc: "0.6363", prauc: "0.1964", mcc: "0.1362" },
];

const DIABETES_FEATURES = [
  { name: "prior_inpatient_cnt", imp: 0.1163 },
  { name: "number_inpatient",    imp: 0.1092 },
  { name: "prior_inpatient",     imp: 0.1092 },
  { name: "insulin_changed",     imp: 0.1028 },
  { name: "change_bin",          imp: 0.0822 },
  { name: "total_visits",        imp: 0.0808 },
  { name: "diabetesMed_bin",     imp: 0.0785 },
  { name: "gender_bin",          imp: 0.0770 },
  { name: "diag1_circ",          imp: 0.0622 },
  { name: "insulin_ord",         imp: 0.0471 },
];

const DEMENTIA_MODELS = [
  { name: "Random Forest",       acc: "93.84%", f1: "0.9239", minF1: "0.8930", auc: "0.9848", prauc: "0.9639", mcc: "0.8559" },
  { name: "Extra Trees",         acc: "93.49%", f1: "0.9194", minF1: "0.8872", auc: "0.9840", prauc: "0.9615", mcc: "0.8462" },
  { name: "Logistic Regression", acc: "94.99%", f1: "0.9379", minF1: "0.9106", auc: "0.9863", prauc: "0.9677", mcc: "0.8759", best: true },
  { name: "LightGBM",            acc: "94.57%", f1: "0.9325", minF1: "0.9028", auc: "0.9860", prauc: "0.9670", mcc: "0.8660" },
  { name: "Voting Ensemble",     acc: "94.76%", f1: "0.9349", minF1: "0.9060", auc: "0.9862", prauc: "0.9676", mcc: "0.8705" },
];

const DEMENTIA_FEATURES = [
  { name: "CDRSUM",         imp: 6.4063 },
  { name: "severe_CDR",     imp: 4.9576 },
  { name: "CDRGLOB",        imp: 3.7117 },
  { name: "CDR_x_age",      imp: 3.6808 },
  { name: "CDR_nonzero",    imp: 1.4449 },
  { name: "CDR_sum_x_mmse", imp: 1.0959 },
];

/* ────────── helpers ────────── */

function MetricPill({ label, value, accent }) {
  return (
    <div className="flex flex-col items-center rounded-xl bg-secondary/60 px-4 py-3">
      <span className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</span>
      <span className={`mt-1 font-display text-xl font-semibold tabular-nums ${accent ? "text-primary" : ""}`}>{value}</span>
    </div>
  );
}

function BarRow({ label, value, maxValue }) {
  const pct = Math.min((value / maxValue) * 100, 100);
  return (
    <div className="flex items-center gap-3 py-1.5">
      <span className="w-40 shrink-0 truncate text-xs text-muted-foreground font-medium">{label}</span>
      <div className="relative h-3 flex-1 rounded-full bg-secondary overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-primary/70 mm-bar-grow"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-14 shrink-0 text-right text-xs font-semibold tabular-nums">{value.toFixed(4)}</span>
    </div>
  );
}

function Collapsible({ title, icon: Icon, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-2xl border border-border bg-card overflow-hidden">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between gap-3 px-5 py-4 text-left transition-colors"
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon className="h-4 w-4 text-primary" />}
          <span className="font-display text-base font-medium">{title}</span>
        </div>
        {open ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
      </button>
      {open && <div className="px-5 pb-5 pt-1">{children}</div>}
    </div>
  );
}

/* ────────── main section ────────── */

function ModelSection({ title, icon: Icon, dataset, records, target, bestModel, models, features, maxFeatureValue }) {
  return (
    <section className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-secondary">
          <Icon className="h-5 w-5 text-primary" />
        </span>
        <div>
          <h2 className="font-display text-xl font-medium tracking-tight">{title}</h2>
          <p className="text-xs text-muted-foreground">Best model: {bestModel}</p>
        </div>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricPill label="Dataset" value={records} />
        <MetricPill label="Best AUC" value={models.find(m => m.best)?.auc ?? "—"} accent />
        <MetricPill label="Accuracy" value={models.find(m => m.best)?.acc ?? "—"} />
        <MetricPill label="Macro F1" value={models.find(m => m.best)?.f1 ?? "—"} />
      </div>

      {/* Dataset info */}
      <Collapsible title="Dataset Information" icon={Database}>
        <div className="space-y-2 text-sm text-muted-foreground">
          <p><span className="font-medium text-foreground">Source:</span> {dataset}</p>
          <p><span className="font-medium text-foreground">Records:</span> {records}</p>
          <p><span className="font-medium text-foreground">Target:</span> {target}</p>
        </div>
      </Collapsible>

      {/* Model comparison */}
      <Collapsible title="Model Comparison" icon={Layers} defaultOpen>
        {/* Desktop table — hidden on mobile */}
        <div className="hidden sm:block overflow-x-auto -mx-1">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted-foreground">
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 px-2 font-medium">Accuracy</th>
                <th className="py-2 px-2 font-medium">AUC</th>
                <th className="py-2 px-2 font-medium">Macro F1</th>
                <th className="py-2 px-2 font-medium">PR-AUC</th>
                <th className="py-2 px-2 font-medium">MCC</th>
              </tr>
            </thead>
            <tbody>
              {models.map(m => (
                <tr key={m.name} className="border-b border-border/50 last:border-0">
                  <td className="py-2.5 pr-4 font-medium whitespace-nowrap">
                    {m.name}
                    {m.best && <span className="ml-2 inline-block rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">BEST</span>}
                  </td>
                  <td className="py-2.5 px-2 tabular-nums">{m.acc}</td>
                  <td className="py-2.5 px-2 tabular-nums font-semibold">{m.auc}</td>
                  <td className="py-2.5 px-2 tabular-nums">{m.f1}</td>
                  <td className="py-2.5 px-2 tabular-nums">{m.prauc}</td>
                  <td className="py-2.5 px-2 tabular-nums">{m.mcc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Mobile cards — visible only on mobile */}
        <div className="sm:hidden space-y-3">
          {models.map(m => (
            <div key={m.name} className="rounded-xl border border-border/60 bg-secondary/30 p-4 space-y-2">
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm">{m.name}</span>
                {m.best && <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">BEST</span>}
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Accuracy</p>
                  <p className="text-sm font-semibold tabular-nums">{m.acc}</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">AUC</p>
                  <p className="text-sm font-semibold tabular-nums">{m.auc}</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Macro F1</p>
                  <p className="text-sm font-semibold tabular-nums">{m.f1}</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">PR-AUC</p>
                  <p className="text-sm font-semibold tabular-nums">{m.prauc}</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">MCC</p>
                  <p className="text-sm font-semibold tabular-nums">{m.mcc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Collapsible>

      {/* Feature importance */}
      <Collapsible title="Top Feature Importances" icon={BarChart3}>
        <div className="space-y-0.5">
          {features.map(f => (
            <BarRow key={f.name} label={f.name} value={f.imp} maxValue={maxFeatureValue} />
          ))}
        </div>
      </Collapsible>
    </section>
  );
}

/* ────────── page ────────── */

export default function Models() {
  const [tab, setTab] = useState("diabetes");

  return (
    <div className="w-full px-6 py-8" data-testid="models-page">
      <div className="mm-fade-up">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">Model Performance</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Training results and evaluation metrics for the ML models powering MedMate.
        </p>
      </div>

      {/* Tab switcher */}
      <div className="mt-6 flex gap-2">
        <button
          onClick={() => setTab("diabetes")}
          className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "diabetes" ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
          }`}
        >
          <Activity className="h-4 w-4" /> Diabetes
        </button>
        <button
          onClick={() => setTab("dementia")}
          className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "dementia" ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
          }`}
        >
          <Brain className="h-4 w-4" /> Dementia
        </button>
      </div>

      {/* Content */}
      <div className="mt-6">
        {tab === "diabetes" ? (
          <ModelSection
            title="Diabetes Readmission"
            icon={Activity}
            dataset="UCI / Kaggle — 130-Hospital Clinical Care Dataset"
            records="101,766"
            target="Readmitted within 30 days (binary: Yes / No)"
            bestModel="Logistic Regression"
            models={DIABETES_MODELS}
            features={DIABETES_FEATURES}
            maxFeatureValue={0.12}
          />
        ) : (
          <ModelSection
            title="Dementia Detection"
            icon={Brain}
            dataset="National Alzheimer's Coordinating Center (NACC) UDS — investigator_nacc73.csv"
            records="214,976"
            target="Demented (binary: 0 = Nondemented, 1 = Demented)"
            bestModel="Logistic Regression"
            models={DEMENTIA_MODELS}
            features={DEMENTIA_FEATURES}
            maxFeatureValue={6.5}
          />
        )}
      </div>
    </div>
  );
}
