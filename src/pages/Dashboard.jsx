import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { api } from "@/lib/api";
import { Card } from "@/components/ui/card";
import {
  Activity, Brain, TrendingUp, ShieldAlert, ShieldCheck,
  ShieldQuestion, Loader2, ArrowRight
} from "lucide-react";

const RISK_DOT = { High: "#E2725B", Medium: "#CC7722", Low: "#71A6D2" };
const RISK_BG  = { High: "#E2725B1A", Medium: "#CC77221A", Low: "#71A6D21A" };

function StatCard({ label, value }) {
  return (
    <Card className="rounded-2xl border-border bg-card p-5">
      <p className="text-xs uppercase tracking-widest text-muted-foreground">{label}</p>
      <p className="mt-2 font-display text-3xl font-medium tabular-nums">{value}</p>
    </Card>
  );
}

function HistoryRow({ item }) {
  const navigate = useNavigate();
  const dot   = RISK_DOT[item.risk_level] || RISK_DOT.Medium;
  const dotBg = RISK_BG[item.risk_level]  || RISK_BG.Medium;

  return (
    <button
      type="button"
      onClick={() => navigate(`/dashboard/history/${item.id}`, { state: { item } })}
      className="w-full flex items-center justify-between gap-4 border-b border-border py-3 last:border-0 text-left hover:bg-secondary/30 px-2 -mx-2 rounded-xl transition-colors group"
      data-testid="history-row"
    >
      {/* Left: icon + label */}
      <div className="flex items-center gap-3 min-w-0">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-secondary">
          {item.model_type === "diabetes"
            ? <Activity className="h-4 w-4 text-primary" />
            : <Brain className="h-4 w-4 text-primary" />}
        </span>
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">{item.prediction}</p>
          <p className="text-xs text-muted-foreground truncate">
            {item.model_type === "diabetes" ? "Diabetes Readmission" : "Dementia"}
            {" · "}
            {new Date(item.created_at).toLocaleString()}
          </p>
        </div>
      </div>

      {/* Right: confidence + badge + arrow */}
      <div className="flex items-center gap-3 shrink-0">
        <span className="text-sm font-semibold tabular-nums">{item.confidence}%</span>
        <span
          className="rounded-full px-2.5 py-1 text-xs font-medium"
          style={{ backgroundColor: dotBg, color: dot }}
        >
          {item.risk_level}
        </span>
        <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-0.5 transition-all" />
      </div>
    </button>
  );
}

export default function Dashboard() {
  const { user, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [data, setData]               = useState(null);
  const [loadingData, setLoadingData] = useState(true);

  useEffect(() => {
    if (!isAuthenticated) navigate("/login", { replace: true });
  }, [isAuthenticated, navigate]);

  const refresh = async () => {
    setLoadingData(true);
    try { const res = await api.dashboard({ limit: 50 }); setData(res); }
    catch { setData(null); }
    finally { setLoadingData(false); }
  };

  useEffect(() => { if (isAuthenticated) refresh(); }, [isAuthenticated]);

  const stats = data?.stats;

  return (
    <div className="w-full px-6 py-8" data-testid="dashboard-page">
      <div className="mm-fade-up">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">
          Welcome{user?.full_name ? `, ${user.full_name}` : ""}
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Here's your clinical prediction activity overview.
        </p>
      </div>

      {loadingData ? (
        <div className="mt-8 flex h-32 items-center justify-center text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : (
        <>
          <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Total Runs"     value={stats?.total_predictions ?? 0} />
            <StatCard label="High Risk"      value={stats?.high_risk_count ?? 0} />
            <StatCard label="Medium Risk"    value={stats?.medium_risk_count ?? 0} />
            <StatCard label="Avg Confidence" value={`${stats?.avg_confidence ?? 0}%`} />
          </div>

          <Card className="mt-8 rounded-2xl border-border bg-card p-6" data-testid="history-card">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-display text-xl font-medium">Recent Activity</h3>
              {data?.history?.length > 0 && (
                <p className="text-xs text-muted-foreground">Click a row to view full details</p>
              )}
            </div>

            {data?.history?.length ? (
              <div className="mt-3">
                {data.history.map((item) => (
                  <HistoryRow key={item.id} item={item} />
                ))}
              </div>
            ) : (
              <p className="mt-3 text-sm text-muted-foreground">
                No assessments yet — navigate to Diabetes or Dementia to run one.
              </p>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
