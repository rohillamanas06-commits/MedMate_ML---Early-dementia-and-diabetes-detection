// API client for MedMate. Talks to a configurable backend base URL,
// or falls back to the built-in Demo Engine (works without a live backend).
import axios from "axios";
import { demoEngine } from "@/lib/demoEngine";

const CONFIG_KEY = "medmate_api_config";

// Default backend URL comes from VITE_API_URL (set in .env / .env.production).
// If it's empty, relative paths are used (handy with the Vite dev proxy or
// when the frontend is served from the same origin as the Flask API).
const ENV_API_URL = import.meta.env.VITE_API_URL || "";

export function getConfig() {
  try {
    const c = JSON.parse(localStorage.getItem(CONFIG_KEY));
    if (c && typeof c === "object") {
      return {
        baseUrl: c.baseUrl ?? ENV_API_URL,
        demoMode: false,
      };
    }
  } catch { /* noop */ }
  return { baseUrl: ENV_API_URL, demoMode: false };
}

export function setConfig(cfg) {
  const current = getConfig();
  localStorage.setItem(CONFIG_KEY, JSON.stringify({ ...current, ...cfg }));
}

function authHeader() {
  const t = localStorage.getItem("medmate_token");
  return t ? { Authorization: `Bearer ${t}` } : {};
}

function normalizeError(e) {
  if (e && e.error) return e; // demo-style error
  const resp = e?.response;
  if (resp) return { status: resp.status, error: resp.data?.error || resp.data?.message || `Request failed (${resp.status})` };
  return { status: 0, error: e?.message || "Network error — check the API URL in Settings." };
}

async function live(method, path, { body, params } = {}) {
  const { baseUrl } = getConfig();
  const url = baseUrl ? `${baseUrl.replace(/\/$/, "")}${path}` : path;
  try {
    const res = await axios({ method, url, data: body, params, headers: { "Content-Type": "application/json", ...authHeader() } });
    return res.data;
  } catch (e) {
    throw normalizeError(e);
  }
}

export const api = {
  isDemo: () => getConfig().demoMode,

  async signup(payload) {
    return getConfig().demoMode ? demoEngine.signup(payload) : live("post", "/auth/signup", { body: payload });
  },
  async login(payload) {
    return getConfig().demoMode ? demoEngine.login(payload) : live("post", "/auth/login", { body: payload });
  },
  async me() {
    return getConfig().demoMode ? demoEngine.me(localStorage.getItem("medmate_token")) : live("get", "/auth/me");
  },
  async predictDiabetes(data) {
    return getConfig().demoMode ? demoEngine.predictDiabetes(localStorage.getItem("medmate_token"), data) : live("post", "/predict/diabetes", { body: data });
  },
  async predictDementia(data) {
    return getConfig().demoMode ? demoEngine.predictDementia(localStorage.getItem("medmate_token"), data) : live("post", "/predict/dementia", { body: data });
  },
  async dashboard(params) {
    return getConfig().demoMode ? demoEngine.dashboard(localStorage.getItem("medmate_token"), params) : live("get", "/dashboard", { params });
  },
  async health() {
    return getConfig().demoMode ? demoEngine.health() : live("get", "/health");
  },
  async modelInfo() {
    return getConfig().demoMode
      ? { diabetes: { dataset: "Demo", target: "Demo" }, dementia: { dataset: "Demo", target: "Demo" } }
      : live("get", "/model/info");
  },
  async getHistoryItem(id) {
    if (getConfig().demoMode) return demoEngine.getHistoryItem(id);
    return live("get", `/dashboard/history/${id}`);
  },
  async deleteHistoryItem(id) {
    if (getConfig().demoMode) return demoEngine.deleteHistoryItem(id);
    return live("delete", `/dashboard/history/${id}`);
  },
  async deleteHistoryAll() {
    if (getConfig().demoMode) return demoEngine.deleteHistoryAll();
    return live("delete", "/dashboard/history/all");
  },
};
