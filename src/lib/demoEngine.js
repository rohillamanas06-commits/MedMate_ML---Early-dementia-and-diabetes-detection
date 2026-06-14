// Local Demo Engine — simulates the MedMate Flask backend entirely in the browser
// so the full app is usable & testable without a live API. Heuristic predictions
// are clearly an APPROXIMATION of the real trained scikit-learn ensemble.

const USERS_KEY = "medmate_demo_users";
const LOGS_KEY = "medmate_demo_logs";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function read(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function write(key, val) { localStorage.setItem(key, JSON.stringify(val)); }

function token(email) { return `demo.${btoa(email)}.medmate`; }
function emailFromToken(t) {
  try { return atob((t || "").split(".")[1] || ""); } catch { return ""; }
}

function ensureSeed() {
  const users = read(USERS_KEY, null);
  if (users) return;
  const now = new Date().toISOString();
  const demoUser = {
    id: 1, email: "demo@medmate.ai", password: "demo123",
    full_name: "Dr. Demo Mate", created_at: now, last_login: now,
  };
  write(USERS_KEY, [demoUser]);
  // Seed a few example predictions for the demo account
  const seeds = [
    { model_type: "diabetes", prediction: "Readmitted <30d", confidence: 78.4, risk_level: "High", probabilities: { "Not Readmitted": 21.6, "Readmitted <30d": 78.4 } },
    { model_type: "dementia", prediction: "Nondemented", confidence: 91.2, risk_level: "Low", probabilities: { Nondemented: 91.2, Demented: 8.8 } },
    { model_type: "diabetes", prediction: "Not Readmitted", confidence: 64.0, risk_level: "Low", probabilities: { "Not Readmitted": 64.0, "Readmitted <30d": 36.0 } },
    { model_type: "dementia", prediction: "Demented", confidence: 82.7, risk_level: "High", probabilities: { Nondemented: 17.3, Demented: 82.7 } },
  ];
  const logs = seeds.map((s, i) => ({
    id: i + 1, user_email: "demo@medmate.ai", inputs: {}, ...s,
    created_at: new Date(Date.now() - (i + 1) * 36e5 * 7).toISOString(),
  }));
  write(LOGS_KEY, logs);
}

function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }

const AGE_MID = { "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95 };

function diabetesScore(d) {
  let s = 0;
  s += clamp(Number(d.number_inpatient) || 0, 0, 10) * 0.9;
  s += clamp(Number(d.number_emergency) || 0, 0, 10) * 0.6;
  s += clamp(Number(d.number_outpatient) || 0, 0, 10) * 0.2;
  s += clamp(Number(d.number_diagnoses) || 0, 0, 16) * 0.25;
  s += clamp(Number(d.num_medications) || 0, 0, 81) * 0.06;
  s += clamp(Number(d.time_in_hospital) || 0, 0, 14) * 0.18;
  s += (AGE_MID[d.age] || 45) * 0.012;
  if (d.A1Cresult === ">8") s += 1.2; else if (d.A1Cresult === ">7") s += 0.7;
  if (d.max_glu_serum === ">300") s += 1.0; else if (d.max_glu_serum === ">200") s += 0.6;
  if (d.insulin === "Up" || d.insulin === "Down") s += 0.6;
  if (d.change === "Ch") s += 0.4;
  if (d.diabetesMed === "Yes") s += 0.3;
  if (String(d.diag_1).split(".")[0] === "250") s += 0.5;
  const p = 1 / (1 + Math.exp(-(s - 4.2)));
  return p; // probability of readmission
}

function dementiaScore(d) {
  let s = 0;
  const cdr = Number(d.CDRGLOB) || 0;
  const mmse = clamp(Number(d.NACCMMSE) ?? 28, 0, 30);
  s += cdr * 2.6;
  s += (28 - mmse) * 0.34;
  s += clamp((Number(d.CDRSUM) || 0), 0, 18) * 0.25;
  if ((Number(d.NACCAGE) || 0) > 75) s += 0.8;
  s += clamp((Number(d.TRAILB) || 0) / 100, 0, 5) * 0.5;
  s -= clamp((Number(d.ANIMALS) || 0) - 15, -15, 30) * 0.05;
  if (Number(d.APOE4) === 1) s += 0.9;
  if (Number(d.NACCDEP) === 1) s += 0.3;
  if (Number(d.DIABETES) === 1) s += 0.2;
  if (Number(d.HYPERTEN) === 1) s += 0.2;
  const p = 1 / (1 + Math.exp(-(s - 2.6)));
  return p; // probability of dementia
}

function pushLog(email, rec) {
  const logs = read(LOGS_KEY, []);
  const id = (logs.reduce((m, l) => Math.max(m, l.id), 0) || 0) + 1;
  logs.unshift({ id, user_email: email, created_at: new Date().toISOString(), ...rec });
  write(LOGS_KEY, logs);
}

export const demoEngine = {
  async signup({ email, password, full_name }) {
    ensureSeed();
    await sleep(450);
    email = (email || "").trim().toLowerCase();
    if (!email || !password) throw { status: 400, error: "email and password are required" };
    if (password.length < 6) throw { status: 400, error: "Password must be at least 6 characters" };
    const users = read(USERS_KEY, []);
    if (users.find((u) => u.email === email)) throw { status: 409, error: "Email already registered" };
    const now = new Date().toISOString();
    const user = { id: (users.reduce((m, u) => Math.max(m, u.id), 0) || 0) + 1, email, password, full_name: (full_name || "").trim(), created_at: now, last_login: now };
    users.push(user);
    write(USERS_KEY, users);
    return { token: token(email), user: { id: user.id, email, full_name: user.full_name, created_at: now } };
  },

  async login({ email, password }) {
    ensureSeed();
    await sleep(450);
    email = (email || "").trim().toLowerCase();
    const users = read(USERS_KEY, []);
    const u = users.find((x) => x.email === email);
    if (!u || u.password !== password) throw { status: 401, error: "Invalid email or password" };
    u.last_login = new Date().toISOString();
    write(USERS_KEY, users);
    return { token: token(email), user: { id: u.id, email: u.email, full_name: u.full_name, last_login: u.last_login } };
  },

  async me(t) {
    await sleep(150);
    const email = emailFromToken(t);
    const u = read(USERS_KEY, []).find((x) => x.email === email);
    if (!u) throw { status: 404, error: "User not found" };
    return { id: u.id, email: u.email, full_name: u.full_name, created_at: u.created_at, last_login: u.last_login };
  },

  async predictDiabetes(t, data) {
    await sleep(700);
    const p = diabetesScore(data);
    const pred = p >= 0.5 ? 1 : 0;
    const label = pred === 1 ? "Readmitted <30d" : "Not Readmitted";
    const confidence = Math.round((pred === 1 ? p : 1 - p) * 1000) / 10;
    const risk_level = pred === 1 ? (confidence >= 70 ? "High" : "Medium") : "Low";
    const probabilities = { "Not Readmitted": Math.round((1 - p) * 1000) / 10, "Readmitted <30d": Math.round(p * 1000) / 10 };
    const email = emailFromToken(t);
    if (email) pushLog(email, { model_type: "diabetes", inputs: data, prediction: label, confidence, risk_level, probabilities });
    return { prediction: label, confidence, risk_level, probabilities };
  },

  async predictDementia(t, data) {
    await sleep(700);
    const mmse = Number(data.NACCMMSE);
    if (!(mmse >= 0 && mmse <= 30)) throw { status: 422, error: `NACCMMSE must be 0–30. Got ${data.NACCMMSE}.` };
    const p = dementiaScore(data);
    const pred = p >= 0.5 ? 1 : 0;
    const label = pred === 1 ? "Demented" : "Nondemented";
    const confidence = Math.round((pred === 1 ? p : 1 - p) * 1000) / 10;
    let risk_level;
    if (pred === 1 && confidence >= 75) risk_level = "High";
    else if (pred === 1) risk_level = "Medium";
    else if (confidence >= 75) risk_level = "Low";
    else risk_level = "Medium";
    const probabilities = { Nondemented: Math.round((1 - p) * 1000) / 10, Demented: Math.round(p * 1000) / 10 };
    const email = emailFromToken(t);
    if (email) pushLog(email, { model_type: "dementia", inputs: data, prediction: label, confidence, risk_level, probabilities });
    return { prediction: label, confidence, risk_level, probabilities };
  },

  async dashboard(t, { limit = 20, offset = 0, model = null } = {}) {
    await sleep(300);
    const email = emailFromToken(t);
    const u = read(USERS_KEY, []).find((x) => x.email === email);
    if (!u) throw { status: 404, error: "User not found" };
    let logs = read(LOGS_KEY, []).filter((l) => l.user_email === email);
    if (model === "diabetes" || model === "dementia") logs = logs.filter((l) => l.model_type === model);
    logs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    const all = read(LOGS_KEY, []).filter((l) => l.user_email === email);
    const stats = {
      total_predictions: all.length,
      diabetes_predictions: all.filter((l) => l.model_type === "diabetes").length,
      dementia_predictions: all.filter((l) => l.model_type === "dementia").length,
      high_risk_count: all.filter((l) => l.risk_level === "High").length,
      medium_risk_count: all.filter((l) => l.risk_level === "Medium").length,
      low_risk_count: all.filter((l) => l.risk_level === "Low").length,
      avg_confidence: all.length ? Math.round((all.reduce((s, l) => s + l.confidence, 0) / all.length) * 100) / 100 : 0,
    };
    return {
      user: { id: u.id, email: u.email, full_name: u.full_name, created_at: u.created_at, last_login: u.last_login },
      stats,
      history: logs.slice(offset, offset + limit),
      pagination: { limit, offset, total: logs.length },
    };
  },

  async health() { return { status: "ok", models_loaded: ["diabetes", "dementia"], version: "v5 (demo)", auth: "jwt" }; },

  async getHistoryItem(id) {
    await sleep(150);
    const log = read(LOGS_KEY, []).find((l) => l.id === id);
    if (!log) throw { status: 404, error: "Record not found" };
    return log;
  },

  async deleteHistoryItem(id) {
    await sleep(150);
    const logs = read(LOGS_KEY, []);
    const next = logs.filter((l) => l.id !== id);
    write(LOGS_KEY, next);
    return { deleted: logs.length - next.length };
  },

  async deleteHistoryAll() {
    await sleep(150);
    const logs = read(LOGS_KEY, []);
    write(LOGS_KEY, []);
    return { deleted: logs.length };
  },
};
