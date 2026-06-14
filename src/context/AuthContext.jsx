import { createContext, useContext, useEffect, useState, useCallback } from "react";
import { api, setConfig } from "@/lib/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem("medmate_user")) || null; } catch { return null; }
  });
  const [token, setToken] = useState(() => localStorage.getItem("medmate_token") || null);
  const [loading, setLoading] = useState(false);

  const persist = useCallback((tok, usr) => {
    if (tok) localStorage.setItem("medmate_token", tok); else localStorage.removeItem("medmate_token");
    if (usr) localStorage.setItem("medmate_user", JSON.stringify(usr)); else localStorage.removeItem("medmate_user");
    setToken(tok || null);
    setUser(usr || null);
  }, []);

  const login = useCallback(async (email, password) => {
    setLoading(true);
    try {
      setConfig({ demoMode: false });
      const res = await api.login({ email, password });
      persist(res.token, res.user);
      return res;
    } finally { setLoading(false); }
  }, [persist]);

  const signup = useCallback(async (email, password, full_name) => {
    setLoading(true);
    try {
      setConfig({ demoMode: false });
      const res = await api.signup({ email, password, full_name });
      persist(res.token, res.user);
      return res;
    } finally { setLoading(false); }
  }, [persist]);

  const logout = useCallback(() => {
    setConfig({ demoMode: false });
    persist(null, null);
  }, [persist]);

  const loginAsGuest = useCallback(() => {
    setConfig({ demoMode: true });
    persist("demo_token", { full_name: "Demo User", email: "demo@medmate.ai", id: "demo" });
  }, [persist]);

  // On mount: restore demoMode if the stored token is a guest token,
  // then refresh the user profile from the API if needed.
  useEffect(() => {
    const storedToken = localStorage.getItem("medmate_token");
    if (storedToken === "demo_token") {
      setConfig({ demoMode: true });
    }
    if (storedToken && !user) {
      api.me().then(setUser).catch(() => persist(null, null));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, loading, login, signup, logout, loginAsGuest, isAuthenticated: !!token }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
