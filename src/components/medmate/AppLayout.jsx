import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate, Outlet } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { LogOut, LayoutDashboard, Activity, Brain, Sidebar, X, Home, Moon, Sun } from "lucide-react";
import { api } from "@/lib/api";

export default function AppLayout({ children }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  useEffect(() => {
    if (theme === "dark") document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  const handleLogout = () => { logout(); navigate("/"); };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur-md">
        <div className="flex w-full items-center justify-between px-6 py-4 sm:px-12">
          <div className="flex items-center gap-6">
            <Link to="/dashboard" className="flex items-center gap-2">
              {api.isDemo() && <Badge variant="secondary" className="ml-1 hidden rounded-full text-[10px] uppercase tracking-wider sm:inline-flex">Demo</Badge>}
            </Link>
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            <span className="hidden text-sm text-muted-foreground md:inline font-medium">
              {user?.full_name || user?.email}
            </span>
            <Button variant="ghost" size="icon" className="rounded-xl hover:bg-secondary hover:text-primary" onClick={() => setSidebarOpen(true)}>
              <Sidebar className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full">
        {children || <Outlet />}
      </main>

      {sidebarOpen && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setSidebarOpen(false)} />
          <div className="relative w-64 bg-background h-full shadow-2xl flex flex-col p-6 animate-in slide-in-from-right duration-300 border-l border-border">
            <div className="flex-1 flex flex-col gap-2 text-sm font-medium mt-4">
              <Link to="/" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                <Home className="h-4 w-4" /> Home
              </Link>
              <Link to="/dashboard" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                <LayoutDashboard className="h-4 w-4" /> Dashboard
              </Link>
              <Link to="/diabetes" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                <Activity className="h-4 w-4" /> Diabetes
              </Link>
              <Link to="/dementia" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                <Brain className="h-4 w-4" /> Dementia
              </Link>
              <button onClick={() => { setTheme(t => t === "light" ? "dark" : "light"); setSidebarOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-colors hover:bg-secondary hover:text-primary">
                {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                {theme === "dark" ? "Light Mode" : "Dark Mode"}
              </button>

              <div className="mt-auto flex flex-col w-full">
                <div className="mb-2 border-t border-border" />
                <button onClick={() => { handleLogout(); setSidebarOpen(false); }} className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left text-destructive font-semibold hover:bg-red-600 hover:text-white transition-colors">
                  <LogOut className="h-4 w-4" /> Sign Out
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
