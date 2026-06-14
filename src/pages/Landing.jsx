import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Footer from "@/components/Footer";
import { useAuth } from "@/context/AuthContext";
import { Button } from "@/components/ui/button";
import { LogIn, Sidebar, X, LayoutDashboard, Activity, Brain, LogOut, Sun, Moon, Home } from "lucide-react";

const HERO_URL = "/akram-huseyn-V_0ES17m9Tc-unsplash.jpg";

export default function Landing() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    if (theme === "dark") document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
    localStorage.setItem("theme", theme);
  }, [theme]);


  return (
    <div className="flex min-h-screen w-full flex-col bg-stone-900 font-sans overflow-x-hidden" data-testid="landing-page">
      <div className="relative h-screen w-full shrink-0">
        <img
          src={HERO_URL}
          alt="Abstract medical technology"
          className="absolute inset-0 h-full w-full object-cover object-[center_35%] transition-opacity duration-1000 opacity-100"
        />
        <div className="absolute inset-0 bg-black/30" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/10" />

        <header className="fixed top-0 left-0 right-0 w-full z-50 flex items-center justify-between px-6 py-3 sm:px-12 bg-black/95 backdrop-blur-sm border-b border-white/10 shadow-lg transition-all duration-300">
          <div className="flex items-center gap-2 text-white">
            <span className="font-display text-xl tracking-tight font-semibold">MedMate</span>
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            {isAuthenticated ? (
              <Link to="/dashboard" data-testid="landing-dashboard-btn">
                <Button className="rounded-md px-3 text-white bg-transparent hover:bg-black hover:text-white transition-colors" variant="ghost">
                  Dashboard
                </Button>
              </Link>
            ) : (
              <Link to="/login" data-testid="landing-login-btn">
                <Button className="rounded-md px-3 text-white bg-transparent hover:bg-black hover:text-white transition-colors" variant="ghost">
                  <LogIn className="mr-1.5 h-4 w-4" /> Sign In
                </Button>
              </Link>
            )}
            <Button size="icon" onClick={() => setSidebarOpen(true)} className="rounded-md text-white bg-transparent hover:bg-black hover:text-white transition-colors flex" variant="ghost">
              <Sidebar className="h-5 w-5" />
            </Button>
          </div>
        </header>


      </div>

      {sidebarOpen && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setSidebarOpen(false)} />
          <div className="relative w-64 bg-background h-full shadow-2xl flex flex-col p-6 animate-in slide-in-from-right duration-300 border-l border-border">
            <div className="flex-1 flex flex-col gap-2 text-sm font-medium mt-4">
              <Link to="/" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                <Home className="h-4 w-4" /> Home
              </Link>
              {isAuthenticated ? (
                <>
                  <Link to="/dashboard" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                    <LayoutDashboard className="h-4 w-4" /> Dashboard
                  </Link>
                  <Link to="/diabetes" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                    <Activity className="h-4 w-4" /> Diabetes
                  </Link>
                  <Link to="/dementia" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                    <Brain className="h-4 w-4" /> Dementia
                  </Link>
                </>
              ) : (
                <Link to="/login" onClick={() => setSidebarOpen(false)} className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-secondary hover:text-primary transition-colors">
                  <LogIn className="h-4 w-4" /> Sign In
                </Link>
              )}
              {isAuthenticated && (
                <button onClick={() => { setTheme(t => t === "light" ? "dark" : "light"); setSidebarOpen(false); }} className="flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-colors hover:bg-secondary hover:text-primary">
                  {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                  {theme === "dark" ? "Light Mode" : "Dark Mode"}
                </button>
              )}
              {isAuthenticated && (
                <div className="mt-auto flex flex-col w-full">
                  <div className="mb-2 border-t border-border" />
                  <button onClick={() => { logout(); setSidebarOpen(false); }} className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left text-destructive font-semibold hover:bg-red-600 hover:text-white transition-colors">
                    <LogOut className="h-4 w-4" /> Sign Out
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      <Footer />
    </div>
  );
}
