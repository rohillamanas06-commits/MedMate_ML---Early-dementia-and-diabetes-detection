import { Link, useLocation } from "react-router-dom";
import { useTheme } from "./ThemeProvider";
import { motion } from "framer-motion";
import { Sun, Moon, Activity } from "lucide-react";

const navItems = [
  { label: "Home", path: "/" },
  { label: "Diabetes", path: "/diabetes" },
  { label: "Dementia", path: "/dementia" },
  { label: "About", path: "/about" },
];

export const Navbar = () => {
  const { theme, toggle } = useTheme();
  const location = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-xl">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
            <Activity className="w-4 h-4 text-primary-foreground" />
          </div>
          <span className="text-lg font-semibold tracking-tight">MedMate</span>
        </Link>

        <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`relative px-4 py-2 text-sm font-medium transition-colors ${isActive ? "bg-secondary rounded-lg text-foreground" : "text-muted-foreground hover:text-foreground"}`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>

        <button
          onClick={toggle}
          className="w-9 h-9 rounded-lg bg-secondary flex items-center justify-center hover:bg-accent transition-colors"
          aria-label="Toggle theme"
        >
          {theme === "light" ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
        </button>
      </div>
      <div className="h-px bg-border" />
    </nav>
  );
};
