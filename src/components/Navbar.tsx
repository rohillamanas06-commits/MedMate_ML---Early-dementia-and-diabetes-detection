import { Link, useLocation } from "react-router-dom";
import { useTheme } from "./ThemeProvider";
import { motion } from "framer-motion";
import { Sun, Moon, Activity } from "lucide-react";

const navItems = [
  { label: "Home", path: "/" },
  { label: "Diabetes", path: "/diabetes" },
  { label: "Dementia", path: "/dementia" },
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

        <div className="flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className="relative px-4 py-2 text-sm font-medium transition-colors"
              >
                {isActive && (
                  <motion.div
                    layoutId="nav-pill"
                    className="absolute inset-0 bg-secondary rounded-lg"
                    transition={{ type: "spring", duration: 0.4, bounce: 0.1 }}
                  />
                )}
                <span className={`relative z-10 ${isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                  {item.label}
                </span>
              </Link>
            );
          })}
        </div>

        <button
          onClick={toggle}
          className="w-9 h-9 rounded-lg bg-secondary flex items-center justify-center hover:bg-accent transition-colors"
          aria-label="Toggle theme"
        >
          <motion.div
            key={theme}
            initial={{ rotate: -90, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            transition={{ duration: 0.3, ease: [0.2, 0, 0, 1] }}
          >
            {theme === "light" ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
          </motion.div>
        </button>
      </div>
      <div className="h-px bg-border" />
    </nav>
  );
};
