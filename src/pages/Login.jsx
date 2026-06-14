import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";

const IMAGES = [
  "/national-cancer-institute-82BHTkmkDfU-unsplash.jpg",
  "/jc-gellidon-uhXlRnt9dTw-unsplash.jpg"
];

export default function Login() {
  const navigate = useNavigate();
  const { login, loginAsGuest, loading } = useAuth();
  const { toast } = useToast();

  const [currentImageIdx, setCurrentImageIdx] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentImageIdx((prev) => (prev + 1) % IMAGES.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const [loginForm, setLoginForm] = useState({ email: "", password: "" });

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      await login(loginForm.email, loginForm.password);
      toast({ title: "Welcome back", description: "Signed in successfully." });
      navigate("/dashboard");
    } catch (err) {
      toast({ title: "Login failed", description: err.error || "Invalid credentials", variant: "destructive" });
    }
  };

  const inputClass = "w-full bg-transparent border-b border-white/20 pb-2 text-white placeholder:text-transparent focus:outline-none focus:border-white transition-colors mt-2";
  const labelClass = "text-[10px] font-semibold uppercase tracking-widest text-white/50";

  return (
    <div className="flex min-h-screen w-full bg-black font-sans" data-testid="auth-page">
      <div className="relative hidden w-[60%] lg:block bg-black">
        {IMAGES.map((img, idx) => (
          <img
            key={img}
            src={img}
            alt="Medical imagery"
            className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-1000 ${idx === currentImageIdx ? "opacity-100" : "opacity-0"
              }`}
          />
        ))}
        <div className="absolute inset-0 bg-black/20" />
      </div>

      <div className="flex w-full flex-col items-center px-8 pb-12 pt-[14vh] lg:pt-[20vh] sm:px-12 lg:w-[40%] xl:px-16">
        <div className="w-full max-w-[380px]">
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="mb-8">
              <p className="text-xs font-semibold uppercase tracking-widest text-white/50 mb-2">Sign in</p>
              <h1 className="text-[2.5rem] text-white font-serif tracking-tight leading-none">Your account</h1>
            </div>

            <form onSubmit={handleLogin} className="flex flex-col gap-4">
              <div className="relative">
                <label htmlFor="login-email" className={labelClass}>Email Address</label>
                <input id="login-email" type="email" required value={loginForm.email}
                  onChange={(e) => setLoginForm((f) => ({ ...f, email: e.target.value }))}
                  className={inputClass} />
              </div>

              <div className="relative">
                <label htmlFor="login-password" className={labelClass}>Password</label>
                <input id="login-password" type="password" required value={loginForm.password}
                  onChange={(e) => setLoginForm((f) => ({ ...f, password: e.target.value }))}
                  className={inputClass} />
              </div>

              <button type="submit" disabled={loading} className="mt-6 w-full bg-white text-black py-4 text-xs font-bold tracking-widest uppercase hover:bg-red-600 hover:text-white transition-colors flex justify-center items-center">
                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null} Sign in
              </button>

              <div className="mt-4 flex flex-col gap-3">
                <p className="text-[13px] text-white/60">
                  No account? <Link to="/register" className="text-white hover:underline font-medium">Create one</Link>
                </p>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
