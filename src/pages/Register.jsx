import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";

const IMAGES = [
  "/national-cancer-institute-82BHTkmkDfU-unsplash.jpg",
  "/jc-gellidon-uhXlRnt9dTw-unsplash.jpg"
];

export default function Register() {
  const navigate = useNavigate();
  const { signup, loginAsGuest, loading } = useAuth();
  const { toast } = useToast();

  const [currentImageIdx, setCurrentImageIdx] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentImageIdx((prev) => (prev + 1) % IMAGES.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const [signupForm, setSignupForm] = useState({ full_name: "", email: "", password: "", confirm: "" });

  const handleSignup = async (e) => {
    e.preventDefault();
    if (signupForm.password !== signupForm.confirm) {
      return toast({ title: "Error", description: "Passwords do not match", variant: "destructive" });
    }
    try {
      await signup(signupForm.email, signupForm.password, signupForm.full_name);
      toast({ title: "Account created", description: "Welcome to MedMate." });
      navigate("/dashboard");
    } catch (err) {
      toast({ title: "Signup failed", description: err.error || "Could not create account", variant: "destructive" });
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

      <div className="flex w-full flex-col items-center px-8 pb-12 pt-[6vh] lg:pt-[8vh] sm:px-12 lg:w-[40%] xl:px-16">
        <div className="w-full max-w-[380px]">
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 mt-[2vh] lg:mt-[4vh]">
            <div className="mb-6">
              <p className="text-xs font-semibold uppercase tracking-widest text-white/50 mb-2">Register</p>
              <h1 className="text-[2.5rem] text-white font-serif tracking-tight leading-none">New account</h1>
            </div>

            <form onSubmit={handleSignup} className="flex flex-col gap-4">
              <div className="relative">
                <label htmlFor="signup-name" className={labelClass}>Name</label>
                <input id="signup-name" type="text" required value={signupForm.full_name}
                  onChange={(e) => setSignupForm((f) => ({ ...f, full_name: e.target.value }))}
                  className={inputClass} />
              </div>

              <div className="relative">
                <label htmlFor="signup-email" className={labelClass}>Email</label>
                <input id="signup-email" type="email" required value={signupForm.email}
                  onChange={(e) => setSignupForm((f) => ({ ...f, email: e.target.value }))}
                  className={inputClass} />
              </div>

              <div className="relative">
                <label htmlFor="signup-password" className={labelClass}>Password (Min 6)</label>
                <input id="signup-password" type="password" required minLength={6} value={signupForm.password}
                  onChange={(e) => setSignupForm((f) => ({ ...f, password: e.target.value }))}
                  className={inputClass} />
              </div>

              <div className="relative">
                <label htmlFor="signup-confirm" className={labelClass}>Confirm Password</label>
                <input id="signup-confirm" type="password" required minLength={6} value={signupForm.confirm}
                  onChange={(e) => setSignupForm((f) => ({ ...f, confirm: e.target.value }))}
                  className={inputClass} />
              </div>

              <button type="submit" disabled={loading} className="mt-6 w-full bg-white text-black py-4 text-xs font-bold tracking-widest uppercase hover:bg-red-600 hover:text-white transition-colors flex justify-center items-center">
                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null} Create Account
              </button>

              <div className="mt-4 flex flex-col gap-3">
                <p className="text-[13px] text-white/60">
                  Have an account? <Link to="/login" className="text-white hover:underline font-medium">Sign in</Link>
                </p>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
