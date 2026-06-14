import { Link } from "react-router-dom";
import { Github, Linkedin, Mail } from "lucide-react";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-black text-white/70 py-12 md:py-16 border-t border-white/10 w-full" data-testid="footer">
      <div className="mx-auto w-full max-w-[1600px] px-8 lg:px-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 md:gap-8">
          
          <div className="flex flex-col gap-4 col-span-1 md:col-span-1">
            <div className="flex items-center gap-2">
              <img src="/favicon.svg" alt="MedMate" className="h-6 w-6" />
              <span className="font-display font-semibold text-xl text-white">MedMate</span>
            </div>
            <p className="text-sm text-white/50">
              Your intelligent companion for ML-powered clinical assessments. Experience the future of medical predictions.
            </p>
            <div className="flex gap-4 mt-2">
              <a href="mailto:rohillamanas06@gmail.com" target="_blank" rel="noopener noreferrer" className="bg-white/5 hover:bg-white/10 p-2 rounded-md transition-colors text-white">
                <Mail className="h-4 w-4" />
              </a>
              <a href="https://github.com/rohillamanas06-commits/MedMate_ML---Early-dementia-and-diabetes-detection" target="_blank" rel="noopener noreferrer" className="bg-white/5 hover:bg-white/10 p-2 rounded-md transition-colors text-white">
                <Github className="h-4 w-4" />
              </a>
              <a href="https://www.linkedin.com/in/manas-rohilla/" target="_blank" rel="noopener noreferrer" className="bg-white/5 hover:bg-white/10 p-2 rounded-md transition-colors text-white">
                <Linkedin className="h-4 w-4" />
              </a>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <h3 className="text-white font-semibold text-sm">Features</h3>
            <Link to="/dashboard" className="text-sm hover:text-white transition-colors w-fit">Dashboard</Link>
            <Link to="/diabetes" className="text-sm hover:text-white transition-colors w-fit">Diabetes Assessment</Link>
            <Link to="/dementia" className="text-sm hover:text-white transition-colors w-fit">Dementia Assessment</Link>
          </div>

          <div className="flex flex-col gap-4">
            <h3 className="text-white font-semibold text-sm">Company</h3>
            <Link to="/about" className="text-sm hover:text-white transition-colors w-fit">About</Link>
            <Link to="/faq" className="text-sm hover:text-white transition-colors w-fit">FAQ</Link>
          </div>

          <div className="flex flex-col gap-4">
            <h3 className="text-white font-semibold text-sm">Legal</h3>
            <Link to="/privacy" className="text-sm hover:text-white transition-colors w-fit">Privacy Policy</Link>
            <Link to="/terms" className="text-sm hover:text-white transition-colors w-fit">Terms & Conditions</Link>
            <Link to="/cookies" className="text-sm hover:text-white transition-colors w-fit">Cookies Policy</Link>
          </div>

        </div>

        <div className="mt-16 pt-8 border-t border-white/10 flex items-center justify-between">
          <p className="text-xs text-white/40">
            © {currentYear} MedMate. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
