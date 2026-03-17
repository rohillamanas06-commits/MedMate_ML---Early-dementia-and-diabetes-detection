import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Brain, Droplets, ArrowRight, Shield, Zap, BarChart3 } from "lucide-react";

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: [0.2, 0, 0, 1] as const },
};

const Index = () => {
  return (
    <div className="min-h-screen pt-16">
      {/* Hero */}
      <section className="relative py-24 md:py-32 overflow-hidden">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <motion.div {...fadeUp}>
            <span className="inline-block px-3 py-1 mb-6 text-xs font-medium tracking-widest uppercase bg-muted rounded-full text-muted-foreground">
              ML-Powered Diagnostics
            </span>
          </motion.div>
          <motion.h1
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.1 }}
            className="text-5xl md:text-7xl font-bold tracking-tighter text-foreground mb-6 text-balance"
          >
            Early detection,
            <br />
            better outcomes.
          </motion.h1>
          <motion.p
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.2 }}
            className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            MedMate uses machine learning to assess diabetes and dementia risk
            from clinical indicators. Fast, private, and evidence-based.
          </motion.p>
          <motion.div
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.3 }}
            className="flex items-center justify-center gap-4 flex-wrap"
          >
            <Link
              to="/diabetes"
              className="inline-flex items-center gap-2 h-12 px-8 bg-primary text-primary-foreground rounded-xl font-medium hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-150"
            >
              Start Assessment <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/dementia"
              className="inline-flex items-center gap-2 h-12 px-8 bg-card text-foreground rounded-xl font-medium shadow-layered hover:bg-muted transition-all duration-150"
            >
              Dementia Screening
            </Link>
          </motion.div>
        </div>
      </section>

      
      
     
    </div>
  );
};

export default Index;
