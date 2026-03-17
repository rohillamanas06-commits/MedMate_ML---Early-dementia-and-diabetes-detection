import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Brain, Droplets, ArrowRight } from "lucide-react";

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: [0.2, 0, 0, 1] as const },
};

const Index = () => {
  return (
    <div className="min-h-screen pt-14 sm:pt-16">
      {/* Hero */}
      <section className="relative py-12 sm:py-16 md:py-24 lg:py-32 overflow-hidden">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 text-center">
          <motion.div {...fadeUp}>
            <span className="inline-block px-3 py-1 mb-4 sm:mb-6 text-xs font-medium tracking-widest uppercase bg-muted rounded-full text-muted-foreground">
              ML-Powered Diagnostics
            </span>
          </motion.div>
          <motion.h1
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.1 }}
            className="text-3xl sm:text-4xl md:text-5xl lg:text-7xl font-bold tracking-tighter text-foreground mb-4 sm:mb-6 text-balance leading-tight"
          >
            Early detection,
            <br />
            better outcomes.
          </motion.h1>
          <motion.p
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.2 }}
            className="text-base sm:text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-6 sm:mb-8 lg:mb-10 leading-relaxed px-4"
          >
            MedMate uses machine learning to assess diabetes and dementia risk
            from clinical indicators. Fast, private, and evidence-based.
          </motion.p>
          <motion.div
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 px-4"
          >
            <Link
              to="/diabetes"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 h-11 sm:h-12 px-6 sm:px-8 bg-primary text-primary-foreground rounded-xl font-medium hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-150 text-sm sm:text-base"
            >
              Start Assessment <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/dementia"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 h-11 sm:h-12 px-6 sm:px-8 bg-card text-foreground rounded-xl font-medium shadow-layered hover:bg-muted transition-all duration-150 text-sm sm:text-base"
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