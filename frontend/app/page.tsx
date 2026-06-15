"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export default function Home() {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    fetch("/api/stats").then(r => r.json()).then(setStats).catch(() => {});
  }, []);

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-6 relative overflow-hidden">
      {/* Animated background scroll painting images */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 grid grid-cols-3 opacity-[0.55]">
          <img src="/hero.jpg" alt="" className="w-full h-full object-cover" />
          <img src="/hero2.jpg" alt="" className="w-full h-full object-cover" />
          <img src="/hero3.jpg" alt="" className="w-full h-full object-cover" />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0a] via-[#0a0a0a]/50 to-[#0a0a0a]" />
        <div className="absolute inset-0 bg-gradient-to-r from-[#0a0a0a] via-transparent to-[#0a0a0a]" />
      </div>

      {/* Floating accent orbs - also animated */}
      <motion.div
        className="absolute top-1/4 right-1/4 w-96 h-96 bg-[#d4a574]/10 rounded-full blur-3xl"
        animate={{ x: [0, 30, -20, 0], y: [0, -20, 30, 0] }}
        transition={{ duration: 20, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" }}
      />
      <motion.div
        className="absolute bottom-1/4 left-1/4 w-64 h-64 bg-[#d4a574]/5 rounded-full blur-3xl"
        animate={{ x: [0, -25, 15, 0], y: [0, 15, -25, 0] }}
        transition={{ duration: 18, repeat: Infinity, repeatType: "reverse", ease: "easeInOut", delay: 3 }}
      />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-10 text-center max-w-3xl"
      >
        <motion.div
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h1 className="text-4xl md:text-7xl font-bold tracking-tight mb-2">
            <span className="text-[#d4a574]">GeMi</span>
          </h1>
          <p className="text-sm md:text-2xl text-[#dcdcdc] tracking-widest uppercase font-medium mb-6 md:mb-8">
            Graph-based Multimodal Recommendation
          </p>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="text-lg md:text-2xl text-[#e4e4e4] leading-relaxed mb-3 md:mb-4 px-2"
        >
          Discover narrative scroll paintings through personalized recommendation.
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7, duration: 0.8 }}
          className="text-base md:text-xl text-[#cccccc] leading-relaxed mb-8 md:mb-12 max-w-xl mx-auto px-2"
        >
          An endangered performing art from Eastern India, conserved through multimodal AI
          that combines vision-language models with graph neural networks.
        </motion.p>

        {stats && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9, duration: 0.6 }}
            className="flex justify-center gap-6 md:gap-12 mb-8 md:mb-12"
          >
            {[
              { label: "Panels", value: stats.total_panels },
              { label: "Scrolls", value: stats.scrolls },
              { label: "AI Models", value: stats.models_available },
              { label: "Concepts", value: 3 },
            ].map((s) => (
              <div key={s.label} className="text-center">
                <div className="text-3xl md:text-5xl font-bold text-[#d4a574]">{s.value}</div>
                <div className="text-base md:text-lg text-[#c2c2c2] mt-1">{s.label}</div>
              </div>
            ))}
          </motion.div>
        )}

        <motion.a
          href="/explore"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.1, duration: 0.6 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
          className="inline-block px-8 md:px-10 py-3 md:py-4 bg-[#d4a574] text-[#0a0a0a] font-semibold text-base md:text-lg rounded-full hover:bg-[#e8c49a] transition-colors cursor-pointer"
        >
          Begin Exploration
        </motion.a>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.4, duration: 0.6 }}
          className="mt-8 flex flex-col items-center gap-1"
        >
          <p className="text-sm text-[#9a9a9a]">
            Workshop on AI & Analytics for Social Good
          </p>
          <p className="text-xs text-[#888888]">
            University of Maryland, April 24, 2026
          </p>
        </motion.div>
      </motion.div>
    </main>
  );
}
