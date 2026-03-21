"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface ScrollInfo {
  scroll_id: string;
  panel_count: number;
  preview_image: string;
}

interface Panel {
  index: number;
  id: string;
  scroll_id: string;
  panel_id: string;
  text: string;
  animal_label: number;
  myth_label: number;
  tree_label: number;
  image_url: string;
}

export default function StoryPage() {
  const [scrolls, setScrolls] = useState<ScrollInfo[]>([]);
  const [selectedScroll, setSelectedScroll] = useState<string | null>(null);
  const [panels, setPanels] = useState<Panel[]>([]);
  const [activePanel, setActivePanel] = useState(0);
  const [loading, setLoading] = useState(false);
  const storyRef = useRef<HTMLDivElement>(null);

  // Keyboard navigation
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" && activePanel < panels.length - 1) {
        setActivePanel(prev => Math.min(prev + 1, panels.length - 1));
      } else if (e.key === "ArrowLeft" && activePanel > 0) {
        setActivePanel(prev => Math.max(prev - 1, 0));
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [activePanel, panels.length]);

  useEffect(() => {
    fetch("/api/scrolls")
      .then((r) => r.json())
      .then((data) => setScrolls(data.scrolls || []))
      .catch(console.error);
  }, []);

  const openScroll = async (scrollId: string) => {
    setLoading(true);
    setSelectedScroll(scrollId);
    setActivePanel(0);
    try {
      const res = await fetch(`/api/scrolls/${scrollId}`);
      const data = await res.json();
      setPanels(data.panels || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // Scroll selection view
  if (!selectedScroll) {
    return (
      <main className="min-h-screen bg-[#0a0a0a] relative">
        {/* Background */}
        <div className="fixed inset-0 z-0">
          <div className="absolute inset-0 grid grid-cols-3 opacity-[0.5]">
            <img src="/bg_explore1.jpg" alt="" className="w-full h-full object-cover" />
            <img src="/bg_explore2.jpg" alt="" className="w-full h-full object-cover" />
            <img src="/bg_explore3.jpg" alt="" className="w-full h-full object-cover" />
          </div>
          <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0a] via-[#0a0a0a]/50 to-[#0a0a0a]/80" />
          <div className="absolute inset-0 bg-gradient-to-r from-[#0a0a0a] via-transparent to-[#0a0a0a]" />
        </div>

        <header className="sticky top-0 z-50 bg-[#0a0a0a]/80 backdrop-blur-md border-b border-[#2a2a2a]">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <a href="/" className="text-2xl font-bold text-[#d4a574]">GeMi</a>
              <span className="text-sm text-[#606060]">/</span>
              <span className="text-sm text-[#a0a0a0]">Scroll Stories</span>
            </div>
            <a href="/explore" className="px-4 py-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-sm text-[#a0a0a0] hover:text-white hover:border-[#404040] transition-all">
              Back to Explore
            </a>
          </div>
        </header>

        <div className="max-w-5xl mx-auto px-6 py-12 relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <h2 className="text-2xl md:text-4xl font-bold mb-3 text-white">Scroll Stories</h2>
            <p className="text-[#a0a0a0] text-sm md:text-lg mb-6 md:mb-10 max-w-2xl">
              Experience Patachitra scroll paintings as they were meant to be seen — panel by panel, 
              with the accompanying narrative song text. Each scroll tells a complete story.
            </p>
          </motion.div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 md:gap-4">
            {scrolls.map((scroll, i) => (
              <motion.button
                key={scroll.scroll_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                onClick={() => openScroll(scroll.scroll_id)}
                className="group relative rounded-2xl overflow-hidden border border-[#2a2a2a] hover:border-[#d4a574]/50 transition-all hover:shadow-lg hover:shadow-[#d4a574]/10 text-left"
              >
                <div className="aspect-square bg-[#141414]">
                  <img src={scroll.preview_image} alt="" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                </div>
                <div className="absolute inset-0 bg-gradient-to-t from-[#0a0a0a] via-[#0a0a0a]/20 to-transparent" />
                <div className="absolute bottom-0 left-0 right-0 p-4">
                  <h3 className="text-base font-bold text-white mb-1">Scroll {scroll.scroll_id}</h3>
                  <p className="text-xs text-[#a0a0a0]">{scroll.panel_count} panels</p>
                </div>
              </motion.button>
            ))}
          </div>
        </div>
      </main>
    );
  }

  // Story reading view
  const currentPanel = panels[activePanel];

  return (
    <main className="min-h-screen bg-[#050505]">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-[#050505]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <a href="/" className="text-2xl font-bold text-[#d4a574]">GeMi</a>
            <span className="text-sm text-white/20">/</span>
            <span className="text-xs md:text-sm text-white/50">Scroll {selectedScroll}</span>
            <span className="text-sm text-white/20">·</span>
            <span className="text-xs md:text-sm text-[#d4a574]">{activePanel + 1} / {panels.length}</span>
          </div>
          <button
            onClick={() => { setSelectedScroll(null); setPanels([]); }}
            className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white/60 hover:text-white hover:border-white/20 transition-all"
          >
            All Scrolls
          </button>
        </div>
      </header>

      {loading ? (
        <div className="flex items-center justify-center h-screen">
          <div className="w-12 h-12 border-2 border-[#d4a574]/30 border-t-[#d4a574] rounded-full animate-spin" />
        </div>
      ) : currentPanel ? (
        <div className="pt-16 min-h-screen flex">
          {/* Main panel display */}
          <div className="flex-1 flex items-center justify-center px-8 pt-8 pb-28">
            <AnimatePresence mode="wait">
              <motion.div
                key={activePanel}
                initial={{ opacity: 0, x: 80 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -80 }}
                transition={{ duration: 0.35, ease: "easeOut" }}
                className="max-w-4xl w-full flex flex-col md:flex-row gap-4 md:gap-8 items-start"
              >
                {/* Image */}
                <div className="flex-1 relative">
                  <div className="rounded-2xl overflow-hidden border border-white/10 shadow-2xl shadow-black/50">
                    <img
                      src={currentPanel.image_url}
                      alt={currentPanel.id}
                      className="w-full object-contain max-h-[40vh] md:max-h-[60vh]"
                    />
                  </div>
                  {/* Labels */}
                  <div className="flex gap-2 mt-4 justify-center">
                    {currentPanel.animal_label === 1 && <span className="text-xs px-3 py-1.5 rounded-full font-medium" style={{ backgroundColor: "#3B82F620", color: "#3B82F6" }}>Animal</span>}
                    {currentPanel.myth_label === 1 && <span className="text-xs px-3 py-1.5 rounded-full font-medium" style={{ backgroundColor: "#EF444420", color: "#EF4444" }}>Mythology</span>}
                    {currentPanel.tree_label === 1 && <span className="text-xs px-3 py-1.5 rounded-full font-medium" style={{ backgroundColor: "#22C55E20", color: "#22C55E" }}>Tree</span>}
                  </div>
                </div>

                {/* Text narrative */}
                <div className="w-full md:w-80 flex-shrink-0">
                  <div className="sticky top-24">
                    <div className="text-[10px] text-white/30 uppercase tracking-widest mb-2">Panel {activePanel + 1} of {panels.length}</div>
                    <h3 className="text-xl font-bold text-[#d4a574] mb-4">Scroll {currentPanel.scroll_id}, Panel {currentPanel.panel_id}</h3>
                    <div className="bg-white/5 rounded-xl p-5 border border-white/5">
                      <p className="text-sm text-white/70 leading-relaxed italic">
                        {currentPanel.text || "No narrative text available for this panel."}
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Navigation */}
          <div className="fixed bottom-0 left-0 right-0 z-40 bg-[#050505]/90 backdrop-blur-xl border-t border-white/5 py-3">
            <div className="max-w-5xl mx-auto px-6">
              {/* Draggable progress slider */}
              <div className="mb-3 relative group">
                <input
                  type="range"
                  min={0}
                  max={panels.length - 1}
                  value={activePanel}
                  onChange={(e) => setActivePanel(parseInt(e.target.value))}
                  className="w-full h-2 appearance-none bg-white/10 rounded-full cursor-pointer outline-none
                    [&::-webkit-slider-thumb]:appearance-none
                    [&::-webkit-slider-thumb]:w-5
                    [&::-webkit-slider-thumb]:h-5
                    [&::-webkit-slider-thumb]:rounded-full
                    [&::-webkit-slider-thumb]:bg-[#d4a574]
                    [&::-webkit-slider-thumb]:shadow-lg
                    [&::-webkit-slider-thumb]:shadow-[#d4a574]/40
                    [&::-webkit-slider-thumb]:cursor-grab
                    [&::-webkit-slider-thumb]:active:cursor-grabbing
                    [&::-webkit-slider-thumb]:hover:scale-125
                    [&::-webkit-slider-thumb]:transition-transform"
                  style={{
                    background: `linear-gradient(to right, #d4a574 0%, #d4a574 ${(activePanel / Math.max(panels.length - 1, 1)) * 100}%, rgba(255,255,255,0.1) ${(activePanel / Math.max(panels.length - 1, 1)) * 100}%, rgba(255,255,255,0.1) 100%)`
                  }}
                />
                <div className="flex justify-between text-[9px] text-white/20 mt-1 px-0.5">
                  <span>Panel 1</span>
                  <span>Panel {panels.length}</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                {/* Prev */}
                <button
                  onClick={() => setActivePanel(Math.max(0, activePanel - 1))}
                  disabled={activePanel === 0}
                  className="px-3 md:px-5 py-1.5 md:py-2 bg-white/5 border border-white/10 rounded-xl text-xs md:text-sm text-white/60 hover:text-white hover:border-white/20 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Previous
                </button>

                {/* Thumbnail strip */}
                <div className="flex gap-1 md:gap-1.5 overflow-x-auto max-w-[50vw] md:max-w-lg px-2 md:px-4">
                  {panels.map((p, i) => (
                    <button
                      key={p.index}
                      onClick={() => setActivePanel(i)}
                      className={`flex-shrink-0 w-7 h-7 md:w-10 md:h-10 rounded-md md:rounded-lg overflow-hidden border-2 transition-all ${
                        i === activePanel
                          ? "border-[#d4a574] shadow-lg shadow-[#d4a574]/30 scale-110"
                          : i < activePanel
                          ? "border-white/20 opacity-60"
                          : "border-white/10 opacity-40"
                      }`}
                    >
                      <img src={p.image_url} alt="" className="w-full h-full object-cover" />
                    </button>
                  ))}
                </div>

                {/* Next */}
                <button
                  onClick={() => setActivePanel(Math.min(panels.length - 1, activePanel + 1))}
                  disabled={activePanel === panels.length - 1}
                  className="px-3 md:px-5 py-1.5 md:py-2 bg-[#d4a574] text-[#0a0a0a] font-semibold rounded-xl text-xs md:text-sm hover:bg-[#e8c49a] transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
