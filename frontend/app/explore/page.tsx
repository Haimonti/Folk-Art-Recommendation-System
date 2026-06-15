"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Heart, Sparkles, Zap } from "lucide-react";
import { panelName } from "../../lib/panelName";
import { logEvent, logRecommendation } from "../../lib/analytics";
import { createPortal } from "react-dom";

interface Panel {
  index: number;
  id: string;
  scroll_id: string;
  panel_id: string;
  text: string;
  animal_label: number;
  myth_label: number;
  tree_label: number;
  split: string;
  image_filename: string;
  image_url: string;
}

const ANIMAL = "\uD83D\uDC18";
const MYTH = "\uD83C\uDFDB\uFE0F";
const TREE = "\uD83C\uDF33";
const PALETTE = "\uD83C\uDFA8";

const CATEGORIES = [
  { key: "all", label: "All Panels", emoji: PALETTE, description: "Browse the full collection" },
  { key: "animal", label: "Animals", emoji: ANIMAL, description: "Elephants, horses, birds & more" },
  { key: "mythology", label: "Mythology", emoji: MYTH, description: "Gods, epics & divine tales" },
  { key: "tree", label: "Trees & Nature", emoji: TREE, description: "Sacred groves & landscapes" },
  { key: "animal+mythology", label: "Animals & Mythology", emoji: ANIMAL + MYTH, description: "Divine creatures & mounts" },
  { key: "mythology+tree", label: "Mythology & Nature", emoji: MYTH + TREE, description: "Sacred scenes in nature" },
  { key: "tree+animal", label: "Trees & Animals", emoji: TREE + ANIMAL, description: "Wildlife amid sacred groves" },
];

function matchesCategory(panel: Panel, category: string): boolean {
  if (category === "all") return true;
  if (category === "animal") return panel.animal_label === 1;
  if (category === "mythology") return panel.myth_label === 1;
  if (category === "tree") return panel.tree_label === 1;
  if (category === "animal+mythology") return panel.animal_label === 1 && panel.myth_label === 1;
  if (category === "mythology+tree") return panel.myth_label === 1 && panel.tree_label === 1;
  if (category === "tree+animal") return panel.tree_label === 1 && panel.animal_label === 1;
  return true;
}

export default function ExplorePage() {
  const [panels, setPanels] = useState<Panel[]>([]);
  const [liked, setLiked] = useState<Set<number>>(new Set());
  const [concepts, setConcepts] = useState({ animal: true, mythology: true, tree: true });
  const [selectedModel, setSelectedModel] = useState("llamasigclip_vgae");
  const [featureBackbone, setFeatureBackbone] = useState("llamasigclip");
  const [setting, setSetting] = useState("transductive");
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [comparison, setComparison] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState<"browse" | "results" | "compare">("browse");
  const [activeCategory, setActiveCategory] = useState("all");
  const [expandedPanel, setExpandedPanel] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/panels?limit=200")
      .then((r) => r.json())
      .then((data) => setPanels(data.panels || []))
      .catch(console.error);
  }, []);
  useEffect(() => {
    if (typeof window !== "undefined") window.scrollTo({ top: 0, behavior: "smooth" });
  }, [step]);

  const toggleLike = (index: number) => {
    const panel = panels.find((x) => x.index === index);
    const wasLiked = liked.has(index);
    logEvent(wasLiked ? "unlike" : "like", { panelIndex: index, panelId: panel?.id });
    setLiked((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const toggleConcept = (key: string) => {
    setConcepts((prev) => ({ ...prev, [key]: !prev[key as keyof typeof prev] }));
  };

  const getRecommendations = async () => {
    if (liked.size === 0) return;
    setLoading(true);
    try {
      const sessionRes = await fetch("/api/session/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          liked_panel_indices: Array.from(liked),
          concept_preferences: concepts,
          description: null,
        }),
      });
      const session = await sessionRes.json();

      const recRes = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session.session_id, model_name: selectedModel, top_k: 8 }),
      });
      const recData = await recRes.json();
      setRecommendations(recData.recommendations || []);
      logRecommendation({
        modelName: selectedModel,
        seedPanelIndices: Array.from(liked),
        servedPanelIndices: (recData.recommendations || []).map((r: any) => r.index),
      });

      const compareModels = ["gcn", "gae", "vgae"].map((g) => setting === "inductive" ? featureBackbone + "_ind_" + g : featureBackbone + "_" + g);
      const compRes = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session.session_id, models: compareModels, top_k: 5 }),
      });
      const compData = await compRes.json();
      setComparison(compData.comparison || null);
      setStep("results");
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const filteredPanels = panels.filter((p) => matchesCategory(p, activeCategory));

  // ══════════════════════════════════════════════════════════
  // BROWSE VIEW
  // ══════════════════════════════════════════════════════════
  if (step === "browse") {
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
        {/* Header */}
        <header className="sticky top-0 z-50 bg-[#0a0a0a]/80 backdrop-blur-md border-b border-[#2a2a2a]">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <a href="/" className="text-3xl md:text-4xl font-bold text-[#d4a574]">GeMi</a>
            <div className="flex items-center gap-2 md:gap-3 flex-wrap">
              <a href="/story" className="px-2 md:px-4 py-1.5 md:py-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-base md:text-lg font-medium text-[#c2c2c2] hover:text-white hover:border-[#404040] transition-all">
                Scroll Stories
              </a>
              {liked.size > 0 && (
                <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} className="bg-[#d4a574]/20 text-[#d4a574] px-3 py-1 rounded-full text-sm">
                  {liked.size} liked
                </motion.span>
              )}
            </div>
          </div>
        </header>

        <div className="max-w-7xl mx-auto px-6 py-8 relative z-10">
          {/* Title */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
            <h2 className="text-2xl md:text-3xl font-bold mb-2">Explore Scroll Paintings</h2>
            <p className="text-base md:text-lg text-[#c8c8c8]">Choose a category to browse, like the panels that interest you, then get personalized recommendations.</p>
          </motion.div>

          {/* Category Cards */}
          <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-2 md:gap-3 mb-8 md:mb-10">
            {CATEGORIES.map((cat, i) => (
              <motion.button
                key={cat.key}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                onClick={() => setActiveCategory(cat.key)}
                className={`relative p-3 md:p-5 rounded-xl border text-left transition-all ${
                  activeCategory === cat.key
                    ? "bg-[#d4a574]/15 border-[#d4a574] shadow-lg shadow-[#d4a574]/10"
                    : "bg-[#141414] border-[#2a2a2a] hover:border-[#404040] hover:bg-[#1a1a1a]"
                }`}
              >
                <div className="text-2xl md:text-3xl mb-1 md:mb-2">{cat.emoji}</div>
                <div className={`text-sm md:text-base font-semibold mb-0.5 md:mb-1 ${activeCategory === cat.key ? "text-[#d4a574]" : "text-white"}`}>{cat.label}</div>
                <div className="text-sm md:text-base text-[#c2c2c2] leading-snug hidden md:block">{cat.description}</div>
                <div className={`absolute top-3 right-3 text-xs px-1.5 py-0.5 rounded-full ${
                  activeCategory === cat.key ? "bg-[#d4a574] text-[#0a0a0a]" : "bg-[#2a2a2a] text-[#9a9a9a]"
                }`}>
                  {panels.filter((p) => matchesCategory(p, cat.key)).length}
                </div>
              </motion.button>
            ))}
          </div>

          {/* Active category heading */}
          <div className="flex items-center gap-3 mb-6">
            <h3 className="text-xl font-semibold">
              {CATEGORIES.find((c) => c.key === activeCategory)?.emoji}{" "}
              {CATEGORIES.find((c) => c.key === activeCategory)?.label}
            </h3>
            <span className="text-sm text-[#9a9a9a]">{filteredPanels.length} panels</span>
          </div>

          {/* Panel Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2 md:gap-4 mb-32">
            {filteredPanels.map((panel, i) => (
              <motion.div
                key={panel.index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.02, 0.5) }}
                whileHover={{ scale: 1.05, y: -8, boxShadow: "0 18px 40px rgba(212,165,116,0.35)", transition: { type: "spring", stiffness: 380, damping: 18 } }}
                whileTap={{ scale: 0.97 }}
                onClick={() => { setExpandedPanel(panel.index); logEvent("view", { panelIndex: panel.index, panelId: panel.id, context: { source: "grid" } }); }}
                className={`relative group rounded-xl overflow-hidden border-2 transition-all cursor-pointer ${
                  liked.has(panel.index)
                    ? "border-[#d4a574] shadow-lg shadow-[#d4a574]/20"
                    : "border-[#2a2a2a] hover:border-[#404040]"
                }`}
              >
                <div className="aspect-square bg-[#141414]">
                  <img src={panel.image_url} alt={panel.id} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" loading="lazy" />
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); toggleLike(panel.index); }}
                  className="absolute top-2 right-2 p-2 rounded-full bg-black/60 backdrop-blur-sm hover:bg-black/80 transition-colors"
                >
                  <Heart className={`w-4 h-4 transition-colors ${liked.has(panel.index) ? "fill-[#d4a574] text-[#d4a574]" : "text-white/70"}`} />
                </button>
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3 pt-8">
                  <div className="flex gap-1.5">
                    {panel.animal_label === 1 && <span className="text-xs font-medium bg-[#2166AC]/80 text-white px-1.5 py-0.5 rounded">{ANIMAL} Animal</span>}
                    {panel.myth_label === 1 && <span className="text-xs font-medium bg-[#B2182B]/80 text-white px-1.5 py-0.5 rounded">{MYTH} Myth</span>}
                    {panel.tree_label === 1 && <span className="text-xs font-medium bg-[#1B7837]/80 text-white px-1.5 py-0.5 rounded">{TREE} Tree</span>}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Panel Detail Modal */}
          {typeof document !== "undefined" && createPortal(
          <AnimatePresence>
            {expandedPanel !== null && (
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                onClick={() => setExpandedPanel(null)}
                className="fixed inset-0 bg-black/75 backdrop-blur-sm z-[100] flex items-center justify-center p-4"
              >
                <motion.div
                  initial={{ scale: 0.92, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.92, opacity: 0 }}
                  transition={{ type: "spring", stiffness: 260, damping: 24 }}
                  onClick={(e) => e.stopPropagation()}
                  className="bg-[#1a1a1a] rounded-2xl w-full max-w-2xl md:max-w-4xl max-h-[88vh] p-5 border border-[#2a2a2a] flex flex-col md:flex-row gap-5"
                >
                  {(() => {
                    const p = panels.find((x) => x.index === expandedPanel);
                    if (!p) return null;
                    return (
                      <>
                        <div className="md:w-1/2 flex items-center justify-center shrink-0">
                          <img src={p.image_url} alt={p.id} className="w-auto max-h-[38vh] md:max-h-[76vh] rounded-xl object-contain" />
                        </div>
                        <div className="md:w-1/2 flex flex-col min-h-0">
                          <h3 className="text-2xl md:text-3xl font-bold mb-2">{panelName(p.id)}</h3>
                          <div className="flex gap-2 mb-3 flex-wrap">
                            {p.animal_label === 1 && <span className="text-sm font-medium bg-[#2166AC]/80 text-white px-2 py-1 rounded">{ANIMAL} Animal</span>}
                            {p.myth_label === 1 && <span className="text-sm font-medium bg-[#B2182B]/80 text-white px-2 py-1 rounded">{MYTH} Mythology</span>}
                            {p.tree_label === 1 && <span className="text-sm font-medium bg-[#1B7837]/80 text-white px-2 py-1 rounded">{TREE} Tree</span>}
                          </div>
                          <p className="italic text-[#d8d8d8] text-base md:text-lg leading-relaxed flex-1 min-h-0 overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">{p.text || "No text available for this panel."}</p>
                          <button
                            onClick={() => { toggleLike(p.index); setExpandedPanel(null); }}
                            className={`mt-4 w-full py-3 rounded-lg font-medium transition-colors shrink-0 ${
                              liked.has(p.index)
                                ? "bg-[#d4a574]/20 text-[#d4a574] border border-[#d4a574]"
                                : "bg-[#d4a574] text-[#0a0a0a] hover:bg-[#e8c49a]"
                            }`}
                          >
                            {liked.has(p.index) ? "Liked" : "Like this panel"}
                          </button>
                        </div>
                      </>
                    );
                  })()}
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>,
          document.body)}
        </div>

        {/* ── Sticky Bottom Bar ── */}
        <AnimatePresence>
        {liked.size > 0 && expandedPanel === null && (
          <motion.div
            key="bottombar"
            initial={{ y: 120, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 120, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="fixed bottom-4 left-1/2 -translate-x-1/2 max-w-[95vw] bg-[#141414]/95 backdrop-blur-md border border-[#2a2a2a] rounded-2xl shadow-2xl py-3 px-4 z-40"
          >
            <div className="flex flex-wrap items-end justify-center gap-3 md:gap-6">
              <div className="flex flex-col items-center gap-1.5">
                <span className="text-sm font-semibold text-[#b8b8b8] uppercase tracking-wider">Interests</span>
                <div className="flex gap-1.5">
                  {[
                    { key: "animal", label: ANIMAL + " Animal" },
                    { key: "mythology", label: MYTH + " Mythology" },
                    { key: "tree", label: TREE + " Tree" },
                  ].map((c) => (
                    <button
                      key={c.key}
                      onClick={() => toggleConcept(c.key)}
                      className={`px-2.5 py-1 rounded-full text-xs font-medium transition-all ${
                        concepts[c.key as keyof typeof concepts]
                          ? "bg-[#d4a574]/20 text-[#d4a574] border border-[#d4a574]/50"
                          : "bg-[#1a1a1a] text-[#9a9a9a] border border-[#2a2a2a]"
                      }`}
                    >
                      {c.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col items-center gap-1.5">
                <span className="text-sm font-semibold text-[#b8b8b8] uppercase tracking-wider">Setting</span>
                <div className="flex bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] overflow-hidden">
                  {["transductive", "inductive"].map((s) => (
                    <button
                      key={s}
                      onClick={() => {
                        setSetting(s);
                        const prefix = s === "inductive" ? featureBackbone + "_ind" : featureBackbone;
                        setSelectedModel(prefix + "_vgae");
                      }}
                      className={`px-2.5 py-1.5 text-xs font-medium transition-all ${
                        setting === s ? "bg-[#d4a574] text-[#0a0a0a]" : "text-[#c2c2c2] hover:text-white"
                      }`}
                    >
                      {s === "transductive" ? "Trans." : "Ind."}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col items-center gap-1.5">
                <span className="text-sm font-semibold text-[#b8b8b8] uppercase tracking-wider">Feature Backbone</span>
                <div className="flex bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] overflow-hidden">
                  {["llamasigclip", "llamavae"].map((f) => (
                    <button
                      key={f}
                      onClick={() => { setFeatureBackbone(f); setSelectedModel(setting === "inductive" ? f + "_ind_vgae" : f + "_vgae"); }}
                      className={`px-3 py-1.5 text-xs font-medium transition-all ${
                        featureBackbone === f ? "bg-[#d4a574] text-[#0a0a0a]" : "text-[#c2c2c2] hover:text-white"
                      }`}
                    >
                      {f === "llamasigclip" ? "SigCLIP" : "VAE"}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col items-center gap-1.5">
                <span className="text-sm font-semibold text-[#b8b8b8] uppercase tracking-wider">GNN Model</span>
                <div className="flex bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] overflow-hidden">
                  {["gcn", "gae", "vgae"].map((g) => {
                    const key = setting === "inductive" ? featureBackbone + "_ind_" + g : featureBackbone + "_" + g;
                    return (
                      <button
                        key={g}
                        onClick={() => setSelectedModel(key)}
                        className={`px-3 py-1.5 text-xs font-medium transition-all ${
                          selectedModel === key ? "bg-[#d4a574] text-[#0a0a0a]" : "text-[#c2c2c2] hover:text-white"
                        }`}
                      >
                        {g.toUpperCase()}
                      </button>
                    );
                  })}
                </div>
              </div>

              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={getRecommendations}
                disabled={loading}
                className="flex items-center gap-2 px-5 py-2 bg-[#d4a574] text-[#0a0a0a] font-semibold rounded-lg hover:bg-[#e8c49a] transition-colors disabled:opacity-50 text-sm"
              >
                {loading ? "..." : <><Sparkles className="w-4 h-4" /> Recommend</>}
              </motion.button>
            </div>
          </motion.div>
        )}
        </AnimatePresence>
      </main>
    );
  }

  // ══════════════════════════════════════════════════════════
  // RESULTS + COMPARE VIEW
  // ══════════════════════════════════════════════════════════
  return (
    <main className="min-h-screen bg-[#0a0a0a] relative">
      {/* Background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 grid grid-cols-3 opacity-[0.5]">
          <img src="/bg_results1.jpg" alt="" className="w-full h-full object-cover" />
          <img src="/bg_results2.jpg" alt="" className="w-full h-full object-cover" />
          <img src="/bg_results3.jpg" alt="" className="w-full h-full object-cover" />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0a] via-[#0a0a0a]/50 to-[#0a0a0a]/80" />
        <div className="absolute inset-0 bg-gradient-to-r from-[#0a0a0a] via-transparent to-[#0a0a0a]" />
        
      </div>
      <header className="sticky top-0 z-50 bg-[#0a0a0a]/80 backdrop-blur-md border-b border-[#2a2a2a]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <a href="/" className="text-2xl font-bold text-[#d4a574]">GeMi</a>
          <div className="flex gap-2 md:gap-3 flex-wrap justify-end">
            
            <a
              href={`/graph?model=${selectedModel}&panel=0&depth=2`}
              className="flex items-center gap-1 md:gap-1.5 px-2 md:px-4 py-1.5 md:py-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-sm md:text-base font-medium text-[#c2c2c2] hover:text-white hover:border-[#404040] transition-all"
            >
              Graph Explorer
            </a>
            <button
              onClick={() => setStep(step === "compare" ? "results" : "compare")}
              className="flex items-center gap-1 md:gap-1.5 px-2 md:px-4 py-1.5 md:py-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-sm md:text-base font-medium text-[#c2c2c2] hover:text-white hover:border-[#404040] transition-all"
            >
              <Zap className="w-3.5 h-3.5" />
              {step === "compare" ? "Single View" : "Compare Models"}
            </button>
            <button
              onClick={() => setStep("browse")}
              className="px-4 py-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg text-base md:text-lg font-medium text-[#c2c2c2] hover:text-white hover:border-[#404040] transition-all"
            >
              Back
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8 relative z-10">
        {step === "results" ? (
          <>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
              <h2 className="text-3xl font-bold mb-2">Your Recommendations</h2>
              <p className="text-[#c2c2c2]">
                Via <span className="text-[#d4a574] font-medium">{selectedModel.replace("llamasigclip_ind_", "LlamaSigCLIP (Inductive) + ").replace("llamavae_ind_", "LlamaVAE (Inductive) + ").replace("llamasigclip_", "LlamaSigCLIP + ").replace("llamavae_", "LlamaVAE + ").replace("vgae", "VGAE").replace("gcn", "GCN").replace("gae", "GAE")}</span> based on {liked.size} liked panels
              </p>
            </motion.div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
              {recommendations.map((rec, i) => (
                <motion.div
                  key={rec.index}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="bg-[#1a1a1a] rounded-2xl overflow-hidden border border-[#2a2a2a] hover:border-[#d4a574]/50 transition-all"
                >
                  <div className="relative">
                    <img src={rec.image_url} alt={rec.id} className="w-full aspect-square object-cover" />
                    <div className="absolute top-3 left-3 bg-[#d4a574] text-[#0a0a0a] px-2.5 py-1 rounded-full text-xs font-bold">#{rec.rank}</div>
                    <div className="absolute top-3 right-3 bg-black/70 backdrop-blur-sm px-2.5 py-1 rounded-full text-xs text-[#d4a574] font-medium">
                      {(rec.similarity_score * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-4">
                    <div className="flex gap-1.5 mb-2">
                      {rec.animal_label === 1 && <span className="text-xs bg-[#2166AC]/20 text-[#6baed6] px-1.5 py-0.5 rounded">{ANIMAL}</span>}
                      {rec.myth_label === 1 && <span className="text-xs bg-[#B2182B]/20 text-[#fc8d62] px-1.5 py-0.5 rounded">{MYTH}</span>}
                      {rec.tree_label === 1 && <span className="text-xs bg-[#1B7837]/20 text-[#66c2a5] px-1.5 py-0.5 rounded">{TREE}</span>}
                      {rec.concept_matches?.length > 0 && <span className="text-xs text-[#d4a574] ml-auto">matches</span>}
                    </div>
                    <p className="text-xs text-[#c2c2c2] line-clamp-3">{rec.text || "No text."}</p>
                    {rec.explanation?.per_panel_similarity?.length > 0 && (
                      <details className="mt-3">
                        <summary className="text-xs text-[#9a9a9a] cursor-pointer hover:text-[#c2c2c2]">Why this?</summary>
                        <div className="mt-2 space-y-1">
                          {rec.explanation.per_panel_similarity.slice(0, 3).map((s: any) => (
                            <div key={s.liked_panel_id} className="flex justify-between text-xs">
                              <span className="text-[#c2c2c2]">~ {s.liked_panel_id}</span>
                              <span className="text-[#d4a574]">{(s.similarity * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                    
                    <a
                      href={`/graph?model=${selectedModel}&panel=${rec.index}&depth=2`}
                      className="mt-3 flex items-center gap-1 text-xs text-[#d4a574]/70 hover:text-[#d4a574] transition-colors"
                    >
                      <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="5" r="2"/><circle cx="20" cy="5" r="2"/><circle cx="4" cy="19" r="2"/><circle cx="20" cy="19" r="2"/><line x1="6" y1="6" x2="10" y2="10"/><line x1="14" y1="10" x2="18" y2="6"/><line x1="6" y1="18" x2="10" y2="14"/><line x1="14" y1="14" x2="18" y2="18"/></svg>
                      View in Graph
                    </a>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        ) : (
          <>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold mb-2">Model Comparison</h2>
                  <p className="text-[#c2c2c2]">How GCN, GAE, and VGAE recommend differently from the same liked panels.</p>
                </div>
                <div className="flex items-end gap-4">
                  <div className="flex flex-col items-center gap-1.5">
                    <span className="text-sm font-semibold text-[#b8b8b8] uppercase tracking-wider">Setting</span>
                    <div className="flex bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] overflow-hidden">
                      {["transductive", "inductive"].map((s) => (
                        <button
                          key={s}
                          onClick={async () => {
                            setSetting(s);
                            const compareModels = ["gcn", "gae", "vgae"].map((g) => s === "inductive" ? featureBackbone + "_ind_" + g : featureBackbone + "_" + g);
                            try {
                              const sessionRes = await fetch("/api/session/create", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ liked_panel_indices: Array.from(liked), concept_preferences: concepts, description: null }),
                              });
                              const session = await sessionRes.json();
                              const compRes = await fetch("/api/compare", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ session_id: session.session_id, models: compareModels, top_k: 5 }),
                              });
                              const compData = await compRes.json();
                              setComparison(compData.comparison || null);
                            } catch (e) { console.error(e); }
                          }}
                          className={`px-2.5 py-1.5 text-xs font-medium transition-all ${
                            setting === s ? "bg-[#d4a574] text-[#0a0a0a]" : "text-[#c2c2c2] hover:text-white"
                          }`}
                        >
                          {s === "transductive" ? "Trans." : "Ind."}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-col items-center gap-1.5">
                    <span className="text-sm font-semibold text-[#b8b8b8] uppercase tracking-wider">Feature Backbone</span>
                  <div className="flex bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] overflow-hidden">
                    {["llamasigclip", "llamavae"].map((f) => (
                      <button
                        key={f}
                        onClick={async () => {
                          setFeatureBackbone(f);
                          const prefix = setting === "inductive" ? f + "_ind" : f;
                          setSelectedModel(prefix + "_vgae");
                          const compareModels = ["gcn", "gae", "vgae"].map((g) => setting === "inductive" ? f + "_ind_" + g : f + "_" + g);
                          try {
                            const sessionRes = await fetch("/api/session/create", {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({
                                liked_panel_indices: Array.from(liked),
                                concept_preferences: concepts,
                                description: null,
                              }),
                            });
                            const session = await sessionRes.json();
                            const compRes = await fetch("/api/compare", {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ session_id: session.session_id, models: compareModels, top_k: 5 }),
                            });
                            const compData = await compRes.json();
                            setComparison(compData.comparison || null);
                          } catch (e) { console.error(e); }
                        }}
                        className={`px-4 py-2 text-sm font-medium transition-all ${
                          featureBackbone === f ? "bg-[#d4a574] text-[#0a0a0a]" : "text-[#c2c2c2] hover:text-white"
                        }`}
                      >
                        {f === "llamasigclip" ? "LlamaSigCLIP" : "LlamaVAE"}
                      </button>
                    ))}
                  </div>
                  </div>
                </div>
              </div>
            </motion.div>
            {comparison && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
                {Object.entries(comparison).map(([modelName, recs]: [string, any]) => {
                  if (recs.error) return null;
                  const gnn = modelName.split("_").pop()?.toUpperCase();
                  const colors: Record<string, string> = { GCN: "#2166AC", GAE: "#B2182B", VGAE: "#1B7837" };
                  return (
                    <motion.div key={modelName} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                      className="bg-[#141414] rounded-2xl border border-[#2a2a2a] overflow-hidden"
                    >
                      <div className="p-4 border-b border-[#2a2a2a]" style={{ borderTopColor: colors[gnn || ""] || "#d4a574", borderTopWidth: 3 }}>
                        <h3 className="text-lg font-bold" style={{ color: colors[gnn || ""] }}>GeMi-{gnn}</h3>
                        <p className="text-xs text-[#c2c2c2]">{modelName.includes("sigclip") ? "LlamaSigCLIP" : "LlamaVAE"}</p>
                      </div>
                      <div className="divide-y divide-[#1a1a1a]">
                        {(Array.isArray(recs) ? recs : []).slice(0, 5).map((rec: any) => (
                          <div key={rec.index} className="flex gap-3 p-3">
                            <img src={rec.image_url} alt="" className="w-14 h-14 rounded-lg object-cover flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="text-xs font-bold text-[#d4a574]">#{rec.rank}</span>
                                <span className="text-xs text-[#c2c2c2]">{(rec.similarity_score * 100).toFixed(1)}%</span>
                                <div className="flex gap-1 ml-auto">
                                  {rec.animal_label === 1 && <span className="text-xs">{ANIMAL}</span>}
                                  {rec.myth_label === 1 && <span className="text-xs">{MYTH}</span>}
                                  {rec.tree_label === 1 && <span className="text-xs">{TREE}</span>}
                                </div>
                              </div>
                              <p className="text-xs text-[#9a9a9a] truncate">{rec.text || rec.id}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}
