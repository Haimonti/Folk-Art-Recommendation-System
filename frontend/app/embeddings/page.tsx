"use client";

import { useState, useEffect, useRef, useCallback } from "react";

function useWindowSize() {
  const [size, setSize] = useState({ width: 800, height: 600 });
  useEffect(() => {
    const update = () => setSize({ width: window.innerWidth, height: window.innerHeight });
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return size;
}
import { motion, AnimatePresence } from "framer-motion";
import { panelName } from "../../lib/panelName";
import dynamic from "next/dynamic";

const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), { ssr: false });

interface EmbNode {
  id: number;
  panel_id: string;
  image_url: string;
  animal: number;
  mythology: number;
  tree: number;
  split: string;
  x: number; y: number; z: number;
  fx?: number; fy?: number; fz?: number;
}

function getNodeColor(node: EmbNode): string {
  const a = node.animal, m = node.mythology, t = node.tree;
  if (a && m && t) return "#FFD700";
  if (a && m) return "#A855F7";
  if (a && t) return "#06B6D4";
  if (m && t) return "#F97316";
  if (a) return "#3B82F6";
  if (m) return "#EF4444";
  if (t) return "#22C55E";
  return "#6B7280";
}

function getNodeLabel(node: EmbNode): string {
  const labels = [];
  if (node.animal) labels.push("Animal");
  if (node.mythology) labels.push("Mythology");
  if (node.tree) labels.push("Tree");
  return labels.length > 0 ? labels.join(" + ") : "No labels";
}

const MODELS = [
  { key: "llamasigclip_vgae", label: "SigCLIP + VGAE (Trans.)" },
  { key: "llamasigclip_gcn", label: "SigCLIP + GCN (Trans.)" },
  { key: "llamasigclip_gae", label: "SigCLIP + GAE (Trans.)" },
  { key: "llamavae_vgae", label: "VAE + VGAE (Trans.)" },
  { key: "llamavae_gcn", label: "VAE + GCN (Trans.)" },
  { key: "llamavae_gae", label: "VAE + GAE (Trans.)" },
  { key: "llamasigclip_ind_vgae", label: "SigCLIP + VGAE (Ind.)" },
  { key: "llamasigclip_ind_gcn", label: "SigCLIP + GCN (Ind.)" },
  { key: "llamasigclip_ind_gae", label: "SigCLIP + GAE (Ind.)" },
  { key: "llamavae_ind_vgae", label: "VAE + VGAE (Ind.)" },
  { key: "llamavae_ind_gcn", label: "VAE + GCN (Ind.)" },
  { key: "llamavae_ind_gae", label: "VAE + GAE (Ind.)" },
];

const LEGEND = [
  { color: "#3B82F6", label: "Animal" },
  { color: "#EF4444", label: "Mythology" },
  { color: "#22C55E", label: "Tree" },
  { color: "#A855F7", label: "Animal + Myth" },
  { color: "#06B6D4", label: "Animal + Tree" },
  { color: "#F97316", label: "Myth + Tree" },
  { color: "#FFD700", label: "All Three" },
];

export default function EmbeddingsPage() {
  const [nodes, setNodes] = useState<EmbNode[]>([]);
  const [selectedModel, setSelectedModel] = useState("llamasigclip_vgae");
  const [loading, setLoading] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<EmbNode | null>(null);
  const [useImages, setUseImages] = useState(true);
  const [imageCache, setImageCache] = useState<Record<string, any>>({});
  const [showSplit, setShowSplit] = useState(false);
  const [transitioning, setTransitioning] = useState(false);
  const fgRef = useRef<any>(null);
  const windowSize = useWindowSize();
  const [showControls, setShowControls] = useState(false);

  const fetchTSNE = useCallback(async (model: string) => {
    setLoading(true);
    setTransitioning(true);
    try {
      const res = await fetch(`/api/tsne/${model}`);
      const data = await res.json();
      if (data.nodes) {
        const fixed = data.nodes.map((n: any) => ({
          ...n,
          fx: n.x * 3,
          fy: n.y * 3,
          fz: n.z * 3,
        }));
        setNodes(fixed);
        setTimeout(() => {
          if (fgRef.current) fgRef.current.zoomToFit(1200, 60);
          setTransitioning(false);
        }, 500);
      }
    } catch (e) {
      console.error(e);
      setTransitioning(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTSNE(selectedModel);
  }, []);

  // Auto-resize and center on window resize
  useEffect(() => {
    const handleResize = () => {
      if (fgRef.current) {
        fgRef.current.zoomToFit(600, 40);
      }
    };
    window.addEventListener("resize", handleResize);
    const timer = setTimeout(() => {
      if (fgRef.current) fgRef.current.zoomToFit(800, 40);
    }, 2000);
    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(timer);
    };
  }, [nodes]);

  // Preload images
  useEffect(() => {
    if (typeof window === "undefined" || nodes.length === 0) return;
    const THREE = require("three");
    const loader = new THREE.TextureLoader();
    nodes.forEach((n) => {
      if (!imageCache[n.image_url]) {
        loader.load(n.image_url, (texture: any) => {
          setImageCache((prev) => ({ ...prev, [n.image_url]: texture }));
        });
      }
    });
  }, [nodes]);

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
    fetchTSNE(model);
  };

  const nodeThreeObject = useCallback((node: any) => {
    if (typeof window === "undefined") return undefined;
    const THREE = require("three");
    const n = node as EmbNode;

    if (useImages && imageCache[n.image_url]) {
      const size = 3.5;
      const geometry = new THREE.SphereGeometry(size, 24, 24);
      const material = new THREE.MeshBasicMaterial({
        map: imageCache[n.image_url],
        transparent: true,
        opacity: showSplit && n.split === "test" ? 0.5 : 0.95,
      });
      const mesh = new THREE.Mesh(geometry, material);

      const glowColor = new THREE.Color(getNodeColor(n));
      const ringGeo = new THREE.RingGeometry(size + 0.5, size + 1.2, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: glowColor,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
      });
      mesh.add(new THREE.Mesh(ringGeo, ringMat));

      if (showSplit && n.split === "test") {
        const outerGeo = new THREE.RingGeometry(size + 1.2, size + 2, 24);
        const outerMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color("#FFFFFF"),
          transparent: true,
          opacity: 0.3,
          side: THREE.DoubleSide,
        });
        mesh.add(new THREE.Mesh(outerGeo, outerMat));
      }

      return mesh;
    }
    return undefined;
  }, [useImages, imageCache, showSplit]);

  const graphData = {
    nodes: nodes,
    links: [],
  };

  // Count stats
  const animalCount = nodes.filter(n => n.animal).length;
  const mythCount = nodes.filter(n => n.mythology).length;
  const treeCount = nodes.filter(n => n.tree).length;
  const trainCount = nodes.filter(n => n.split === "train").length;
  const testCount = nodes.filter(n => n.split === "test").length;

  return (
    <main className="min-h-screen bg-[#020208] relative overflow-hidden">
      {/* Ambient glow */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-[#A855F7]/5 rounded-full blur-[150px]" />
        <div className="absolute bottom-1/3 right-1/3 w-[600px] h-[600px] bg-[#06B6D4]/5 rounded-full blur-[150px]" />
      </div>

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-[#020208]/60 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-3 md:px-6 py-2 md:py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <a href="/" className="text-2xl md:text-3xl font-bold text-[#d4a574]">GeMi</a>
            <span className="text-sm text-white/20 hidden sm:inline">/</span>
            <span className="text-sm md:text-base text-white/70 font-semibold hidden sm:inline">Embedding Space</span>
          </div>
          <div className="flex gap-3">
            <a href="/story" className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm md:text-base font-medium text-white/75 hover:text-white hover:border-white/20 transition-all">
              Scroll Stories
            </a>
            <a href="/graph" className="px-2 md:px-4 py-1.5 md:py-2 bg-white/5 border border-white/10 rounded-lg text-sm md:text-base font-medium text-white/75 hover:text-white hover:border-white/20 transition-all">
              Graph Explorer
            </a>
            <a href="/explore" className="px-2 md:px-4 py-1.5 md:py-2 bg-white/5 border border-white/10 rounded-lg text-sm md:text-base font-medium text-white/75 hover:text-white hover:border-white/20 transition-all">
              Back to Explore
            </a>
          </div>
        </div>
      </header>

      {/* Mobile toggle */}
      <button
        onClick={() => setShowControls(!showControls)}
        className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 block md:hidden bg-[#d4a574] text-[#0a0a0a] px-5 py-2.5 rounded-full text-sm font-bold shadow-lg shadow-[#d4a574]/30"
      >
        {showControls ? "Close" : "Controls"}
      </button>

      {/* Controls */}
      <motion.div
        initial={{ x: -300, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        style={{ display: !showControls && windowSize.width < 768 ? "none" : "block" }}
        className="fixed z-40 bg-[#0a0a12]/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl shadow-black/50 overflow-y-auto left-2 right-2 bottom-14 max-h-[60vh] p-3 md:left-4 md:right-auto md:bottom-auto md:top-20 md:w-72 md:max-h-none md:p-4"
      >
        <h3 className="text-xl md:text-2xl font-bold text-[#d4a574] mb-3 md:mb-4">Embedding Space</h3>
        <p className="text-sm text-white/55 mb-4 leading-relaxed">3D t-SNE projection of all 189 panel embeddings. Switch models to see how GNN variants organize the latent space.</p>

        {/* Model */}
        <div className="mb-3">
          <label className="text-sm text-white/55 font-semibold uppercase tracking-widest block mb-2">Model</label>
          <select
            value={selectedModel}
            onChange={(e) => handleModelChange(e.target.value)}
            className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-base font-medium text-white focus:outline-none focus:border-[#d4a574]/50"
          >
            {MODELS.map((m) => (
              <option key={m.key} value={m.key} className="bg-[#0a0a12]">{m.label}</option>
            ))}
          </select>
        </div>

        {/* Toggles */}
        <div className="flex gap-2 mb-3">
          <button
            onClick={() => setUseImages(!useImages)}
            className={`flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all ${
              useImages ? "bg-[#d4a574]/20 text-[#d4a574] border border-[#d4a574]/30" : "bg-white/5 text-white/40 border border-white/10"
            }`}
          >
            {useImages ? "Images" : "Colors"}
          </button>
          <button
            onClick={() => setShowSplit(!showSplit)}
            className={`flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all ${
              showSplit ? "bg-[#d4a574]/20 text-[#d4a574] border border-[#d4a574]/30" : "bg-white/5 text-white/40 border border-white/10"
            }`}
          >
            {showSplit ? "Split Shown" : "Show Split"}
          </button>
        </div>

        {/* Legend */}
        <div className="border-t border-white/5 pt-3">
          <label className="text-sm text-white/55 font-semibold uppercase tracking-widest block mb-2">Concepts</label>
          <div className="grid grid-cols-2 gap-y-1.5 gap-x-3">
            {LEGEND.map((item) => (
              <div key={item.label} className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: item.color, boxShadow: `0 0 6px ${item.color}60` }} />
                <span className="text-sm text-white/70">{item.label}</span>
              </div>
            ))}
          </div>
          {showSplit && (
            <div className="mt-2 pt-2 border-t border-white/5">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2.5 h-2.5 rounded-full bg-white/80" />
                <span className="text-xs text-white/50">Test set (white ring)</span>
              </div>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="border-t border-white/5 pt-3 mt-3">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-lg font-bold text-[#3B82F6]">{animalCount}</div>
              <div className="text-xs text-white/50">Animal</div>
            </div>
            <div>
              <div className="text-lg font-bold text-[#EF4444]">{mythCount}</div>
              <div className="text-xs text-white/50">Myth</div>
            </div>
            <div>
              <div className="text-lg font-bold text-[#22C55E]">{treeCount}</div>
              <div className="text-xs text-white/50">Tree</div>
            </div>
          </div>
          <div className="flex justify-center gap-6 mt-2">
            <div className="text-center">
              <span className="text-lg font-bold text-[#d4a574]">{trainCount}</span>
              <span className="text-xs text-white/50 ml-1">Train</span>
            </div>
            <div className="text-center">
              <span className="text-lg font-bold text-[#d4a574]">{testCount}</span>
              <span className="text-xs text-white/50 ml-1">Test</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Hovered node */}
      <AnimatePresence>
        {hoveredNode && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="fixed right-2 md:right-4 top-16 md:top-20 z-40 w-48 md:w-64 hidden sm:block bg-[#0a0a12]/90 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl"
          >
            <div className="relative">
              <img src={hoveredNode.image_url} alt="" className="w-full h-44 object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-[#0a0a12] via-transparent to-transparent" />
              <div className="absolute top-2 right-2 bg-white/10 backdrop-blur-sm px-2 py-0.5 rounded-full text-[9px] text-white/60">
                {hoveredNode.split}
              </div>
            </div>
            <div className="p-3 -mt-4 relative">
              <h4 className="text-base font-bold text-white mb-1">{panelName(hoveredNode.panel_id)}</h4>
              <p className="text-xs text-white/40 mb-2">{getNodeLabel(hoveredNode)}</p>
              <div className="flex gap-1">
                {hoveredNode.animal === 1 && <span className="text-xs px-1.5 py-0.5 rounded font-medium" style={{ backgroundColor: "#3B82F620", color: "#3B82F6" }}>Animal</span>}
                {hoveredNode.mythology === 1 && <span className="text-xs px-1.5 py-0.5 rounded font-medium" style={{ backgroundColor: "#EF444420", color: "#EF4444" }}>Myth</span>}
                {hoveredNode.tree === 1 && <span className="text-xs px-1.5 py-0.5 rounded font-medium" style={{ backgroundColor: "#22C55E20", color: "#22C55E" }}>Tree</span>}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Transition overlay */}
      <AnimatePresence>
        {transitioning && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-30 bg-[#020208]/60 flex items-center justify-center pointer-events-none"
          >
            <div className="text-center">
              <div className="w-12 h-12 border-2 border-[#d4a574]/30 border-t-[#d4a574] rounded-full animate-spin mx-auto mb-3" />
              <p className="text-white/40 text-sm">Reshaping embedding space...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 3D Visualization */}
      <div className="w-full h-[100dvh]">
        {nodes.length > 0 ? (
          <ForceGraph3D
            ref={fgRef}
            graphData={graphData}
            nodeId="id"
            width={windowSize.width}
            height={windowSize.height}
            backgroundColor="rgba(2,2,8,0)"
            nodeRelSize={4}
            nodeVal={4}
            nodeColor={(node: any) => getNodeColor(node as EmbNode)}
            nodeOpacity={useImages ? 1 : 0.9}
            nodeResolution={24}
            nodeThreeObject={useImages ? nodeThreeObject : undefined}
            nodeThreeObjectExtend={false}
            nodeLabel={(node: any) => {
              const n = node as EmbNode;
              return `<div style="background:rgba(10,10,18,0.9);padding:6px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);font-size:11px;color:white">
                <div style="font-weight:bold">${n.panel_id}</div>
                <div style="color:rgba(255,255,255,0.4);font-size:9px">${getNodeLabel(n)} · ${n.split}</div>
              </div>`;
            }}
            linkWidth={0}
            enableNodeDrag={false}
            enableNavigationControls={true}
            showNavInfo={false}
            onNodeHover={(node: any) => setHoveredNode(node as EmbNode || null)}
            onNodeClick={() => {}}
            cooldownTicks={0}
            d3AlphaDecay={1}
            d3VelocityDecay={1}
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center">
              <div className="w-16 h-16 border-2 border-[#d4a574]/30 border-t-[#d4a574] rounded-full animate-spin mx-auto mb-4" />
              <p className="text-white/30 text-sm">Loading embedding space...</p>
            </motion.div>
          </div>
        )}
      </div>
    </main>
  );
}
