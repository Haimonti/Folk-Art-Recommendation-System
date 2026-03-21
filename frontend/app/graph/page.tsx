"use client";

import { useState, useEffect, useRef, useCallback, Suspense } from "react";

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
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";

const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), { ssr: false });

interface GraphNode {
  id: number;
  panel_id: string;
  image_url: string;
  is_center: boolean;
  animal: number;
  mythology: number;
  tree: number;
  x?: number; y?: number; z?: number;
}

interface GraphEdge {
  source: any;
  target: any;
  weight: number;
}

function getNodeColor(node: GraphNode): string {
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

function getNodeLabel(node: GraphNode): string {
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

function GraphPageInner() {
  const searchParams = useSearchParams();
  const urlModel = searchParams.get("model");
  const urlPanel = searchParams.get("panel");
  const urlDepth = searchParams.get("depth");

  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphEdge[] } | null>(null);
  const [selectedModel, setSelectedModel] = useState(urlModel || "llamasigclip_vgae");
  const [centerPanel, setCenterPanel] = useState(urlPanel ? parseInt(urlPanel) : 0);
  const [depth, setDepth] = useState(urlDepth ? parseInt(urlDepth) : 2);
  const [loading, setLoading] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [panels, setPanels] = useState<any[]>([]);
  const [useImageNodes, setUseImageNodes] = useState(true);
  const [imageCache, setImageCache] = useState<Record<string, any>>({});
  const fgRef = useRef<any>(null);
  const windowSize = useWindowSize();
  const [showControls, setShowControls] = useState(false);

  useEffect(() => {
    fetch("/api/panels?limit=200")
      .then((r) => r.json())
      .then((data) => setPanels(data.panels || []))
      .catch(console.error);
  }, []);

  // Preload images for node textures
  useEffect(() => {
    if (typeof window === "undefined" || !graphData) return;
    const THREE = require("three");
    const loader = new THREE.TextureLoader();
    graphData.nodes.forEach((n) => {
      if (!imageCache[n.image_url]) {
        loader.load(n.image_url, (texture: any) => {
          setImageCache((prev) => ({ ...prev, [n.image_url]: texture }));
        });
      }
    });
  }, [graphData]);

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/graph/${selectedModel}?panel_index=${centerPanel}&depth=${depth}`);
      const data = await res.json();
      if (data.nodes && data.edges) {
        setGraphData({ nodes: data.nodes, links: data.edges });
        setTimeout(() => {
          if (fgRef.current) fgRef.current.zoomToFit(1000, 80);
        }, 800);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [selectedModel, centerPanel, depth]);

  useEffect(() => {
    if (panels.length > 0) fetchGraph();
  }, [panels.length, fetchGraph]);

  // Auto-resize and center on window resize
  useEffect(() => {
    const handleResize = () => {
      if (fgRef.current) {
        fgRef.current.zoomToFit(600, 40);
      }
    };
    window.addEventListener("resize", handleResize);
    // Also center on initial load after a delay
    const timer = setTimeout(() => {
      if (fgRef.current) fgRef.current.zoomToFit(800, 40);
    }, 2000);
    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(timer);
    };
  }, [graphData]);

  // Node renderer with image textures
  const nodeThreeObject = useCallback((node: any) => {
    if (typeof window === "undefined") return undefined;
    const THREE = require("three");
    const n = node as GraphNode;

    if (useImageNodes && imageCache[n.image_url]) {
      const size = n.is_center ? 12 : 6;
      const geometry = new THREE.SphereGeometry(size, 32, 32);
      const material = new THREE.MeshBasicMaterial({
        map: imageCache[n.image_url],
        transparent: true,
        opacity: 0.95,
      });
      const mesh = new THREE.Mesh(geometry, material);

      // Glow ring around node
      const glowColor = new THREE.Color(getNodeColor(n));
      const ringGeo = new THREE.RingGeometry(size + 1, size + 2.5, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: glowColor,
        transparent: true,
        opacity: n.is_center ? 0.9 : 0.4,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      mesh.add(ring);

      // Outer soft glow for center node
      if (n.is_center) {
        const outerGeo = new THREE.RingGeometry(size + 2.5, size + 5, 32);
        const outerMat = new THREE.MeshBasicMaterial({
          color: glowColor,
          transparent: true,
          opacity: 0.15,
          side: THREE.DoubleSide,
        });
        mesh.add(new THREE.Mesh(outerGeo, outerMat));
      }

      return mesh;
    }
    return undefined;
  }, [useImageNodes, imageCache]);

  return (
    <main className="min-h-screen bg-[#020208] relative overflow-hidden">
      {/* Ambient glow */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-1/3 left-1/3 w-[600px] h-[600px] bg-[#3B82F6]/5 rounded-full blur-[150px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-[#EF4444]/5 rounded-full blur-[150px]" />
        <div className="absolute top-1/2 right-1/3 w-[400px] h-[400px] bg-[#22C55E]/5 rounded-full blur-[150px]" />
      </div>

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-[#020208]/60 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-3 md:px-6 py-2 md:py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <a href="/" className="text-xl md:text-2xl font-bold text-[#d4a574]">GeMi</a>
            <span className="text-sm text-white/20 hidden sm:inline">/</span>
            <span className="text-xs md:text-sm text-white/50 font-medium hidden sm:inline">Graph Explorer</span>
          </div>
          <a href="/story" className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white/60 hover:text-white hover:border-white/20 transition-all">
            Scroll Stories
          </a>
          <a href="/embeddings" className="px-2 md:px-4 py-1.5 md:py-2 bg-white/5 border border-white/10 rounded-lg text-xs md:text-sm text-white/60 hover:text-white hover:border-white/20 transition-all">
            Embedding Space
          </a>
          <a href="/explore" className="px-2 md:px-4 py-1.5 md:py-2 bg-white/5 border border-white/10 rounded-lg text-xs md:text-sm text-white/60 hover:text-white hover:border-white/20 transition-all">
            Back to Explore
          </a>
        </div>
      </header>

      {/* Mobile toggle button */}
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
        className="fixed z-40 bg-[#0a0a12]/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl shadow-black/50 overflow-y-auto left-2 right-2 bottom-14 max-h-[60vh] p-3 md:left-4 md:right-auto md:bottom-auto md:top-20 md:w-72 md:max-h-[calc(100vh-6rem)] md:p-4"
      >
        <h3 className="text-base md:text-lg font-bold text-[#d4a574] mb-2 md:mb-3">Graph Controls</h3>

        {/* Model */}
        <div className="mb-3">
          <label className="text-[10px] text-white/30 uppercase tracking-widest block mb-2">Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-[#d4a574]/50"
          >
            {MODELS.map((m) => (
              <option key={m.key} value={m.key} className="bg-[#0a0a12]">{m.label}</option>
            ))}
          </select>
        </div>

        {/* Center panel */}
        <div className="mb-3">
          <label className="text-[10px] text-white/30 uppercase tracking-widest block mb-2">Center Panel</label>
          <input
            type="range"
            min={0}
            max={panels.length - 1}
            value={centerPanel}
            onChange={(e) => setCenterPanel(parseInt(e.target.value))}
            className="w-full accent-[#d4a574] h-1.5"
          />
          <div className="flex justify-between text-xs text-white/30 mt-1.5">
            <span>Panel {centerPanel}</span>
            <span className="text-[#d4a574]">{panels[centerPanel]?.id || ""}</span>
          </div>
        </div>

        {/* Depth */}
        <div className="mb-3">
          <label className="text-[10px] text-white/30 uppercase tracking-widest block mb-2">Neighborhood Depth</label>
          <div className="flex gap-2">
            {[1, 2, 3].map((d) => (
              <button
                key={d}
                onClick={() => setDepth(d)}
                className={`flex-1 py-2 rounded-xl text-xs font-medium transition-all ${
                  depth === d
                    ? "bg-[#d4a574] text-[#0a0a0a] shadow-lg shadow-[#d4a574]/20"
                    : "bg-white/5 text-white/40 border border-white/10"
                }`}
              >
                {d}-hop
              </button>
            ))}
          </div>
        </div>

        {/* Toggle */}
        <div className="mb-5">
          <button
            onClick={() => setUseImageNodes(!useImageNodes)}
            className={`w-full py-2 rounded-xl text-xs font-medium transition-all ${
              useImageNodes
                ? "bg-[#d4a574]/20 text-[#d4a574] border border-[#d4a574]/30"
                : "bg-white/5 text-white/40 border border-white/10"
            }`}
          >
            {useImageNodes ? "Image Nodes" : "Color Nodes"}
          </button>
        </div>

        {/* Explore */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={fetchGraph}
          disabled={loading}
          className="w-full py-3 bg-gradient-to-r from-[#d4a574] to-[#e8c49a] text-[#0a0a0a] font-bold rounded-xl hover:shadow-lg hover:shadow-[#d4a574]/20 transition-all disabled:opacity-50 text-sm mb-5"
        >
          {loading ? "Building Graph..." : "Explore Neighborhood"}
        </motion.button>

        {/* Legend */}
        <div className="border-t border-white/5 pt-4">
          <label className="text-[10px] text-white/30 uppercase tracking-widest block mb-2">Legend</label>
          <div className="grid grid-cols-2 gap-y-2 gap-x-3">
            {LEGEND.map((item) => (
              <div key={item.label} className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: item.color, boxShadow: `0 0 8px ${item.color}60` }} />
                <span className="text-[11px] text-white/50">{item.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Edge Strength + Stats */}
        <div className="border-t border-white/5 pt-3 mt-3">
          <label className="text-[10px] text-white/30 uppercase tracking-widest block mb-2">Edge Strength</label>
          <div className="flex justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <div className="w-5 h-0.5 rounded-full" style={{ backgroundColor: "#00FFFF", boxShadow: "0 0 6px #00FFFF" }} />
              <span className="text-[10px] text-white/40">&gt;99.9%</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-5 h-0.5 rounded-full" style={{ backgroundColor: "#FF6BFF", boxShadow: "0 0 6px #FF6BFF" }} />
              <span className="text-[10px] text-white/40">&gt;99.7%</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-5 h-0.5 rounded-full" style={{ backgroundColor: "#FFAA44", boxShadow: "0 0 6px #FFAA44" }} />
              <span className="text-[10px] text-white/40">&lt;99.7%</span>
            </div>
          </div>
        </div>
        {graphData && (
          <div className="flex justify-center gap-8 pt-3 mt-2">
            <div className="text-center">
              <span className="text-base font-bold text-[#d4a574]">{graphData.nodes.length}</span>
              <span className="text-[10px] text-white/30 uppercase ml-1.5">Nodes</span>
            </div>
            <div className="text-center">
              <span className="text-base font-bold text-[#d4a574]">{graphData.links.length}</span>
              <span className="text-[10px] text-white/30 uppercase ml-1.5">Edges</span>
            </div>
          </div>
        )}
      </motion.div>

      {/* Hovered node */}
      <AnimatePresence>
        {hoveredNode && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="fixed right-2 md:right-4 top-16 md:top-20 z-40 w-48 md:w-72 hidden sm:block bg-[#0a0a12]/90 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl"
          >
            <div className="relative">
              <img src={hoveredNode.image_url} alt="" className="w-full h-52 object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-[#0a0a12] via-transparent to-transparent" />
              {hoveredNode.is_center && (
                <div className="absolute top-3 right-3 bg-[#d4a574] text-[#0a0a0a] px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider">Center</div>
              )}
            </div>
            <div className="p-4 -mt-6 relative">
              <h4 className="text-base font-bold text-white mb-1">Panel {hoveredNode.panel_id}</h4>
              <p className="text-xs text-white/40 mb-3">{getNodeLabel(hoveredNode)}</p>
              <div className="flex gap-1.5">
                {hoveredNode.animal === 1 && <span className="text-xs px-2 py-1 rounded-lg font-medium" style={{ backgroundColor: "#3B82F620", color: "#3B82F6" }}>Animal</span>}
                {hoveredNode.mythology === 1 && <span className="text-xs px-2 py-1 rounded-lg font-medium" style={{ backgroundColor: "#EF444420", color: "#EF4444" }}>Myth</span>}
                {hoveredNode.tree === 1 && <span className="text-xs px-2 py-1 rounded-lg font-medium" style={{ backgroundColor: "#22C55E20", color: "#22C55E" }}>Tree</span>}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 3D Graph */}
      <div className="w-full h-[100dvh]">
        {graphData ? (
          <ForceGraph3D
            ref={fgRef}
            graphData={graphData}
            nodeId="id"
            width={windowSize.width}
            height={windowSize.height}
            backgroundColor="rgba(2,2,8,0)"
            nodeRelSize={5}
            nodeVal={(node: any) => node.is_center ? 18 : 5}
            nodeColor={(node: any) => getNodeColor(node as GraphNode)}
            nodeOpacity={useImageNodes ? 1 : 0.95}
            nodeResolution={32}
            nodeThreeObject={useImageNodes ? nodeThreeObject : undefined}
            nodeThreeObjectExtend={false}
            nodeLabel={(node: any) => {
              const n = node as GraphNode;
              return `<div style="background:rgba(10,10,18,0.9);padding:8px 12px;border-radius:10px;border:1px solid rgba(255,255,255,0.1);font-size:12px;color:white;backdrop-filter:blur(10px)">
                <div style="font-weight:bold;margin-bottom:2px">Panel ${n.panel_id}</div>
                <div style="color:rgba(255,255,255,0.5);font-size:10px">${getNodeLabel(n)}</div>
              </div>`;
            }}
            linkWidth={(link: any) => {
              const w = link.weight || 0;
              return Math.max(0.4, (w - 0.98) * 40);
            }}
            linkOpacity={0.25}
            linkColor={(link: any) => {
              const w = link.weight || 0;
              if (w > 0.999) return "#d4a574";
              if (w > 0.997) return "#b08050";
              if (w > 0.995) return "#806040";
              return "#403020";
            }}
            linkDirectionalParticles={3}
            linkDirectionalParticleWidth={(link: any) => {
              const w = link.weight || 0;
              return w > 0.997 ? 1.2 : 0.7;
            }}
            linkDirectionalParticleSpeed={(link: any) => {
              const w = link.weight || 0;
              return 0.002 + (w - 0.98) * 0.15;
            }}
            linkDirectionalParticleColor={(link: any) => {
              const w = link.weight || 0;
              if (w > 0.999) return "#00FFFF";
              if (w > 0.997) return "#FF6BFF";
              return "#FFAA44";
            }}
            linkCurvature={0.15}
            linkCurveRotation={Math.PI * 0.5}
            enableNodeDrag={true}
            enableNavigationControls={true}
            showNavInfo={false}
            onNodeHover={(node: any) => {
              setHoveredNode(node as GraphNode || null);
              if (fgRef.current) {
                const el = fgRef.current.renderer().domElement;
                el.style.cursor = node ? "pointer" : "grab";
              }
            }}
            onNodeClick={() => {}}
            warmupTicks={100}
            cooldownTicks={300}
            d3AlphaDecay={0.012}
            d3VelocityDecay={0.2}
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center">
              <div className="w-16 h-16 border-2 border-[#d4a574]/30 border-t-[#d4a574] rounded-full animate-spin mx-auto mb-4" />
              <p className="text-white/30 text-sm">Building graph neighborhood...</p>
            </motion.div>
          </div>
        )}
      </div>
    </main>
  );
}

export default function GraphPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#020208] flex items-center justify-center"><div className="w-16 h-16 border-2 border-[#d4a574]/30 border-t-[#d4a574] rounded-full animate-spin" /></div>}>
      <GraphPageInner />
    </Suspense>
  );
}
