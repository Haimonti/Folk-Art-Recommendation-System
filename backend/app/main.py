"""
GeMi Demo — FastAPI Backend
"""

import os, json, uuid
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Config ──
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
EMB_DIR  = DATA_DIR / "embeddings"
META_DIR = DATA_DIR / "metadata"
IMG_DIR  = DATA_DIR / "images" / "panels"

AVAILABLE_MODELS = [
    "llamasigclip_gcn", "llamasigclip_gae", "llamasigclip_vgae",
    "llamavae_gcn", "llamavae_gae", "llamavae_vgae",
    "llamasigclip_ind_gcn", "llamasigclip_ind_gae", "llamasigclip_ind_vgae",
    "llamavae_ind_gcn", "llamavae_ind_gae", "llamavae_ind_vgae",
]

# ── Load data at startup ──
def load_all():
    panels_path = META_DIR / "panels.json"
    if not panels_path.exists():
        print(f"⚠ panels.json not found at {panels_path}")
        print("  Run the Colab export notebook first!")
        print("  Starting in DEMO MODE with empty data.\n")
        return [], {"n_train": 0, "n_test": 0, "n_total": 0}, {}, {}

    with open(panels_path) as f:
        panels = json.load(f)

    with open(META_DIR / "split_info.json") as f:
        split_info = json.load(f)

    embeddings = {}
    for model_name in AVAILABLE_MODELS:
        npy_path = EMB_DIR / f"{model_name}.npy"
        if npy_path.exists():
            embeddings[model_name] = np.load(str(npy_path))
            print(f"  ✓ {model_name}: {embeddings[model_name].shape}")

    graph_edges = {}
    for f in META_DIR.glob("graph_edges_*.json"):
        model_key = f.stem.replace("graph_edges_", "")
        with open(f) as fh:
            graph_edges[model_key] = json.load(fh)
        print(f"  ✓ graph edges: {model_key}")

    return panels, split_info, embeddings, graph_edges


print("Loading GeMi demo data...")
PANELS, SPLIT_INFO, EMBEDDINGS, GRAPH_EDGES = load_all()
print(f"Ready: {len(PANELS)} panels, {len(EMBEDDINGS)} models.\n")

SESSIONS: dict = {}

# ── FastAPI App ──
app = FastAPI(
    title="GeMi Demo API",
    description="Graph-based Multimodal Recommendation for Narrative Scroll Paintings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(os.getenv("CORS_ORIGINS", '["http://localhost:3000"]')),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if IMG_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMG_DIR)), name="images")


# ── Pydantic Models ──
class UserSession(BaseModel):
    liked_panel_indices: list[int] = Field(..., min_length=1)
    concept_preferences: dict[str, bool] = Field(
        default={"animal": True, "mythology": True, "tree": True}
    )
    description: Optional[str] = None

class RecommendRequest(BaseModel):
    session_id: str
    model_name: str = "llamasigclip_vgae"
    top_k: int = Field(5, ge=1, le=20)

class CompareRequest(BaseModel):
    session_id: str
    models: list[str] = ["llamasigclip_gcn", "llamasigclip_gae", "llamasigclip_vgae"]
    top_k: int = 5


# ── Core recommendation logic ──
def get_recommendations(liked_indices, model_name, top_k=5, concept_prefs=None):
    if model_name not in EMBEDDINGS:
        raise ValueError(f"Model '{model_name}' not loaded.")

    emb = EMBEDDINGS[model_name]
    user_emb = emb[liked_indices].mean(axis=0, keepdims=True)
    sims = cosine_similarity(user_emb, emb)[0]

    for idx in liked_indices:
        sims[idx] = -1.0

    top_indices = np.argsort(sims)[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_indices):
        panel = PANELS[idx].copy()
        panel["rank"] = rank + 1
        panel["similarity_score"] = round(float(sims[idx]), 4)
        panel["image_url"] = f"/images/{panel['image_filename']}"

        # Explainability: per-liked-panel breakdown
        breakdown = []
        for liked_idx in liked_indices:
            s = float(cosine_similarity(emb[liked_idx:liked_idx+1], emb[idx:idx+1])[0][0])
            breakdown.append({
                "liked_panel_index": liked_idx,
                "liked_panel_id": PANELS[liked_idx]["id"],
                "similarity": round(s, 4),
            })
        breakdown.sort(key=lambda x: x["similarity"], reverse=True)
        panel["explanation"] = {
            "model_used": model_name,
            "per_panel_similarity": breakdown[:5],
        }

        if concept_prefs:
            label_map = {"animal": "animal_label", "mythology": "myth_label", "tree": "tree_label"}
            panel["concept_matches"] = [
                c for c, wanted in concept_prefs.items()
                if wanted and panel.get(label_map.get(c, ""), 0) == 1
            ]

        results.append(panel)
    return results


# ── Endpoints ──
@app.get("/")
def root():
    return {
        "app": "GeMi Demo API",
        "status": "running",
        "panels_loaded": len(PANELS),
        "models_loaded": list(EMBEDDINGS.keys()),
    }

@app.get("/api/panels")
def list_panels(split: Optional[str] = None, limit: int = 200):
    panels = PANELS if not split else [p for p in PANELS if p["split"] == split]
    for p in panels[:limit]:
        p["image_url"] = f"/images/{p['image_filename']}"
    return {"panels": panels[:limit], "total": len(panels)}

@app.get("/api/panels/{panel_index}")
def get_panel(panel_index: int):
    if panel_index < 0 or panel_index >= len(PANELS):
        raise HTTPException(404, f"Panel {panel_index} not found")
    panel = PANELS[panel_index].copy()
    panel["image_url"] = f"/images/{panel['image_filename']}"
    return panel

@app.get("/api/models")
def list_models():
    info = {
        "llamasigclip_gcn":  {"feature": "LlamaSigCLIP", "gnn": "GCN",  "type": "Supervised"},
        "llamasigclip_gae":  {"feature": "LlamaSigCLIP", "gnn": "GAE",  "type": "Unsupervised"},
        "llamasigclip_vgae": {"feature": "LlamaSigCLIP", "gnn": "VGAE", "type": "Semi-supervised"},
        "llamavae_gcn":      {"feature": "LlamaVAE",     "gnn": "GCN",  "type": "Supervised"},
        "llamavae_gae":      {"feature": "LlamaVAE",     "gnn": "GAE",  "type": "Unsupervised"},
        "llamavae_vgae":     {"feature": "LlamaVAE",     "gnn": "VGAE", "type": "Semi-supervised"},
        "llamasigclip_base": {"feature": "LlamaSigCLIP", "gnn": "None", "type": "Baseline"},
        "llamavae_base":     {"feature": "LlamaVAE",     "gnn": "None", "type": "Baseline"},
    }
    return {"models": {k: v for k, v in info.items() if k in EMBEDDINGS}}

@app.post("/api/session/create")
def create_session(user: UserSession):
    for idx in user.liked_panel_indices:
        if idx < 0 or idx >= len(PANELS):
            raise HTTPException(400, f"Invalid panel index: {idx}")
    session_id = str(uuid.uuid4())[:8]
    SESSIONS[session_id] = {
        "liked_panel_indices": user.liked_panel_indices,
        "concept_preferences": user.concept_preferences,
        "description": user.description,
    }
    return {"session_id": session_id, "liked_count": len(user.liked_panel_indices)}

@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    session = SESSIONS[req.session_id]
    try:
        results = get_recommendations(
            session["liked_panel_indices"], req.model_name,
            req.top_k, session.get("concept_preferences"),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"session_id": req.session_id, "model": req.model_name, "recommendations": results}

@app.post("/api/compare")
def compare_models(req: CompareRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    session = SESSIONS[req.session_id]
    comparison = {}
    for m in req.models:
        if m not in EMBEDDINGS:
            comparison[m] = {"error": f"Not loaded"}
            continue
        comparison[m] = get_recommendations(
            session["liked_panel_indices"], m, req.top_k, session.get("concept_preferences"),
        )
    return {"session_id": req.session_id, "comparison": comparison}

@app.get("/api/graph/{model_name}")
def get_graph_data(model_name: str, panel_index: int = 0, depth: int = 2):
    if model_name not in GRAPH_EDGES:
        raise HTTPException(404, f"No graph data for '{model_name}'. Available: {list(GRAPH_EDGES.keys())}")
    edges = GRAPH_EDGES[model_name]
    neighbor_set = {panel_index}
    frontier = {panel_index}
    for _ in range(depth):
        new_frontier = set()
        for edge in edges:
            if edge["source"] in frontier:
                new_frontier.add(edge["target"])
            if edge["target"] in frontier:
                new_frontier.add(edge["source"])
        frontier = new_frontier - neighbor_set
        neighbor_set |= frontier
    filtered = [e for e in edges if e["source"] in neighbor_set and e["target"] in neighbor_set]
    nodes = []
    for idx in neighbor_set:
        if idx < len(PANELS):
            nodes.append({
                "id": idx,
                "panel_id": PANELS[idx]["id"],
                "image_url": f"/images/{PANELS[idx]['image_filename']}",
                "is_center": idx == panel_index,
                "animal": PANELS[idx]["animal_label"],
                "mythology": PANELS[idx]["myth_label"],
                "tree": PANELS[idx]["tree_label"],
            })
    return {"nodes": nodes, "edges": filtered}


@app.get("/api/tsne/{model_name}")
def get_tsne(model_name: str):
    tsne_path = META_DIR / "tsne_3d_all_models.json"
    if not tsne_path.exists():
        raise HTTPException(404, "t-SNE data not found")
    with open(tsne_path) as f:
        all_tsne = json.load(f)
    if model_name not in all_tsne:
        raise HTTPException(404, f"No t-SNE for '{model_name}'. Available: {list(all_tsne.keys())}")
    coords = all_tsne[model_name]
    nodes = []
    for i, c in enumerate(coords):
        if i < len(PANELS):
            nodes.append({
                "id": i,
                "panel_id": PANELS[i]["id"],
                "image_url": f"/images/{PANELS[i]['image_filename']}",
                "animal": PANELS[i]["animal_label"],
                "mythology": PANELS[i]["myth_label"],
                "tree": PANELS[i]["tree_label"],
                "split": PANELS[i]["split"],
                "x": c[0], "y": c[1], "z": c[2],
            })
    return {"model": model_name, "nodes": nodes}


@app.get("/api/scrolls")
def list_scrolls():
    scroll_map = {}
    for p in PANELS:
        sid = p["scroll_id"]
        if sid not in scroll_map:
            scroll_map[sid] = {"scroll_id": sid, "panel_count": 0, "preview_image": None}
        scroll_map[sid]["panel_count"] += 1
        if scroll_map[sid]["preview_image"] is None:
            scroll_map[sid]["preview_image"] = f"/images/{p['image_filename']}"
    return {"scrolls": sorted(scroll_map.values(), key=lambda x: x["scroll_id"])}


@app.get("/api/scrolls/{scroll_id}")
def get_scroll_panels(scroll_id: str):
    panels_in_scroll = [p.copy() for p in PANELS if p["scroll_id"] == scroll_id]
    if not panels_in_scroll:
        raise HTTPException(404, f"Scroll '{scroll_id}' not found")
    panels_in_scroll.sort(key=lambda p: int(p["panel_id"]) if p["panel_id"].isdigit() else 0)
    for p in panels_in_scroll:
        p["image_url"] = f"/images/{p['image_filename']}"
    return {"scroll_id": scroll_id, "panel_count": len(panels_in_scroll), "panels": panels_in_scroll}


@app.get("/api/stats")
def dataset_stats():
    return {
        "total_panels": len(PANELS),
        "train_panels": sum(1 for p in PANELS if p["split"] == "train"),
        "test_panels": sum(1 for p in PANELS if p["split"] == "test"),
        "scrolls": len(set(p["scroll_id"] for p in PANELS)),
        "label_counts": {
            "animal": sum(1 for p in PANELS if p["animal_label"] == 1),
            "mythology": sum(1 for p in PANELS if p["myth_label"] == 1),
            "tree": sum(1 for p in PANELS if p["tree_label"] == 1),
        },
        "models_available": len(EMBEDDINGS),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)