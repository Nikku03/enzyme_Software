from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    Draw = None

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, torch
from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol
from enzyme_software.liquid_nn_v2.utils.mol_provenance import mol_provenance_context


def plot_molecule_with_predictions(smiles: str, site_scores, outfile: Optional[str] = None):
    if Chem is None or Draw is None:
        raise RuntimeError("RDKit drawing support is required for visualization")
    with mol_provenance_context(module_triggered="visualization", source_category="visualization", parsed_smiles=smiles):
        prep = prepare_mol(smiles)
    if prep.mol is None:
        raise ValueError(f"Invalid SMILES: {smiles} ({prep.error})")
    mol = prep.mol
    if hasattr(site_scores, "detach"):
        values = site_scores.detach().cpu().numpy().reshape(-1)
    else:
        values = list(site_scores)
    ranked = sorted(range(len(values)), key=lambda idx: float(values[idx]), reverse=True)
    top_atoms = ranked[: min(5, len(ranked))]
    image = Draw.MolToImage(mol, highlightAtoms=top_atoms)
    if outfile:
        image.save(outfile)
    return image


def _count_parameters(model) -> int:
    if not TORCH_AVAILABLE:
        return 0
    total = 0
    for param in model.parameters():
        try:
            total += int(param.numel())
        except Exception:
            continue
    return total


def _average_abs_weight(model) -> float:
    if not TORCH_AVAILABLE:
        return 0.0
    values = []
    for _, param in model.named_parameters():
        try:
            if param.ndim >= 2:
                values.append(float(param.detach().abs().mean().item()))
        except Exception:
            continue
    return float(sum(values) / len(values)) if values else 0.0


def _layer_specs(config: ModelConfig) -> List[Dict[str, object]]:
    group_count = 16
    site_hidden = config.som_head_hidden_dim
    cyp_hidden = config.cyp_head_hidden_dim
    shared_hidden = config.shared_hidden_dim or config.hidden_dim
    layers = [
        {"id": "input", "label": "Atom Input", "neurons": config.atom_input_dim, "kind": "input", "connectivity": "feature vector", "details": "140 per-atom chemistry features"},
        {"id": "physics_raw", "label": "Physics Raw", "neurons": 32, "kind": "physics", "connectivity": "deterministic", "details": "BDE, bond class, electronegativity, aromaticity, group membership"},
        {"id": "manual_prior", "label": "Manual Prior Encoder", "neurons": config.som_branch_dim + config.cyp_branch_dim, "kind": "physics", "connectivity": "dense", "details": "Manual chemistry engine priors and residual scaffold"},
        {"id": "steric_3d", "label": "Steric 3D Branch", "neurons": config.steric_hidden_dim if config.use_3d_branch else 0, "kind": "physics", "connectivity": "optional dense", "details": "Optional lightweight atom exposure / steric descriptors"},
        {"id": "physics_proj", "label": "Physics Projection", "neurons": config.physics_dim, "kind": "physics", "connectivity": "dense", "details": f"32 -> {config.physics_dim} linear projection"},
        {"id": "shared_input", "label": "Shared Encoder Input", "neurons": shared_hidden, "kind": "liquid", "connectivity": "dense", "details": f"{config.atom_input_dim} -> {shared_hidden} input projection"},
        {"id": "shared_liquid", "label": "Shared Contextual LTC", "neurons": shared_hidden, "kind": "liquid", "connectivity": "edge-aware graph + ODE", "details": f"{config.shared_encoder_layers} shared layer(s), {config.ode_steps} RK4 steps"},
        {"id": "fusion_gate", "label": "Physics Fusion Gate", "neurons": shared_hidden, "kind": "fusion", "connectivity": "branched dense", "details": f"Physics {config.physics_dim} + shared {shared_hidden} -> {shared_hidden}"},
        {"id": "som_branch", "label": "SoM Branch", "neurons": config.som_branch_dim, "kind": "liquid", "connectivity": "competition-aware", "details": f"{config.som_branch_layers} branch layer(s) + site competition refinement"},
        {"id": "cyp_branch", "label": "CYP Branch", "neurons": config.cyp_branch_dim, "kind": "liquid", "connectivity": "hierarchical context", "details": f"{config.cyp_branch_layers} branch layer(s) + global context"},
        {"id": "group_pool", "label": "Chemistry Groups", "neurons": group_count * config.group_pooling_hidden_dim, "kind": "pool", "connectivity": "hierarchical attention", "details": f"{group_count} group embeddings with attention pooling"},
        {"id": "mol_pool", "label": "Molecule Context", "neurons": config.cyp_branch_dim, "kind": "pool", "connectivity": "group -> molecule", "details": "Chemistry-aware hierarchical molecule representation"},
        {"id": "site_hidden", "label": "SoM Residual Head", "neurons": site_hidden, "kind": "head", "connectivity": "prior + residual", "details": f"Per-atom residual fusion ({config.som_branch_dim} -> {site_hidden})"},
        {"id": "site_out", "label": "Site Output", "neurons": 1, "kind": "output", "connectivity": "dense", "details": "Per-atom site logit"},
        {"id": "cyp_hidden", "label": "CYP Residual Head", "neurons": cyp_hidden, "kind": "head", "connectivity": "prior + residual", "details": f"Molecule residual fusion ({config.cyp_branch_dim} -> {cyp_hidden})"},
        {"id": "cyp_out", "label": "CYP Output", "neurons": config.num_cyp_classes, "kind": "output", "connectivity": "dense", "details": "Isoform logits"},
    ]
    if config.model_variant == "advanced":
        advanced_layers = []
        if config.use_phase_augmented_state:
            advanced_layers.append({"id": "phase_state", "label": "Phase State", "neurons": shared_hidden, "kind": "liquid", "connectivity": "phase modulation", "details": "Amplitude + phase-augmented hidden state"})
        if config.use_physics_residual:
            advanced_layers.append({"id": "physics_residual", "label": "Physics Residual", "neurons": config.som_branch_dim + config.cyp_branch_dim, "kind": "fusion", "connectivity": "gated residual", "details": "Compact physics-guided residual fusion"})
        if config.use_energy_module:
            advanced_layers.append({"id": "energy_module", "label": "Energy Landscape", "neurons": config.energy_hidden_dim, "kind": "physics", "connectivity": "auxiliary / dynamics", "details": "Node/group/molecule energy heads"})
        if config.use_tunneling_module:
            advanced_layers.append({"id": "tunneling_module", "label": "Barrier Tunneling", "neurons": config.tunneling_hidden_dim, "kind": "physics", "connectivity": "barrier -> exp(-alpha*barrier)", "details": "Barrier estimation with tunneling probabilities"})
        if config.use_graph_tunneling:
            advanced_layers.append({"id": "graph_tunneling", "label": "Graph Tunneling", "neurons": config.graph_tunneling_dim, "kind": "liquid", "connectivity": "sparse non-local", "details": f"Top-k sparse tunneling edges ({config.max_tunneling_edges_per_node}/node)"})
        if config.use_higher_order_coupling:
            advanced_layers.append({"id": "higher_order", "label": "Higher-Order Coupling", "neurons": config.higher_order_hidden_dim, "kind": "liquid", "connectivity": "sparse salient-node coupling", "details": f"Top-k={config.higher_order_topk} higher-order interactions"})
        if config.use_deliberation_loop:
            advanced_layers.append({"id": "deliberation", "label": "Deliberation Loop", "neurons": config.deliberation_hidden_dim, "kind": "head", "connectivity": "proposer + critic", "details": f"{config.num_deliberation_steps} proposer-critic refinement steps"})
        insert_at = next((idx for idx, layer in enumerate(layers) if layer["id"] == "site_hidden"), len(layers))
        layers[insert_at:insert_at] = advanced_layers
    elif config.model_variant == "hybrid_selective":
        hybrid_layers = []
        if config.use_local_tunneling_bias:
            hybrid_layers.append({"id": "local_tunneling", "label": "Local Tunneling Bias", "neurons": config.tunneling_hidden_dim, "kind": "physics", "connectivity": "local bounded bias", "details": "Per-atom barrier/tunneling score used only for site-logit bias"})
        if config.use_output_refinement:
            hybrid_layers.append({"id": "output_refine", "label": "Output Refiner", "neurons": config.output_refinement_hidden_dim, "kind": "head", "connectivity": "single-step rerank", "details": "Single gated output correction near the site head"})
        insert_at = next((idx for idx, layer in enumerate(layers) if layer["id"] == "site_out"), len(layers))
        layers[insert_at:insert_at] = hybrid_layers
    return layers


def _edge_specs(config: ModelConfig) -> List[Dict[str, object]]:
    shared_hidden = config.shared_hidden_dim or config.hidden_dim
    site_hidden = config.som_head_hidden_dim
    cyp_hidden = config.cyp_head_hidden_dim
    edges = [
        {"source": "input", "target": "physics_raw", "connections": 4480, "label": "hand-crafted slice"},
        {"source": "input", "target": "manual_prior", "connections": config.atom_input_dim * config.som_branch_dim, "label": "manual features"},
        {"source": "input", "target": "steric_3d", "connections": config.atom_input_dim * max(1, config.steric_hidden_dim), "label": "optional 3D descriptors"},
        {"source": "physics_raw", "target": "physics_proj", "connections": 32 * config.physics_dim, "label": "projection"},
        {"source": "input", "target": "shared_input", "connections": config.atom_input_dim * shared_hidden, "label": "shared input projection"},
        {"source": "shared_input", "target": "shared_liquid", "connections": shared_hidden * shared_hidden * config.shared_encoder_layers, "label": "edge-aware liquid"},
        {"source": "physics_proj", "target": "fusion_gate", "connections": config.physics_dim * shared_hidden, "label": "physics branch"},
        {"source": "shared_liquid", "target": "fusion_gate", "connections": shared_hidden * shared_hidden, "label": "shared liquid"},
        {"source": "fusion_gate", "target": "som_branch", "connections": shared_hidden * config.som_branch_dim, "label": "SoM split"},
        {"source": "fusion_gate", "target": "cyp_branch", "connections": shared_hidden * config.cyp_branch_dim, "label": "CYP split"},
        {"source": "cyp_branch", "target": "group_pool", "connections": config.cyp_branch_dim * config.group_pooling_hidden_dim * 16, "label": "group pooling"},
        {"source": "group_pool", "target": "mol_pool", "connections": config.cyp_branch_dim * config.cyp_branch_dim, "label": "molecule pooling"},
        {"source": "som_branch", "target": "site_hidden", "connections": config.som_branch_dim * site_hidden, "label": "site residual"},
        {"source": "site_hidden", "target": "site_out", "connections": site_hidden, "label": "site logit"},
        {"source": "mol_pool", "target": "cyp_hidden", "connections": config.cyp_branch_dim * cyp_hidden, "label": "cyp residual"},
        {"source": "cyp_hidden", "target": "cyp_out", "connections": cyp_hidden * config.num_cyp_classes, "label": "cyp logits"},
    ]
    if config.model_variant == "advanced":
        if config.use_phase_augmented_state:
            edges.extend([
                {"source": "fusion_gate", "target": "phase_state", "connections": shared_hidden * shared_hidden, "label": "phase modulation"},
                {"source": "phase_state", "target": "som_branch", "connections": shared_hidden * config.som_branch_dim, "label": "phase -> SoM"},
                {"source": "phase_state", "target": "cyp_branch", "connections": shared_hidden * config.cyp_branch_dim, "label": "phase -> CYP"},
            ])
        if config.use_physics_residual:
            edges.extend([
                {"source": "physics_proj", "target": "physics_residual", "connections": config.physics_dim * (config.som_branch_dim + config.cyp_branch_dim), "label": "physics residual"},
                {"source": "physics_residual", "target": "site_hidden", "connections": config.som_branch_dim * site_hidden, "label": "physics -> site"},
                {"source": "physics_residual", "target": "cyp_hidden", "connections": config.cyp_branch_dim * cyp_hidden, "label": "physics -> cyp"},
            ])
        if config.use_energy_module:
            edges.extend([
                {"source": "som_branch", "target": "energy_module", "connections": config.som_branch_dim * config.energy_hidden_dim, "label": "node energy"},
                {"source": "mol_pool", "target": "energy_module", "connections": config.cyp_branch_dim * config.energy_hidden_dim, "label": "mol energy"},
            ])
        if config.use_tunneling_module:
            edges.append({"source": "som_branch", "target": "tunneling_module", "connections": config.som_branch_dim * config.tunneling_hidden_dim, "label": "barrier head"})
        if config.use_graph_tunneling:
            source_node = "tunneling_module" if config.use_tunneling_module else "som_branch"
            edges.extend([
                {"source": source_node, "target": "graph_tunneling", "connections": config.som_branch_dim * config.graph_tunneling_dim, "label": "sparse tunnel edges"},
                {"source": "graph_tunneling", "target": "site_hidden", "connections": config.graph_tunneling_dim * site_hidden, "label": "tunnel -> site"},
            ])
        if config.use_higher_order_coupling:
            source_node = "graph_tunneling" if config.use_graph_tunneling else "som_branch"
            edges.extend([
                {"source": source_node, "target": "higher_order", "connections": config.som_branch_dim * config.higher_order_hidden_dim, "label": "salient coupling"},
                {"source": "higher_order", "target": "site_hidden", "connections": config.higher_order_hidden_dim * site_hidden, "label": "higher-order -> site"},
            ])
        if config.use_deliberation_loop:
            deliberation_inputs = 0
            if config.use_energy_module:
                deliberation_inputs += config.energy_hidden_dim
            if config.use_tunneling_module:
                deliberation_inputs += config.tunneling_hidden_dim
            deliberation_inputs += config.som_branch_dim + config.cyp_branch_dim
            edges.extend([
                {"source": "som_branch", "target": "deliberation", "connections": config.som_branch_dim * config.deliberation_hidden_dim, "label": "proposer atoms"},
                {"source": "mol_pool", "target": "deliberation", "connections": config.cyp_branch_dim * config.deliberation_hidden_dim, "label": "proposer molecule"},
                {"source": "deliberation", "target": "site_hidden", "connections": config.deliberation_hidden_dim * site_hidden, "label": "critic -> site"},
                {"source": "deliberation", "target": "cyp_hidden", "connections": config.deliberation_hidden_dim * cyp_hidden, "label": "critic -> cyp"},
            ])
    elif config.model_variant == "hybrid_selective":
        if config.use_local_tunneling_bias:
            edges.extend([
                {"source": "som_branch", "target": "local_tunneling", "connections": config.som_branch_dim * config.tunneling_hidden_dim, "label": "local barrier head"},
                {"source": "local_tunneling", "target": "site_out", "connections": config.tunneling_hidden_dim, "label": "bounded tunnel bias"},
            ])
        if config.use_output_refinement:
            source = "local_tunneling" if config.use_local_tunneling_bias else "site_hidden"
            edges.extend([
                {"source": "som_branch", "target": "output_refine", "connections": config.som_branch_dim * config.output_refinement_hidden_dim, "label": "site features"},
                {"source": source, "target": "output_refine", "connections": config.output_refinement_hidden_dim, "label": "current site logits"},
                {"source": "output_refine", "target": "site_out", "connections": config.output_refinement_hidden_dim, "label": "single-step correction"},
            ])
    return edges


def build_architecture_graph_data(model: Optional[LiquidMetabolismNetV2] = None, config: Optional[ModelConfig] = None) -> Dict[str, object]:
    if config is None:
        config = ModelConfig()
    if model is None:
        model = LiquidMetabolismNetV2(config)

    layers = _layer_specs(config)
    edges = _edge_specs(config)
    total_connections = int(sum(edge["connections"] for edge in edges))
    total_neurons = int(sum(int(layer["neurons"]) for layer in layers))
    stats = {
        "trainable_parameters": _count_parameters(model),
        "estimated_dense_connections": total_connections,
        "total_reported_neurons": total_neurons,
        "model_variant": config.model_variant,
        "liquid_layers": config.num_liquid_layers,
        "shared_encoder_layers": config.shared_encoder_layers,
        "som_branch_layers": config.som_branch_layers,
        "cyp_branch_layers": config.cyp_branch_layers,
        "ode_steps_per_layer": config.ode_steps,
        "effective_recurrent_steps": (config.shared_encoder_layers + config.som_branch_layers + config.cyp_branch_layers) * config.ode_steps,
        "functional_groups": 16,
        "mean_abs_weight": round(_average_abs_weight(model), 6),
        "connection_profile": {
            "physics": "deterministic -> dense projection",
            "liquid": "edge-aware graph message passing + contextual RK4 integration",
            "fusion": "branched dense gate with residual priors",
            "pooling": "hierarchical chemistry attention pooling",
            "heads": "residual fusion heads",
        },
    }
    if config.model_variant == "advanced":
        stats["advanced_features"] = {
            "energy": config.use_energy_module,
            "tunneling": config.use_tunneling_module,
            "graph_tunneling": config.use_graph_tunneling,
            "phase": config.use_phase_augmented_state,
            "higher_order": config.use_higher_order_coupling,
            "physics_residual": config.use_physics_residual,
            "deliberation_steps": config.num_deliberation_steps,
        }
    elif config.model_variant == "hybrid_selective":
        stats["hybrid_features"] = {
            "local_tunneling_bias": config.use_local_tunneling_bias,
            "local_tunneling_scale": config.local_tunneling_scale,
            "output_refinement": config.use_output_refinement,
            "output_refinement_scale": config.output_refinement_scale,
        }
    return {"config": config.__dict__, "layers": layers, "edges": edges, "stats": stats}


def _sampled_nodes_and_edges(graph: Dict[str, object], sample_size: int = 8) -> Dict[str, object]:
    color_map = {
        "input": "#355c7d",
        "physics": "#c06c84",
        "liquid": "#6c5b7b",
        "fusion": "#f67280",
        "pool": "#99b898",
        "head": "#f8b195",
        "output": "#2a9d8f",
    }
    nodes = []
    edges = []
    x_positions = {}
    max_neuron_rows = 0
    for idx, layer in enumerate(graph["layers"]):
        layer_id = str(layer["id"])
        layer_kind = str(layer["kind"])
        neuron_count = int(layer["neurons"])
        shown = min(sample_size, max(1, neuron_count if neuron_count < sample_size else sample_size))
        max_neuron_rows = max(max_neuron_rows, shown)
        x_positions[layer_id] = 180 + idx * 220
        nodes.append({
            "id": layer_id,
            "label": f"{layer['label']}\n{neuron_count} units",
            "shape": "rect",
            "w": 150,
            "h": 58,
            "x": x_positions[layer_id],
            "y": 96,
            "color": color_map.get(layer_kind, "#888"),
            "meta": layer,
        })
        for neuron_idx in range(shown):
            node_id = f"{layer_id}:n{neuron_idx}"
            nodes.append({
                "id": node_id,
                "label": f"n{neuron_idx + 1}",
                "shape": "circle",
                "r": 9,
                "x": x_positions[layer_id],
                "y": 190 + neuron_idx * 34,
                "color": color_map.get(layer_kind, "#888"),
                "meta": {**layer, "sample_neuron_index": neuron_idx},
            })
            edges.append({"from": layer_id, "to": node_id, "kind": "anchor", "width": 1, "color": "#bcccdc"})
    node_index = {node["id"]: node for node in nodes}
    for edge in graph["edges"]:
        source = str(edge["source"])
        target = str(edge["target"])
        src_nodes = [node for node in nodes if str(node["id"]).startswith(f"{source}:n")][:sample_size]
        dst_nodes = [node for node in nodes if str(node["id"]).startswith(f"{target}:n")][:sample_size]
        for src in src_nodes:
            for dst in dst_nodes[: min(4, len(dst_nodes))]:
                edges.append({"from": src["id"], "to": dst["id"], "kind": "sample", "width": 1, "color": "rgba(127,140,141,0.25)"})
        edges.append({"from": source, "to": target, "kind": "summary", "width": 3, "color": "#34495e", "label": f"{int(edge['connections']):,}", "meta": edge})
    width = max(1800, 360 + max(0, len(graph["layers"]) - 1) * 220)
    height = max(760, 300 + max_neuron_rows * 36)
    return {"nodes": nodes, "edges": edges, "node_index": node_index, "width": width, "height": height}


def export_architecture_html(
    output_path: str,
    model: Optional[LiquidMetabolismNetV2] = None,
    config: Optional[ModelConfig] = None,
    checkpoint_path: Optional[str] = None,
) -> Path:
    if config is None:
        config = ModelConfig()
    if model is None:
        model = LiquidMetabolismNetV2(config)
    loaded_checkpoint = None
    if checkpoint_path and TORCH_AVAILABLE and Path(checkpoint_path).exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(state, strict=False)
        loaded_checkpoint = str(checkpoint_path)

    graph = build_architecture_graph_data(model=model, config=config)
    sampled = _sampled_nodes_and_edges(graph)
    payload = {"graph": graph, "sampled": sampled, "loaded_checkpoint": loaded_checkpoint}

    canvas_width = int(sampled["width"])
    canvas_height = int(sampled["height"])

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>LiquidMetabolismNetV2 Architecture</title>
  <style>
    body {{ margin: 0; font-family: Menlo, Monaco, monospace; background: #f3efe7; color: #1f2933; }}
    .layout {{ display: grid; grid-template-columns: minmax(980px, 1fr) 380px; min-height: 100vh; }}
    #network-wrap {{ height: 100vh; overflow: auto; background: radial-gradient(circle at top left, #fffaf0 0%, #e7dfd6 100%); border-right: 1px solid #d1c7ba; }}
    #network {{ width: {canvas_width}px; height: {canvas_height}px; }}
    .sidebar {{ padding: 20px; overflow: auto; background: linear-gradient(180deg, #faf7f2 0%, #efe7dc 100%); }}
    .card {{ background: rgba(255,255,255,0.82); border: 1px solid #d8cfc3; border-radius: 14px; padding: 14px 16px; margin-bottom: 14px; box-shadow: 0 10px 30px rgba(60, 45, 30, 0.06); }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .metric {{ padding: 8px 10px; background: #fff; border-radius: 10px; border: 1px solid #ece2d6; }}
    .metric strong {{ display: block; font-size: 18px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ text-align: left; padding: 6px 4px; border-bottom: 1px solid #e4dacd; vertical-align: top; }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    .hint {{ font-size: 12px; color: #52606d; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: #1f2933; color: #fff; font-size: 11px; margin-right: 6px; }}
    code {{ font-family: Menlo, Monaco, monospace; }}
    .node-label {{ fill: #ffffff; font-size: 16px; font-weight: 700; text-anchor: middle; pointer-events: none; }}
    .node-sublabel {{ fill: #ffffff; font-size: 11px; text-anchor: middle; pointer-events: none; }}
    .neuron-label {{ fill: #ffffff; font-size: 10px; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }}
    .clickable {{ cursor: pointer; }}
    .toolbar {{ display: flex; gap: 8px; margin-top: 10px; }}
    .toolbar button {{ border: 1px solid #d8cfc3; background: #fff; border-radius: 10px; padding: 6px 10px; font: inherit; cursor: pointer; }}
  </style>
</head>
<body>
  <div class=\"layout\">
    <div id=\"network-wrap\">
      <svg id=\"network\" viewBox=\"0 0 {canvas_width} {canvas_height}\" preserveAspectRatio=\"xMinYMin meet\"></svg>
    </div>
    <div class=\"sidebar\">
      <div class=\"card\">
        <h1>LiquidMetabolismNetV2</h1>
        <div class=\"hint\">Self-contained interactive architecture graph. Click any layer or sampled neuron to inspect details.</div>
        <div style=\"margin-top:10px\">{('<span class="badge">checkpoint loaded</span><code>' + loaded_checkpoint + '</code>') if loaded_checkpoint else '<span class="badge">config only</span>'}</div>
        <div class=\"toolbar\">
          <button id=\"fit-overview\">Fit Overview</button>
          <button id=\"actual-size\">Actual Size</button>
        </div>
      </div>
      <div class=\"card\">
        <h2>Stats</h2>
        <div class=\"grid\" id=\"stats-grid\"></div>
      </div>
      <div class=\"card\">
        <h2>Selected Node</h2>
        <div id=\"node-detail\" class=\"hint\">Select a layer or neuron in the graph.</div>
      </div>
      <div class=\"card\">
        <h2>Layer Summary</h2>
        <table>
          <thead><tr><th>Layer</th><th>Units</th><th>Connectivity</th></tr></thead>
          <tbody id=\"layer-table\"></tbody>
        </table>
      </div>
      <div class=\"card\">
        <h2>Connection Summary</h2>
        <table>
          <thead><tr><th>From</th><th>To</th><th>Edges</th></tr></thead>
          <tbody id=\"edge-table\"></tbody>
        </table>
      </div>
    </div>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const svg = document.getElementById('network');
    const wrap = document.getElementById('network-wrap');
    const statsGrid = document.getElementById('stats-grid');
    Object.entries(payload.graph.stats).forEach(([key, value]) => {{
      if (typeof value === 'object') return;
      const card = document.createElement('div');
      card.className = 'metric';
      card.innerHTML = `<span>${{key.replaceAll('_', ' ')}}</span><strong>${{value}}</strong>`;
      statsGrid.appendChild(card);
    }});

    const layerTable = document.getElementById('layer-table');
    payload.graph.layers.forEach((layer) => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{layer.label}}</td><td>${{layer.neurons}}</td><td>${{layer.connectivity}}</td>`;
      layerTable.appendChild(tr);
    }});

    const edgeTable = document.getElementById('edge-table');
    payload.graph.edges.forEach((edge) => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{edge.source}}</td><td>${{edge.target}}</td><td>${{Number(edge.connections).toLocaleString()}}</td>`;
      edgeTable.appendChild(tr);
    }});

    const nodeMap = Object.fromEntries(payload.sampled.nodes.map((node) => [node.id, node]));

    function make(tag, attrs = {{}}, parent = svg) {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      parent.appendChild(el);
      return el;
    }}

    function renderNodeDetail(node) {{
      const detail = document.getElementById('node-detail');
      if (!node) {{
        detail.innerHTML = 'Select a layer or neuron in the graph.';
        return;
      }}
      const meta = node.meta || {{}};
      const cleanLabel = String(node.label).split('\\n').join(' ');
      detail.innerHTML = `
        <div><strong>${{cleanLabel}}</strong></div>
        <div style="margin-top:8px"><strong>kind:</strong> ${{meta.kind || 'n/a'}}</div>
        <div><strong>units:</strong> ${{meta.neurons ?? 'n/a'}}</div>
        <div><strong>connectivity:</strong> ${{meta.connectivity || 'n/a'}}</div>
        <div><strong>details:</strong> ${{meta.details || 'sample neuron node'}}</div>
        ${{meta.sample_neuron_index !== undefined ? `<div><strong>sample neuron:</strong> #${{meta.sample_neuron_index + 1}}</div>` : ''}}
      `;
    }}

    function drawEdge(edge) {{
      const src = nodeMap[edge.from];
      const dst = nodeMap[edge.to];
      if (!src || !dst) return;
      const srcY = src.shape === 'rect' ? src.y + src.h / 2 : src.y;
      const dstY = dst.shape === 'rect' ? dst.y - dst.h / 2 : dst.y;
      make('line', {{ x1: src.x, y1: srcY, x2: dst.x, y2: dstY, stroke: edge.color, 'stroke-width': edge.width }});
      if (edge.kind === 'summary' && edge.label) {{
        const mx = (src.x + dst.x) / 2;
        const my = (srcY + dstY) / 2 - 8;
        const text = make('text', {{ x: mx, y: my, fill: '#34495e', 'font-size': 12, 'text-anchor': 'middle' }});
        text.textContent = edge.label;
      }}
    }}

    payload.sampled.edges.forEach(drawEdge);

    payload.sampled.nodes.forEach((node) => {{
      const group = make('g', {{ class: 'clickable', 'data-id': node.id }});
      if (node.shape === 'rect') {{
        make('rect', {{ x: node.x - node.w / 2, y: node.y - node.h / 2, width: node.w, height: node.h, rx: 14, fill: node.color, stroke: '#263238', 'stroke-width': 1.2 }}, group);
        const lines = String(node.label).split('\\n');
        const label = make('text', {{ x: node.x, y: node.y - 4, class: 'node-label' }}, group);
        label.textContent = lines[0];
        const sub = make('text', {{ x: node.x, y: node.y + 18, class: 'node-sublabel' }}, group);
        sub.textContent = lines[1] || '';
      }} else {{
        make('circle', {{ cx: node.x, cy: node.y, r: node.r, fill: node.color, stroke: '#263238', 'stroke-width': 1 }}, group);
        const text = make('text', {{ x: node.x, y: node.y + 0.5, class: 'neuron-label' }}, group);
        text.textContent = node.label;
      }}
      group.addEventListener('click', () => renderNodeDetail(node));
    }});

    function fitOverview() {{
      wrap.scrollLeft = 0;
      wrap.scrollTop = 0;
      svg.style.width = '100%';
      svg.style.height = 'auto';
      svg.style.minWidth = '{canvas_width}px';
    }}

    function actualSize() {{
      svg.style.width = '{canvas_width}px';
      svg.style.height = '{canvas_height}px';
      svg.style.minWidth = '{canvas_width}px';
    }}

    document.getElementById('fit-overview').addEventListener('click', fitOverview);
    document.getElementById('actual-size').addEventListener('click', actualSize);
    fitOverview();
  </script>
</body>
</html>
"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
