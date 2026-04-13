"""
Visualization Module for Cellular Field Model
==============================================

Provides visualization capabilities for:
1. 2D concentration field slices
2. 3D isosurface rendering
3. Pathway/reaction flow diagrams
4. Knockout effect comparisons
5. Trajectory animations
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def create_concentration_slice(
    model: nn.Module,
    gene_expression: torch.Tensor,
    metabolite_idx: int = 21,  # ATP by default
    z_level: float = 0.0,
    t_value: float = 0.0,
    resolution: int = 50,
    x_range: Tuple[float, float] = (-0.5, 0.5),
    y_range: Tuple[float, float] = (-0.5, 0.5),
) -> Dict[str, np.ndarray]:
    """
    Create a 2D slice of the metabolite concentration field.
    
    Args:
        model: CellularFieldModel
        gene_expression: (1, n_genes) gene expression
        metabolite_idx: Which metabolite to visualize
        z_level: Z-coordinate for the slice
        t_value: Time coordinate
        resolution: Grid resolution
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        
    Returns:
        Dict with 'concentration', 'x', 'y', 'compartment' arrays
    """
    # Create grid
    x = torch.linspace(x_range[0], x_range[1], resolution)
    y = torch.linspace(y_range[0], y_range[1], resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten and add z, t
    n_points = resolution * resolution
    coords = torch.stack([
        xx.flatten(),
        yy.flatten(),
        torch.full((n_points,), z_level),
        torch.full((n_points,), t_value),
    ], dim=-1).unsqueeze(0)  # (1, N, 4)
    
    # Query field
    with torch.no_grad():
        if hasattr(model, 'metabolite_field'):
            field_out = model.metabolite_field(coords, gene_expression)
        else:
            # Assume model is MetaboliteField directly
            field_out = model(coords, gene_expression)
    
    # Extract concentration and compartment
    conc = field_out['concentrations'][0, :, metabolite_idx].numpy()
    comp = field_out['compartment'][0, :].numpy()
    
    # Reshape to grid
    conc_grid = conc.reshape(resolution, resolution)
    comp_grid = comp.reshape(resolution, resolution)
    
    return {
        'concentration': conc_grid,
        'compartment': comp_grid,
        'x': x.numpy(),
        'y': y.numpy(),
        'metabolite_idx': metabolite_idx,
        'z_level': z_level,
        't_value': t_value,
    }


def create_3d_volume(
    model: nn.Module,
    gene_expression: torch.Tensor,
    metabolite_idx: int = 21,
    t_value: float = 0.0,
    resolution: int = 20,
    bounds: Tuple[float, float] = (-0.5, 0.5),
) -> Dict[str, np.ndarray]:
    """
    Create a 3D volume of metabolite concentrations.
    
    Returns:
        Dict with 'volume' (3D array), 'x', 'y', 'z' coordinate arrays
    """
    x = torch.linspace(bounds[0], bounds[1], resolution)
    y = torch.linspace(bounds[0], bounds[1], resolution)
    z = torch.linspace(bounds[0], bounds[1], resolution)
    
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    n_points = resolution ** 3
    coords = torch.stack([
        xx.flatten(),
        yy.flatten(),
        zz.flatten(),
        torch.full((n_points,), t_value),
    ], dim=-1).unsqueeze(0)
    
    with torch.no_grad():
        if hasattr(model, 'metabolite_field'):
            field_out = model.metabolite_field(coords, gene_expression)
        else:
            field_out = model(coords, gene_expression)
    
    conc = field_out['concentrations'][0, :, metabolite_idx].numpy()
    volume = conc.reshape(resolution, resolution, resolution)
    
    return {
        'volume': volume,
        'x': x.numpy(),
        'y': y.numpy(),
        'z': z.numpy(),
    }


def knockout_comparison(
    model: nn.Module,
    gene_expression: torch.Tensor,
    metabolite_state: torch.Tensor,
    gene_indices: List[int],
    gene_names: Optional[List[str]] = None,
    metabolite_names: Optional[List[str]] = None,
    key_metabolites: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compare metabolite effects across multiple gene knockouts.
    
    Returns:
        Dict with 'delta_matrix' (genes x metabolites), names, etc.
    """
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in gene_indices]
    
    n_mets = metabolite_state.shape[-1]
    if metabolite_names is None:
        metabolite_names = [f"Met_{i}" for i in range(n_mets)]
    
    if key_metabolites is None:
        key_metabolites = list(range(min(10, n_mets)))
    
    # Baseline
    with torch.no_grad():
        baseline = model(gene_expression, metabolite_state, n_steps=5)
    baseline_mets = baseline['next_metabolites'][0].numpy()
    
    # Knockouts
    delta_matrix = np.zeros((len(gene_indices), len(key_metabolites)))
    
    for i, gene_idx in enumerate(gene_indices):
        with torch.no_grad():
            ko = model.knockout(gene_expression, metabolite_state, gene_idx=gene_idx)
        ko_mets = ko['next_metabolites'][0].numpy()
        
        for j, met_idx in enumerate(key_metabolites):
            delta_matrix[i, j] = ko_mets[met_idx] - baseline_mets[met_idx]
    
    return {
        'delta_matrix': delta_matrix,
        'gene_names': gene_names,
        'metabolite_names': [metabolite_names[i] for i in key_metabolites],
        'baseline': baseline_mets[key_metabolites],
    }


def trajectory_data(
    model: nn.Module,
    gene_expression: torch.Tensor,
    initial_state: torch.Tensor,
    n_steps: int = 50,
    metabolite_indices: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate trajectory data for metabolite dynamics.
    
    Returns:
        Dict with 'trajectories' (time x metabolites), 'time' array
    """
    n_mets = initial_state.shape[-1]
    if metabolite_indices is None:
        metabolite_indices = list(range(min(10, n_mets)))
    
    with torch.no_grad():
        out = model(gene_expression, initial_state, n_steps=n_steps)
    
    trajectory = out['trajectory'][:, 0, :].numpy()  # (time, metabolites)
    
    return {
        'trajectories': trajectory[:, metabolite_indices],
        'time': np.arange(len(trajectory)),
        'metabolite_indices': metabolite_indices,
    }


def create_svg_heatmap(
    data: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Concentration Field",
    colormap: str = "viridis",
    width: int = 600,
    height: int = 500,
) -> str:
    """
    Create an SVG heatmap visualization.
    """
    rows, cols = data.shape
    
    if x_labels is None:
        x_labels = [str(i) for i in range(cols)]
    if y_labels is None:
        y_labels = [str(i) for i in range(rows)]
    
    # Normalize data to 0-1
    vmin, vmax = data.min(), data.max()
    if vmax - vmin > 1e-10:
        normalized = (data - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(data)
    
    # Cell dimensions
    margin_left = 80
    margin_top = 60
    margin_right = 100  # For colorbar
    margin_bottom = 40
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    cell_w = plot_width / cols
    cell_h = plot_height / rows
    
    # Color function (viridis-like)
    def value_to_color(v):
        # Simple viridis approximation
        r = int(255 * (0.267 + 0.329 * v + 0.144 * v**2))
        g = int(255 * (0.004 + 0.873 * v - 0.134 * v**2))
        b = int(255 * (0.329 + 0.280 * v + 0.362 * v**2))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return f"rgb({r},{g},{b})"
    
    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width//2}" y="30" text-anchor="middle" font-size="16" font-weight="bold">{title}</text>',
    ]
    
    # Heatmap cells
    for i in range(rows):
        for j in range(cols):
            x = margin_left + j * cell_w
            y = margin_top + i * cell_h
            color = value_to_color(normalized[i, j])
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                f'fill="{color}" stroke="white" stroke-width="0.5"/>'
            )
    
    # Colorbar
    cb_x = width - margin_right + 20
    cb_width = 20
    cb_height = plot_height
    
    for i in range(50):
        v = i / 49
        y = margin_top + (1 - v) * cb_height
        color = value_to_color(v)
        svg_parts.append(
            f'<rect x="{cb_x}" y="{y:.1f}" width="{cb_width}" height="{cb_height/50:.1f}" '
            f'fill="{color}" stroke="none"/>'
        )
    
    # Colorbar labels
    svg_parts.append(
        f'<text x="{cb_x + cb_width + 5}" y="{margin_top + 10}" font-size="10">{vmax:.2f}</text>'
    )
    svg_parts.append(
        f'<text x="{cb_x + cb_width + 5}" y="{margin_top + cb_height}" font-size="10">{vmin:.2f}</text>'
    )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def create_svg_pathway(
    metabolites: List[str],
    reactions: List[Tuple[int, int, str]],  # (from_idx, to_idx, enzyme_name)
    flux_values: Optional[List[float]] = None,
    width: int = 800,
    height: int = 600,
) -> str:
    """
    Create an SVG pathway diagram.
    
    Args:
        metabolites: List of metabolite names
        reactions: List of (from_idx, to_idx, enzyme_name) tuples
        flux_values: Optional flux values for arrow thickness
    """
    n_mets = len(metabolites)
    
    # Arrange metabolites in a grid or circle
    if n_mets <= 12:
        # Circle layout
        cx, cy = width // 2, height // 2
        radius = min(width, height) // 3
        
        positions = []
        for i in range(n_mets):
            angle = 2 * math.pi * i / n_mets - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            positions.append((x, y))
    else:
        # Grid layout
        cols = int(math.ceil(math.sqrt(n_mets)))
        rows = int(math.ceil(n_mets / cols))
        
        cell_w = (width - 100) / cols
        cell_h = (height - 100) / rows
        
        positions = []
        for i in range(n_mets):
            row = i // cols
            col = i % cols
            x = 50 + col * cell_w + cell_w / 2
            y = 50 + row * cell_h + cell_h / 2
            positions.append((x, y))
    
    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<defs>',
        '  <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">',
        '    <path d="M0,0 L0,6 L9,3 z" fill="#666"/>',
        '  </marker>',
        '</defs>',
    ]
    
    # Draw reactions (arrows)
    for i, (from_idx, to_idx, enzyme) in enumerate(reactions):
        x1, y1 = positions[from_idx]
        x2, y2 = positions[to_idx]
        
        # Shorten arrows to not overlap nodes
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx / length, dy / length
            x1 += dx * 25
            y1 += dy * 25
            x2 -= dx * 25
            y2 -= dy * 25
        
        # Arrow thickness from flux
        stroke_width = 2
        if flux_values and i < len(flux_values):
            stroke_width = 1 + 3 * abs(flux_values[i])
        
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#666" stroke-width="{stroke_width:.1f}" marker-end="url(#arrow)"/>'
        )
        
        # Enzyme label at midpoint
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 - 5
        svg_parts.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" font-size="10" fill="#888">{enzyme}</text>'
        )
    
    # Draw metabolite nodes
    for i, (x, y) in enumerate(positions):
        # Node circle
        svg_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="20" fill="#4CAF50" stroke="#388E3C" stroke-width="2"/>'
        )
        
        # Label
        label = metabolites[i]
        if len(label) > 6:
            label = label[:5] + "..."
        svg_parts.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="10" fill="white" font-weight="bold">{label}</text>'
        )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def create_ascii_heatmap(
    data: np.ndarray,
    width: int = 40,
    height: int = 20,
) -> str:
    """
    Create an ASCII art heatmap.
    """
    # Resample to target size
    from scipy.ndimage import zoom
    
    zoom_y = height / data.shape[0]
    zoom_x = width / data.shape[1]
    
    # Use simple nearest neighbor
    resampled = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            src_i = int(i / zoom_y)
            src_j = int(j / zoom_x)
            src_i = min(src_i, data.shape[0] - 1)
            src_j = min(src_j, data.shape[1] - 1)
            resampled[i, j] = data[src_i, src_j]
    
    # Normalize
    vmin, vmax = resampled.min(), resampled.max()
    if vmax - vmin > 1e-10:
        normalized = (resampled - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(resampled)
    
    # Convert to ASCII
    chars = " ░▒▓█"
    lines = []
    lines.append("┌" + "─" * width + "┐")
    
    for i in range(height):
        row = "│"
        for j in range(width):
            idx = int(normalized[i, j] * (len(chars) - 1))
            row += chars[idx]
        row += "│"
        lines.append(row)
    
    lines.append("└" + "─" * width + "┘")
    
    return "\n".join(lines)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Visualization Module...")
    
    # Import model
    import sys
    sys.path.insert(0, '/home/claude/dark_manifold_cell')
    from src.qft.cellular_field import CellularFieldModel
    
    # Create model
    model = CellularFieldModel(n_genes=531, n_metabolites=83)
    
    # Test inputs
    gene_expr = torch.rand(1, 531)
    met_state = torch.rand(1, 83)
    
    # Test concentration slice
    print("\n1. Concentration slice...")
    slice_data = create_concentration_slice(model, gene_expr, metabolite_idx=21)
    print(f"   Shape: {slice_data['concentration'].shape}")
    
    # Test 3D volume
    print("\n2. 3D volume...")
    vol_data = create_3d_volume(model, gene_expr, metabolite_idx=21, resolution=10)
    print(f"   Shape: {vol_data['volume'].shape}")
    
    # Test knockout comparison
    print("\n3. Knockout comparison...")
    ko_data = knockout_comparison(
        model, gene_expr, met_state,
        gene_indices=[10, 20, 30],
        key_metabolites=[21, 22, 23],
    )
    print(f"   Delta matrix: {ko_data['delta_matrix'].shape}")
    
    # Test trajectory
    print("\n4. Trajectory...")
    traj_data = trajectory_data(model, gene_expr, met_state, n_steps=10)
    print(f"   Trajectories: {traj_data['trajectories'].shape}")
    
    # Test ASCII heatmap
    print("\n5. ASCII heatmap:")
    ascii_map = create_ascii_heatmap(slice_data['concentration'], width=30, height=10)
    print(ascii_map)
    
    print("\n✓ All visualization tests passed!")
