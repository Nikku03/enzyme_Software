from __future__ import annotations

from pathlib import Path

from enzyme_software.liquid_nn_v2.analysis.visualization import build_architecture_graph_data, export_architecture_html
from enzyme_software.liquid_nn_v2.config import ModelConfig


def test_build_architecture_graph_data():
    graph = build_architecture_graph_data(config=ModelConfig())
    assert graph['stats']['trainable_parameters'] > 0
    assert len(graph['layers']) >= 10
    assert len(graph['edges']) >= 10


def test_export_architecture_html(tmp_path):
    output = tmp_path / 'architecture.html'
    path = export_architecture_html(str(output), config=ModelConfig())
    html = Path(path).read_text()
    assert output.exists()
    assert 'LiquidMetabolismNetV2' in html
    assert 'trainable_parameters' in html
    assert '<svg id="network"' in html
