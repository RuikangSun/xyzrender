"""Tests for thermal ellipsoid (ORTEP-style) rendering."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from xyzrender.thermal import adp_to_cartesian, ellipse_from_adp, probability_scale, support_radius

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"
THERMAL_CIF = STRUCTURES / "thermal_ellipsoid.cif"
CAFFEINE_XYZ = STRUCTURES / "caffeine.xyz"


def test_probability_scale_ortep_values() -> None:
    """ORTEP's 50 % primer constant is 1.54; the exact chi-square value is 1.5382."""
    assert probability_scale(50.0) == pytest.approx(1.5382, abs=5e-4)
    assert probability_scale(90.0) == pytest.approx(2.5003, abs=5e-4)


@pytest.mark.parametrize("probability", [0.0, 100.0, -5.0, 150.0])
def test_probability_scale_rejects_out_of_range(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        probability_scale(probability)


def test_adp_to_cartesian_orthorhombic() -> None:
    """A fractional-basis tensor converts back to the requested Cartesian U."""
    lattice = np.diag([10.0, 20.0, 30.0])
    expected = np.diag([0.04, 0.02, 0.03])
    # U_star = U_cart / (a_i a_j) for an orthorhombic cell.
    u_star = expected / np.outer(np.diag(lattice), np.diag(lattice))
    cart = adp_to_cartesian(u_star, lattice)
    np.testing.assert_allclose(cart, expected)


def test_ellipse_from_isotropic_adp_is_circle() -> None:
    rx, ry, _angle = ellipse_from_adp(np.diag([0.04, 0.04, 0.04]), probability_scale(50.0), 100.0)
    assert rx == pytest.approx(ry)
    assert rx == pytest.approx(1.5382 * 0.2 * 100.0, rel=1e-3)


def test_ellipse_from_anisotropic_adp_uses_projected_shadow() -> None:
    """The silhouette uses the upper-left block, not the z=0 cross-section."""
    rx, ry, angle = ellipse_from_adp(np.diag([0.09, 0.01, 0.04]), probability_scale(50.0), 100.0)
    assert rx == pytest.approx(1.5382 * 0.3 * 100.0, rel=1e-3)
    assert ry == pytest.approx(1.5382 * 0.1 * 100.0, rel=1e-3)
    assert min(abs(angle), abs(abs(angle) - 180.0)) < 1e-6


def test_support_radius_along_principal_axis() -> None:
    tensor = np.diag([0.04, 0.01, 0.01])
    assert support_radius(tensor, np.array([1.0, 0.0, 0.0])) == pytest.approx(0.2)
    assert support_radius(tensor, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.1)


@pytest.mark.filterwarnings("ignore::UserWarning:ase")
def test_cif_adp_parsed_and_symmetry_rotated() -> None:
    pytest.importorskip("ase", reason="ase required")
    from xyzrender.parsers import parse_cif

    data = parse_cif(THERMAL_CIF)
    assert len(data.atoms) == 20
    assert data.adp is not None
    assert len(data.adp) == len(data.atoms)
    assert all(tensor is not None for tensor in data.adp)

    # P21/c generates four copies of each general site.  They share the same
    # eigenvalues (the ellipsoid shape) but not necessarily the same tensor.
    carbon_tensors = [
        adp_to_cartesian(tensor, data.pbc_cell)
        for (symbol, _), tensor in zip(data.atoms, data.adp, strict=True)
        if symbol == "C"
    ]
    assert len(carbon_tensors) == 8
    # The first four carbons are the anisotropic C1 site.
    carbon_tensors = carbon_tensors[:4]
    eigenvalues = [np.linalg.eigvalsh(tensor) for tensor in carbon_tensors]
    for values in eigenvalues[1:]:
        np.testing.assert_allclose(values, eigenvalues[0], atol=1e-12)
    assert max(np.abs(carbon_tensors[0] - tensor).max() for tensor in carbon_tensors[1:]) > 1e-6


@pytest.mark.filterwarnings("ignore::UserWarning:ase")
def test_cif_adp_survives_ghost_generation() -> None:
    """Periodic image atoms inherit the ADP tensor of their source atom."""
    import networkx as nx

    from xyzrender.crystal import add_crystal_images
    from xyzrender.types import CellData

    adp = np.diag([0.02, 0.03, 0.01])
    graph = nx.Graph()
    graph.add_node(0, symbol="C", position=(0.0, 0.0, 0.0), adp=adp)
    graph.add_node(1, symbol="O", position=(9.8, 0.0, 0.0), adp=adp)
    graph.add_edge(0, 1, bond_order=1.0)
    add_crystal_images(graph, CellData(lattice=np.diag([10.0, 10.0, 10.0])))
    ghosts = [node for node in graph.nodes() if graph.nodes[node].get("image", False)]
    assert ghosts
    for node in ghosts:
        np.testing.assert_allclose(graph.nodes[node]["adp"], adp)


@pytest.mark.filterwarnings("ignore::UserWarning:ase")
def test_render_thermal_ellipses() -> None:
    pytest.importorskip("ase", reason="ase required")
    from xyzrender import load, render

    mol = load(THERMAL_CIF)
    svg = str(render(mol, orient=False, thermal=True))
    assert svg.count("<ellipse") == 20
    assert "<ellipse" not in str(render(mol, orient=False))

    # Thermal plots show C-H by default; --no-hy hides them again.
    no_h = str(render(mol, orient=False, thermal=True, no_hy=True))
    assert no_h.count("<ellipse") == 16


@pytest.mark.filterwarnings("ignore::UserWarning:ase")
def test_thermal_probability_scales_ellipses() -> None:
    pytest.importorskip("ase", reason="ase required")
    from xyzrender import load, render

    mol = load(THERMAL_CIF)

    def mean_rx(probability: float) -> float:
        svg = str(render(mol, orient=False, thermal=True, thermal_prob=probability))
        values = [float(value) for value in re.findall(r'rx="([0-9.]+)"', svg)]
        assert values
        return float(np.mean(values))

    ratio = mean_rx(90.0) / mean_rx(50.0)
    assert ratio == pytest.approx(probability_scale(90.0) / probability_scale(50.0), rel=5e-3)

    # thermal_prob alone implies thermal=True (matching the CLI).
    assert "<ellipse" in str(render(mol, orient=False, thermal_prob=60.0))


def test_thermal_ignored_without_lattice(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from xyzrender import load, render

    mol = load(CAFFEINE_XYZ)
    assert mol.cell_data is None
    with caplog.at_level(logging.WARNING, logger="xyzrender.api"):
        svg = str(render(mol, orient=False, thermal=True))
    assert "<ellipse" not in svg
    assert "thermal ellipsoids require" in caplog.text


def test_gif_viewport_reserves_thermal_ellipsoid_size() -> None:
    """The fixed GIF viewport must fit the ellipsoids, not just the display spheres."""
    from xyzrender.gif import _fixed_viewport
    from xyzrender.types import CellData, RenderConfig

    cfg = RenderConfig(thermal=True, thermal_prob=50, cell_data=CellData(lattice=np.diag([10.0, 10.0, 10.0])))
    frame = {
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "symbols": ["C", "C"],
        "adp": [np.diag([0.25, 0.01, 0.01]), None],
    }
    fixed = _fixed_viewport([frame], cfg)
    assert fixed.fixed_span is not None
    # 1 Å separation + two 0.77 Å ellipsoid pads (instead of two 0.13 Å atom pads).
    assert fixed.fixed_span > 2.0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cif_nonstandard_spacegroup_symbol() -> None:
    """Full H-M symbols with explicit identity operations fall back to ASE's short form."""
    pytest.importorskip("ase", reason="ase required")
    from xyzrender.parsers import parse_cif

    data = parse_cif(STRUCTURES / "p21_full_symbol.cif")
    assert len(data.atoms) == 2
