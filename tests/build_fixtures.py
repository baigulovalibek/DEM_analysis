"""
Build golden-output fixtures for the regression test suite.

Run once after a known-good state — e.g. immediately after Phase A —
to lock the expected outputs.  Phase B's Numba kernels must reproduce
these arrays exactly (for integer outputs and priority_flood) or
within the documented float tolerance.

Usage::

    python -m tests.build_fixtures             # 512² fixtures
    python -m tests.build_fixtures --size 256  # smaller / quicker

Fixtures live in ``tests/fixtures/`` and are git-tracked so CI can
verify against them without re-running the heavy hydrology routines.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from tests.synth_dem import NODATA, make_dem, make_dem_with_void


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


def _save(name: str, arr: np.ndarray) -> None:
    out = FIXTURE_DIR / f"{name}.npy"
    np.save(out, arr)
    print(f"  wrote {out.name}  shape={arr.shape}  dtype={arr.dtype}")


def build(size: int = 512) -> None:
    cell_size = 30.0
    dem = make_dem(size)
    dem_void = make_dem_with_void(size)
    print(f"Synthetic DEM {size}x{size} float32 — relief "
          f"[{dem.min():.1f}, {dem.max():.1f}] m")

    _save(f"dem_{size}", dem)
    _save(f"dem_void_{size}", dem_void)

    # ── primary derivatives ───────────────────────────────────────────
    from app.core.derivatives.hillshade import hillshade
    from app.core.derivatives.slope import slope
    from app.core.derivatives.aspect import aspect
    from app.core.derivatives.curvature import (
        profile_curvature, plan_curvature, mean_curvature,
    )

    _save(f"hillshade_{size}", hillshade(dem, cell_size))
    _save(f"slope_{size}", slope(dem, cell_size))
    _save(f"aspect_{size}", aspect(dem, cell_size))
    _save(f"profile_curv_{size}", profile_curvature(dem, cell_size))
    _save(f"plan_curv_{size}", plan_curvature(dem, cell_size))
    _save(f"mean_curv_{size}", mean_curvature(dem, cell_size))

    # nodata-path coverage
    _save(f"hillshade_void_{size}", hillshade(dem_void, cell_size, nodata=NODATA))
    _save(f"slope_void_{size}", slope(dem_void, cell_size, nodata=NODATA))

    # ── indices ───────────────────────────────────────────────────────
    from app.core.indices.tri import tri
    from app.core.indices.roughness import (
        roughness_range, roughness_sd, vrm, surface_area_ratio,
    )

    _save(f"tri_{size}", tri(dem))
    _save(f"roughness_range_{size}", roughness_range(dem))
    _save(f"roughness_sd_{size}", roughness_sd(dem))
    _save(f"vrm_{size}", vrm(dem, cell_size))
    _save(f"sar_{size}", surface_area_ratio(dem, cell_size))

    # ── visibility (small parameters so this stays under a few seconds) ─
    from app.core.visibility.multidirectional import multidirectional_hillshade
    from app.core.visibility.svf import (
        sky_view_factor, positive_openness, negative_openness,
    )

    _save(f"multidir_{size}", multidirectional_hillshade(dem, cell_size))
    _save(f"svf_{size}", sky_view_factor(dem, cell_size, n_directions=8, max_radius=20))
    _save(f"pos_open_{size}", positive_openness(dem, cell_size, n_directions=8, max_radius=20))
    _save(f"neg_open_{size}", negative_openness(dem, cell_size, n_directions=8, max_radius=20))

    # ── hydrology (the routines targeted by Phase B JIT) ──────────────
    from app.core.hydrology.fill_sinks import priority_flood, breach_and_fill
    from app.core.hydrology.flow_direction import d8_flow_direction, d_infinity_flow_direction
    from app.core.hydrology.flow_accumulation import (
        d8_flow_accumulation, d_inf_flow_accumulation,
    )
    from app.core.hydrology.streams import extract_streams, strahler_order

    filled = priority_flood(dem)
    _save(f"priority_flood_{size}", filled)
    _save(f"breach_fill_{size}", breach_and_fill(dem))

    fd = d8_flow_direction(filled, cell_size)
    _save(f"d8_flow_dir_{size}", fd)
    fa = d8_flow_accumulation(filled, fd)
    _save(f"d8_flow_accum_{size}", fa)

    angle, _slope = d_infinity_flow_direction(filled, cell_size)
    _save(f"d_inf_angle_{size}", angle)
    fa_inf = d_inf_flow_accumulation(filled, angle)
    _save(f"d_inf_flow_accum_{size}", fa_inf)

    # ── viewshed (single deterministic observer near grid centre) ─────
    from app.core.visibility.viewshed import viewshed
    obs_r, obs_c = size // 2, size // 2
    vs = viewshed(filled, cell_size, obs_r, obs_c,
                  observer_height=10.0, max_radius=size // 4)
    _save(f"viewshed_{size}", vs)

    streams = extract_streams(fa, threshold=float(np.nanpercentile(fa, 99)))
    _save(f"streams_{size}", streams)
    _save(f"strahler_{size}", strahler_order(streams, fd))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    build(args.size)


if __name__ == "__main__":
    main()
