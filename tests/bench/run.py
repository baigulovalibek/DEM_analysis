"""
DEM Analyst — analysis-routine benchmark.

Times each Phase A / Phase B target routine on three problem sizes
(512², 2 048², 4 096²) and writes a markdown timing table next to this
script so we can track progress through the optimization plan.

Usage::

    python -m tests.bench.run                       # all three sizes
    python -m tests.bench.run --sizes 512,1024      # subset
    python -m tests.bench.run --repeats 5           # more samples
    python -m tests.bench.run --skip-viewshed       # viewshed at 4k² is slow

The output file is ``tests/bench/RESULTS.md``.  Each Phase A / B / C commit
should re-run this and update that file so we have a trail of measured wins.
"""
from __future__ import annotations
import argparse
import gc
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from tests.synth_dem import make_dem


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "RESULTS.md"
DEFAULT_SIZES = (512, 2048, 4096)


@dataclass
class Bench:
    name: str
    fn: Callable[[np.ndarray], object]
    # routine-specific overrides per-size (e.g. viewshed radius scales with size)
    skip_at: tuple[int, ...] = field(default_factory=tuple)


def _time(fn: Callable[[], object], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return statistics.median(samples)


def _all_benches(cell_size: float, viewshed_radius: int) -> list[Bench]:
    from app.core.derivatives.hillshade import hillshade
    from app.core.derivatives.slope import slope
    from app.core.derivatives.aspect import aspect
    from app.core.derivatives.curvature import profile_curvature
    from app.core.indices.tri import tri
    from app.core.indices.roughness import roughness_sd, vrm
    from app.core.visibility.multidirectional import multidirectional_hillshade
    from app.core.visibility.svf import sky_view_factor
    from app.core.visibility.viewshed import viewshed
    from app.core.hydrology.fill_sinks import priority_flood, breach_and_fill
    from app.core.hydrology.flow_direction import (
        d8_flow_direction, d_infinity_flow_direction,
    )
    from app.core.hydrology.flow_accumulation import (
        d8_flow_accumulation, d_inf_flow_accumulation,
    )

    return [
        Bench("hillshade",        lambda dem: hillshade(dem, cell_size)),
        Bench("slope",            lambda dem: slope(dem, cell_size)),
        Bench("aspect",           lambda dem: aspect(dem, cell_size)),
        Bench("profile_curv",     lambda dem: profile_curvature(dem, cell_size)),
        Bench("tri",              lambda dem: tri(dem)),
        Bench("roughness_sd",     lambda dem: roughness_sd(dem)),
        Bench("vrm",              lambda dem: vrm(dem, cell_size)),
        Bench("multidirectional", lambda dem: multidirectional_hillshade(dem, cell_size)),
        Bench("svf",              lambda dem: sky_view_factor(dem, cell_size,
                                                              n_directions=8,
                                                              max_radius=20)),
        # Hydrology — the heaviest routines.  priority_flood at 4 k² takes
        # a minute on the pure-Python path; benchmark with --sizes 512,2048
        # to skip it if iterating quickly.
        Bench("priority_flood",   lambda dem: priority_flood(dem)),
        Bench("breach_and_fill",  lambda dem: breach_and_fill(dem),
              skip_at=(4096,)),
        Bench("d8_flow_dir",      lambda dem: d8_flow_direction(dem, cell_size)),
        Bench("d_inf_flow_dir",   lambda dem: d_infinity_flow_direction(dem, cell_size)),
        Bench("d8_accum",
              # Requires flow_dir — build it per call to keep the timing fair.
              lambda dem: d8_flow_accumulation(dem, d8_flow_direction(dem, cell_size))),
        Bench("d_inf_accum",
              lambda dem: d_inf_flow_accumulation(
                  dem, d_infinity_flow_direction(dem, cell_size)[0])),
        Bench("viewshed",
              lambda dem: viewshed(dem, cell_size, dem.shape[0] // 2,
                                   dem.shape[1] // 2, observer_height=10.0,
                                   max_radius=viewshed_radius),
              skip_at=()),
    ]


def run(sizes: tuple[int, ...], repeats: int, skip_viewshed: bool) -> dict[str, dict[int, float]]:
    results: dict[str, dict[int, float]] = {}
    for size in sizes:
        print(f"\n-- {size}x{size} --")
        dem = make_dem(size)
        # Cap viewshed radius so its O(n²·r) cost stays sane at 4 k²
        # (full-grid radius would take well over an hour pre-Phase B).
        vr = min(size // 4, 200)
        benches = _all_benches(cell_size=30.0, viewshed_radius=vr)
        for b in benches:
            if size in b.skip_at:
                continue
            if skip_viewshed and b.name == "viewshed":
                continue
            t = _time(lambda: b.fn(dem), repeats)
            results.setdefault(b.name, {})[size] = t
            print(f"  {b.name:>18s}  {t * 1000:9.1f} ms")
    return results


def write_results(results: dict[str, dict[int, float]], sizes: tuple[int, ...]) -> None:
    header = "| Routine | " + " | ".join(f"{s}^2" for s in sizes) + " |"
    sep = "|" + "---|" * (len(sizes) + 1)
    lines = [
        "# DEM Analyst -- Benchmark results",
        "",
        "Median wall-time (ms) of 3 runs per routine, per DEM size.  Refresh ",
        "by re-running ``python -m tests.bench.run`` after each optimization ",
        "commit so we have a measured trail of wins through Phases A -> C of ",
        "OPTIMIZATION.md.",
        "",
        header,
        sep,
    ]
    for name, by_size in results.items():
        cells = [
            f"{by_size[s] * 1000:.1f}" if s in by_size else "—"
            for s in sizes
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                        help="Comma-separated DEM side lengths to benchmark")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-viewshed", action="store_true",
                        help="Skip viewshed at all sizes (very slow pre-Phase B)")
    args = parser.parse_args()

    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())
    results = run(sizes, args.repeats, args.skip_viewshed)
    write_results(results, sizes)


if __name__ == "__main__":
    main()
