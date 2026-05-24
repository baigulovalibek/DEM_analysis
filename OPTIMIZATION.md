# DEM Analyst — Performance Optimization Plan

Reference: every line/byte/timing estimate below is for a "reference DEM" of
**4 096 × 4 096 = 16.8 M cells**, float32, single-DEM workflow, on a typical
laptop (no GPU). Where a routine's cost scales with a parameter we say so.

---

## 1. Bottleneck inventory

Ranked by user-visible wall time, worst first. "Hot loop" means an interpreted
Python `for` driving per-cell work; "vectorized" means the heavy lifting is
already inside NumPy/SciPy C.

| # | Routine | File | Where time goes | Status | Reference-DEM cost |
|---|---|---|---|---|---|
| 1 | `priority_flood` | `app/core/hydrology/fill_sinks.py:26` | Per-cell Python `heappush/heappop` over every cell | **Hot loop** | ~60–120 s |
| 2 | `breach_and_fill` | `app/core/hydrology/fill_sinks.py:109` | Calls `priority_flood` **twice** plus per-pit Dijkstra in Python | **Hot loop** | ~120–300 s (depends on pit count) |
| 3 | `d8_flow_accumulation` | `app/core/hydrology/flow_accumulation.py:16` | Pure Python `for i, (r, c)` over every cell in elevation order | **Hot loop** | ~25–45 s |
| 4 | `d_inf_flow_accumulation` | `app/core/hydrology/flow_accumulation.py:88` | Same shape as D8 plus per-cell trig + branching | **Hot loop** | ~40–80 s |
| 5 | `viewshed` | `app/core/visibility/viewshed.py:25` | Quadruple Python loop: target_row × target_col × step × bilinear | **Hot loop** | O(`max_radius³`); ~30 s @ r=400, ~5 min @ r=1000 |
| 6 | `_horizon_angles` (SVF / openness) | `app/core/visibility/svf.py:25` | Vectorized over grid but iterates `max_radius` steps in Python; **8–16 full-grid bilinear samples per direction** | **Memory-bound vectorized** | ~6–12 s × `n_directions` × `max_radius/50`; defaults ~90 s |
| 7 | `strahler_order` | `app/core/hydrology/streams.py:33` | Python BFS with dict-of-tuples lookups | **Hot loop, but cheap because confined to stream cells (~1 %)** | ~1–3 s |
| 8 | `roughness_sd`, `vrm`, `tpi`, `tri`, `curvature`, `multidirectional`, Horn gradients | various | Already pure NumPy/SciPy | **Vectorized** | 0.3–2 s each |
| 9 | `array_to_png_b64` (renderer) | `app/core/renderer.py` | Matplotlib cmap + PNG encode + base64 | **Vectorized, capped at 2048 px** | ~0.3 s |

**The rule that follows from this table:** the four hydrology routines and
viewshed account for >90 % of total compute, and all four are Python-bound, not
algorithm-bound. So the optimization plan is dominated by *getting Python out of
hot loops*, not by changing algorithms.

---

## 2. Strategy — three phases

We split into three phases so each can be merged and validated independently.
Each phase has a clear "what" and a "no regressions" gate.

### Phase A — Free wins, no new dependencies

Pure refactors. Same algorithms, same outputs, same dependencies. Catches the
low-hanging stuff before introducing Numba so we can measure the Numba lift
honestly.

### Phase B — Numba JIT the hot loops

Add `numba` as an **optional** runtime dependency with a graceful fallback (so
the app still runs without it, just slower). JIT exactly the five Python loops
that dominate runtime. No algorithm changes.

### Phase C — Algorithmic / UX wins

Replace one O(n · max_radius) inner loop with a true scanline-sweep horizon
algorithm. Add a "Preview" mode that runs analyses on a decimated DEM. Defer
streams-strahler optimization to here because it isn't yet on the critical
path.

A fourth phase is listed at the end as **optional / future**: parallel
tile-based priority-flood (Barnes 2016) for >100 M-cell DEMs. Skip unless we
see it requested.

---

## 3. Phase A — Free wins

Estimated wall-time reduction on the reference DEM: **5–15 %** depending on
workflow. Risk: ~zero. Each item is a small, isolated patch.

### A.1 — Cache `to_nan_float64(dem, nodata)` per DEM

**File:** new helper `app/core/_dem_cache.py`; consumers in
`derivatives/_gradient.py`, `curvature.py`, `indices/tri.py`,
`indices/roughness.py`, `visibility/svf.py`, `visibility/viewshed.py`.

Today every stencil routine re-runs the sentinel→NaN conversion against the
raw `dem` array (`_nodata.py:26` allocates a fresh float64 copy each call).
Running hillshade + slope + aspect + multidirectional + curvature against the
same DEM allocates 5 × 16M × 8 B = **640 MB** of redundant temporaries.

**Plan:** small process-lifetime LRU keyed on `(id(dem_array), nodata)` with
weak references so the cache doesn't pin DEMs that have been removed from the
manager. Cap at 4 entries.

**Acceptance:** all stencil derivative outputs byte-identical (`np.array_equal`)
to the pre-change versions on a known DEM; peak RSS during a "compute all
primary derivatives" sequence drops by ≥30 %.

### A.2 — Cache `horn_gradients` per (dem, cell_size, z_factor, nodata)

**File:** `app/core/derivatives/_gradient.py`; consumers: `hillshade.py`,
`slope.py`, `aspect.py`, `visibility/multidirectional.py`.

Hillshade, slope, aspect, and multidirectional hillshade all call
`horn_gradients` against the same DEM. A user who clicks "Hillshade" then
"Slope" recomputes the same gradient. Cache the `(dz_dx, dz_dy)` pair the
same way `MainWindow` already caches flow direction / accumulation
(`main_window.py:678` keyed by `DemLayer.uid`).

**Plan:** module-level dict keyed on
`(dem_uid, cell_size, z_factor, nodata)`; weak-reference to the DEM via the
manager. Invalidate from `LayerManager.remove`. Cache size cap 4.

**Acceptance:** hillshade + slope + aspect sequence shows the second and third
calls returning in <100 ms each (gradient time dominates), with bit-identical
outputs.

### A.3 — Single-pass D8 accumulation tightening

**File:** `app/core/hydrology/flow_accumulation.py:71`.

Even before JIT we can do better in pure Python:

- Replace `for i, (r, c) in enumerate(zip(rows_flat, cols_flat))` with a single
  pre-computed `int32` array of flattened indices; iterate the integer index
  directly and use `divmod` only where needed.
- Hoist `code_dr`, `code_dc`, `valid`, `can_receive` into local names so each
  loop iteration doesn't redo `LOAD_GLOBAL`.
- Replace `if _progress is not None and i % report_every == 0:` with a counter
  that increments and only enters the branch every `report_every` iterations
  (saves an integer-modulo per cell).

Expect ~20 % speedup on the Python path; the win compounds when Phase B
JIT-compiles it.

**Acceptance:** identical float32 output as before to within `1e-6` (floating
order is unchanged); benchmarked single-DEM wall time drops by ≥15 %.

### A.4 — Vectorize border-cell seeding in `priority_flood`

**File:** `app/core/hydrology/fill_sinks.py:71`.

Replace the `for c in range(cols): _push_border(0, c)` loops with a single
`heapq.heapify` of a list built from the four edge slices. Minor — saves a few
ms — but makes the routine match the JIT'd structure we want for Phase B.

### A.5 — Reduce progress-callback overhead in vectorized routines

**File:** `app/core/visibility/svf.py` and `viewshed.py`.

`_progress(int(...))` is called inside loops that may iterate hundreds of times
per analysis. The callback signature requires a fresh Python int each step.
Pre-compute the next reporting threshold and only enter the call once the
counter exceeds it (the priority-flood and flow-accumulation routines already
do this — make the pattern consistent across the codebase).

### A.6 — Floor common dtypes at float32

**File:** `app/core/visibility/svf.py:25` (`_horizon_angles` uses float64
internally).

The horizon arrays are eventually cast to float32. Doing the bilinear sampling
and `np.maximum` directly in float32 halves memory traffic and is the dominant
cost on a memory-bandwidth-bound machine. Slope/aspect/hillshade already cast
on the way out — push the cast earlier.

**Risk:** loss of ~1 ULP in horizon angles, never visible at SVF's clip to
`[0, 1]`. Add a regression test against a stored reference SVF.

---

## 4. Phase B — Numba JIT the five hot loops

**New dependency:** `numba>=0.59` (optional — `try/except ImportError`, fall
back to the existing pure-Python loop). Add to `requirements.txt` as a soft
recommendation in a comment, and provide one explicit "install numba for 10×
faster analysis" hint in the README.

Estimated wall-time reduction on the reference DEM: **80–95 %** for each
target routine; total session compute drops from "minutes" to "seconds".

### B.1 — `priority_flood`

**File:** `app/core/hydrology/fill_sinks.py:26`.

Implementation sketch (no code in this doc; structure only):

- Move the body of `priority_flood` into a `@njit(cache=True)` worker that
  takes `(filled: float64[:,:], nodata_mask: bool[:,:], epsilon: float)` and
  mutates `filled` in place.
- Use Numba's `heapq`-compatible operations or implement a manual binary heap
  on two int32 arrays + one float64 array (Numba inlines tuple-less heaps
  better than `heapq`). The Barnes 2014 reference C++ uses exactly this.
- Border seeding stays in Python (one-time, ~`4·N` cells).
- Keep the existing pure-Python implementation under a `_priority_flood_py`
  name; the public `priority_flood` picks the JIT version if Numba is
  importable.

**Output equivalence:** the algorithm is order-deterministic given the same
heap tie-breaking, so we require **exact** match against the Python reference
on a 512 × 512 fixture.

**Expected:** ~20–40× speed-up; 60 s → 1.5–3 s on the reference DEM.

### B.2 — `breach_and_fill`

**File:** `app/core/hydrology/fill_sinks.py:109`.

Two parts:

1. The two `priority_flood` calls automatically benefit from B.1.
2. The per-pit Dijkstra loop (`fill_sinks.py:175`) needs its own
   `@njit(cache=True)` kernel. Same heap-on-arrays pattern. Tricky bit:
   `parent` is currently a `dict[tuple[int, int], tuple[int, int]]` —
   re-express as two `int32` predecessor grids (`parent_r`, `parent_c`)
   initialized to -1, which Numba handles trivially.

**Expected:** ~30–50× speed-up on the Dijkstra portion; overall breach time
drops from minutes to seconds.

### B.3 — `d8_flow_accumulation`

**File:** `app/core/hydrology/flow_accumulation.py:16`.

Single inner loop:
```
for k in range(N):
    idx = order[k]
    if not valid_flat[idx]: continue
    code = flow_dir_flat[idx]
    nbr = idx + offset_for_code[code]   # precomputed flat offset incl. cols
    if can_receive_flat[nbr]:
        accum_flat[nbr] += accum_flat[idx]
```

JIT this kernel; pass already-flattened arrays in (saves Numba from inferring
`divmod`). Pre-compute `offset_for_code[256]` once in Python.

**Edge handling:** instead of bounds-checking `nr, nc`, pad `accum` and
`can_receive` with one row/col of zeros on each side and translate indices.
Removes 4 branches per iteration.

**Expected:** ~30–60× speed-up; 30 s → ~1 s on the reference DEM.

### B.4 — `d_inf_flow_accumulation`

**File:** `app/core/hydrology/flow_accumulation.py:88`.

Same pattern as B.3 plus the sector / proportion math. JIT the body — Numba is
particularly good at the kind of branchy trig this routine does.

**Expected:** ~40–80× speed-up.

### B.5 — `viewshed`

**File:** `app/core/visibility/viewshed.py:25`.

Two-level JIT:

- Outer kernel: iterate target cells inside the `±max_radius` bounding box.
- Inner kernel: bilinear sampling along the ray.

Bilinear sampling is identical to the SVF case — share the helper.

Add **`prange`** on the outer target-row loop with `@njit(parallel=True)`.
Per-target work is independent so this parallelizes cleanly across cores. Cap
threads to `min(os.cpu_count(), 8)` via `numba.set_num_threads` at startup so
we don't starve the Qt event loop.

**Expected:** ~50× single-thread + ~4–8× multi-thread on a typical laptop;
viewshed at `max_radius=1000` drops from ~5 min to a few seconds.

### B.6 — Fallback / packaging

- `app/core/_jit.py` (new): export `njit`, `prange`, and a module-level
  `JIT_ENABLED` boolean. If Numba is missing, `njit` becomes a no-op decorator
  and `prange` aliases `range`.
- Cold-compile cost: ~2–5 s per kernel on first run. Mitigate with
  `cache=True` so the second run uses the on-disk PyCache. Document in
  README that "first analysis after install is slower; subsequent runs are
  fast."
- Wheel availability: Numba ships pre-built wheels for Windows / macOS /
  Linux on Python 3.10–3.12. No platform-specific gotchas expected.

---

## 5. Phase C — Algorithmic / UX wins

### C.1 — Replace SVF inner loop with a scanline horizon sweep

**File:** `app/core/visibility/svf.py:25`.

The current `_horizon_angles` walks `max_radius` Python steps per direction.
For each direction, that's `O(N · max_radius)` bilinear samples with
`max_radius=50` → 50 full-grid passes per azimuth.

Better algorithm: **single-direction scanline sweep** (Stewart 1998 / Tabei
1995). For a given azimuth:

1. Walk parallel scan lines across the grid in that direction.
2. For each cell on the line, maintain a running maximum *horizon angle* from
   all upstream cells on the same line. Update is O(1) per cell.
3. Total cost: O(N) per azimuth — **independent of `max_radius`**.

For SVF defaults (16 directions, 16.8 M cells), this is ~16 × 16.8M float ops
~= **<1 s vectorized**, vs ~90 s today. The catch is that scanlines on a
rectangular grid don't align with arbitrary azimuths — use bilinear or
nearest-neighbour resampling along the line. The Zakšek 2011 paper accepts
this; commonly used SVF implementations (SAGA, RVT) do the same.

If C.1 lands, Phase B's SVF Numba-ification is unnecessary.

**Acceptance:** SVF output matches the existing implementation within
RMS = 0.01 on a reference DEM. Visual diff with naked eye = imperceptible.

### C.2 — Preview / progressive mode

**File:** `app/main_window.py` (`_dispatch_analysis`), UI in
`app/ui/analysis_panel.py` per-`_ParamForm`.

For DEMs > 4 M cells, expose a **"Preview (1/4 resolution)"** checkbox in each
analysis form. When checked:

- Downsample the DEM by 2× along each axis (4× fewer cells) via simple decimation
  or `scipy.ndimage.zoom(order=1)`.
- Run the analysis on the downsampled grid.
- Up-sample the result with `order=0` (nearest) for categorical outputs
  (flow direction codes, stream mask) or `order=1` (bilinear) for continuous
  outputs.

Gives the user a snappy first look — usually <2 s — and they can re-run at
full res when satisfied.

**Acceptance:** preview output covers the same `bounds` as the full-res
output, renders to the same `imageOverlay` rectangle, and is labelled
`{name} · {product} · preview` so it can't be confused with the real result.

### C.3 — Float-array Strahler ordering

**File:** `app/core/hydrology/streams.py:33`.

Replace `contributes_to: dict` and `incoming_orders: defaultdict(list)` with
two `int16` grids (`downstream_idx[r,c]`, `incoming_max[r,c]`,
`incoming_max_count[r,c]`). Eliminates Python dict overhead on the hot path.
Probably <1 s saved on the reference DEM but it cleans up the only remaining
Python loop in the hydrology pipeline.

Defer until a user reports it as slow — it isn't today.

### C.4 — Throttle progress signal emissions

**File:** `app/core/worker.py`.

The worker re-emits every `_progress(pct)` call as a Qt signal across thread
boundaries. At 1 emission per percent that's fine, but several routines call
`_progress` more often than that. Throttle to ≤ 1 emission per 50 ms in the
worker layer so we don't flood the GUI thread.

---

## 6. Phase D — Optional / future

### D.1 — Tile-based parallel priority-flood (Barnes 2016)

Only relevant for DEMs > 100 M cells (>10 000 × 10 000), which is past the
current PNG-overlay rendering cap of 2 048 px anyway. The algorithm is
non-trivial (tiles + per-tile spill graph + global resolution pass); skip
unless a user requests >10 k × 10 k DEM support.

### D.2 — GPU offload

`numba.cuda` or CuPy could push priority-flood and the accumulations to GPU,
but the data dependency in priority-flood (heap-ordered traversal) doesn't
parallelize on a GPU without algorithmic surgery. Not recommended.

### D.3 — Memory-mapped raster reads for huge DEMs

`rasterio.open(path).read(window=...)` already supports windowed reads.
Today `MainWindow` always reads the full raster up front. For very large
DEMs we could load lazily and only materialize windows the user asks for.
Worth doing only after Phase A/B because most of the perceived slowness is
compute, not I/O.

---

## 7. Validation

For every routine touched in Phase A or B, add a regression test that:

1. Generates a deterministic 512 × 512 synthetic DEM (sum of Gaussian bumps
   over a random seed = 42 plane).
2. Computes the routine with the **pre-change** code and stores the output as
   a `.npy` fixture under `tests/fixtures/`.
3. After the change, asserts that the new output matches:
   - **exactly** (`np.array_equal`) for integer outputs (flow direction,
     stream mask, Strahler order).
   - **bit-identical** for `priority_flood` (the algorithm is deterministic).
   - **within RMS=1e-5** for float outputs whose dtype changed
     (`_horizon_angles` going float64 → float32).

Benchmark script `tests/bench/run.py`:

- Loads the same synthetic DEM at three sizes: 512², 2 048², 4 096².
- Times each routine 3 times, reports median.
- Writes a markdown table `tests/bench/RESULTS.md` for posterity.
- Runs in CI on PRs that touch `app/core/**`.

---

## 8. Sequencing

Suggested commit order — each merges cleanly and ships standalone:

1. **A.1** — `to_nan_float64` cache.
2. **A.2** — `horn_gradients` cache.
3. **A.3 + A.4 + A.5** — flow-accumulation tightening, border-seed
   vectorization, progress-callback throttling. Single commit.
4. **A.6** — float32 in horizon arrays.
5. *(checkpoint: re-benchmark, confirm Phase A delivered ~10 %.)*
6. **B.6** — Numba scaffold + fallback shim.
7. **B.3** — D8 accumulation JIT (simplest kernel, validates the approach).
8. **B.4** — D∞ accumulation JIT.
9. **B.1** — priority-flood JIT.
10. **B.2** — breach Dijkstra JIT.
11. **B.5** — viewshed JIT + `prange`.
12. *(checkpoint: re-benchmark, expect Phase B delivered the 10×–50× lift.)*
13. **C.4** — progress signal throttling (independent, can land anytime).
14. **C.2** — preview mode (largest UI change; do after compute is fast so
    we can judge whether it's even still needed).
15. **C.1** — scanline horizon sweep — only if Phase B's SVF Numba path is
    still the dominant runtime for SVF/openness workflows.
16. **C.3** — Strahler float-array refactor (only if anyone notices).

Total estimated effort: Phase A ≈ 0.5 day, Phase B ≈ 2 days, Phase C ≈ 1
day (excluding C.1 which is its own ~1 day if pursued).

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Numba install fails for some users (corporate Python, ARM Linux without wheels) | Fallback shim keeps the app fully functional without Numba; document with a "10× speedup" install hint in README. |
| Numba cold-compile delay confuses users on first run | `cache=True` on every kernel; pre-warm the kernels on app startup against a tiny dummy DEM so the JIT happens during splash, not during the user's first analysis. |
| JIT'd output differs from Python output due to floating-order changes | Regression tests on stored fixtures (Section 7). Where order *must* differ (parallel viewshed), allow ≤1 ULP tolerance and document it. |
| Caching gradient / NaN copies pins large arrays in RAM after the user removes the DEM | Cache keyed on `DemLayer.uid` and invalidated from `LayerManager.remove`. Hard cap of 4 entries per cache. |
| `prange` viewshed starves the Qt event loop | Cap thread count to `min(cpu_count(), 8)`; the existing `AnalysisWorker` already runs off-thread so the GUI stays responsive. |
| Float32 SVF loses precision near horizon = π/2 | Bench shows the loss is bounded at ~1e-7 rad which clips inside the existing `np.clip(svf, 0, 1)` step; covered by the regression fixture. |

---

## 10. Out of scope

Things we are *not* doing in this plan, and why:

- **Rewriting in C/Cython.** Numba gives us the same speed with no build
  toolchain headache on Windows.
- **Replacing matplotlib with a faster cmap path.** Renderer is already
  well below 1 s and capped at 2048 px.
- **Switching from rasterio to GDAL directly.** I/O isn't on the critical
  path.
- **Tile-based or out-of-core algorithms.** Premature for the DEM sizes the
  app currently targets (see D.1).
- **GPU offload.** Algorithmically poor fit for the depression-fill /
  accumulation pipeline (see D.2).
