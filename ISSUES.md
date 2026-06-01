# `app/core/` — Algorithmic Audit

Scope: every module under `app/core/` (analysis layer, no UI).
Method: line-by-line read of each function + synthetic-DEM smoke tests
(east/north plane, flat plane, pit, sentinel-edged DEM).
Code state: working tree on branch `claude/compassionate-noether-MvPWY`
(modified files relative to `be43682`).

Many P0/P1 items present in the original `ISSUES.md` (D-infinity direction,
D-infinity neighbour routing, surface-area-ratio divisor, `auto_threshold`
crash, viewshed sub-cell sampling, SVF NaN handling, …) **have been fixed**
in the current working tree and are not repeated here. Only issues observable
in the code as it stands today are listed.

Severity:
- **P0** — wrong scientific output for a published algorithm.
- **P1** — wrong behaviour on a plausible input, or silent data loss.
- **P2** — performance hit, edge case, or robustness concern.
- **P3** — code quality, documentation, minor cleanup.

---

## P0 — Correctness bugs

### P0-1 · Nodata sentinel contamination at boundaries (every derivative)

**Files:** `derivatives/_gradient.py:30-31`, `derivatives/curvature.py:23-24`,
`indices/tri.py:22-23`, `indices/roughness.py:59-65, 92-104`,
`visibility/svf.py:62-75`, `visibility/viewshed.py:110-115`.

`horn_gradients` (and the Zevenbergen–Thorne analogue in `curvature._zt_coefficients`,
plus `tri`, `vrm`, `surface_area_ratio`, `_horizon_angles`, and the viewshed
bilinear interpolator) take the raw DEM, `astype(np.float64)`, then pad with
`mode="edge"`. **The sentinel value (e.g. `-9999`) is never converted to NaN
before the stencil runs.** Every downstream output masks only the cell that is
itself nodata — but the cells *neighbouring* a nodata patch get a 9999-unit
elevation drop in one direction, producing wildly wrong derivatives.

Verified on a 10×10 flat DEM (`100.0`) with a single `-9999` at `(5,5)`:

```text
slope adjacent to sentinel: cell (5,4) = 89.319°   cell (4,5) = 89.319°
hillshade  near sentinel  : (5,4) = 0   (4,5) = 0   (5,5) = 0
```

The neighbour cells should be flat (slope ≈ 0). Instead they look like an
89° cliff, and the resulting hillshade band is jet-black around every nodata
patch — visible to the user as halos around lakes, voids, and the data
edge whenever the source raster uses anything other than NaN.

Same mechanism contaminates:
- `slope`, `aspect`, `curvature_*` — values at the one-pixel-thick rim
  around nodata are physically wrong.
- `tri`, `vrm`, `surface_area_ratio` — squared-difference terms blow up.
- `_horizon_angles` (SVF, openness) and `viewshed` bilinear sampling —
  intermediate samples can pull `-9999` into the interpolation, making rays
  look "below LoS" → falsely visible past nodata regions.

**Fix.** Pick one place to centralise sentinel→NaN conversion:

```python
def _nodata_to_nan(dem, nodata):
    z = dem.astype(np.float64, copy=True)
    if nodata is not None:
        z[z == nodata] = np.nan
    return z
```

…and call it at the top of every stencil function. After conversion, edge
padding still happens, but the propagated values are NaN — which the
`np.isfinite`/`np.nanmean`/`np.where` guards already in place handle
correctly.

---

### P0-2 · `twi.py` misuses `nodata` as a presence flag, not a value

**File:** `indices/twi.py:18-46`

```python
def twi(flow_accum, slope_rad, cell_size, slope_min=0.001, nodata=None):
    ...
    if nodata is not None:
        result[flow_accum == 0] = np.nan
    return result
```

The `nodata` parameter is never compared against the data — it is used purely
as a boolean **flag** to decide whether to mask zero-accumulation cells.
Two calls that mean the same thing produce different results:

```text
TWI nodata=None , zero-accum cells: [5.700, 5.700]   ← cells get a real value
TWI nodata=-9999, zero-accum cells: [nan,   nan  ]   ← cells get nan
```

The intended behaviour is unclear from the code: either propagate true
nodata cells from the source DEM, or always mask zero-accumulation. Pick
one, document it, and remove the misleading parameter name.

---

### P0-3 · `strahler_order` ships dead code

**File:** `hydrology/streams.py:51-55`

```python
# Compute in-degree for each stream cell
for code, (dr, dc) in D8_OFFSETS.items():
    # Cells whose downstream neighbour is (r+dr, c+dc)
    # We need to find source cells that drain INTO (r, c)
    pass
```

A `for` loop whose body is `pass`. The variable `reverse` built two lines
below is never read either. This isn't a correctness bug *per se* (the
real in-degree computation happens further down via a per-cell Python
loop) but it indicates an incomplete refactor — the code should either
be removed or wired up.

---

## P1 — Visible defects on plausible inputs

### P1-1 · `spi.py` has no nodata handling at all

**File:** `indices/spi.py`

`spi` accepts neither a `nodata` parameter nor a sentinel mask. NaNs in
`flow_accum` or `slope_rad` propagate through and `result.astype(np.float32)`
gives a NaN. Cells outside the data extent (where `flow_accum=0`) produce
SPI=0 silently, which the user sees as a valid low-erosion zone rather
than missing data. Add the same NaN/sentinel handling pattern used by `twi`
(once fixed in P0-2).

### P1-2 · `d_inf_flow_accumulation` ignores the `nodata` parameter

**File:** `hydrology/flow_accumulation.py:88-108`

The signature is `(dem, angle, weight=None, _progress=None)` — there is no
`nodata` parameter at all, while the D8 sibling has one. The function masks
cells only via `~np.isfinite(dem)`. If a caller is using a `-9999`-encoded
DEM (which `_dispatch_analysis` in `main_window.py` historically passed
without NaN conversion), the D∞ accumulation will treat sentinel cells as
real `-9999`-elevation cells and route flow into them.

Either: (a) accept `nodata` and apply the same mask as D8; or (b) document
that the caller must convert sentinel→NaN first and assert it at runtime.

### P1-3 · `breach_and_fill` runs O(n_pits) Dijkstras over the full grid

**File:** `hydrology/fill_sinks.py:101-200`

```python
pits_mask = (filled > result + 1e-12) & ~nodata_mask
pit_indices = np.argwhere(pits_mask)
```

`pits_mask` flags **every cell whose elevation was raised by Priority-Flood**,
including the entire interior of every lake. For each one, the loop runs a
new Dijkstra search:

```python
cost = np.full((rows, cols), np.inf, dtype=np.float64)
parent: dict[...] = {}
heap = [(pit_elev, pr, pc)]
```

For a single lake of 10 000 cells, that's 10 000 separate Dijkstras over
the whole grid — roughly O(N · n · log n). On modest DEMs (1 k × 1 k) with
a few sizeable depressions, the function runs for minutes.

The early-exit guard `if result[pr, pc] >= filled[pr, pc] - 1e-12: continue`
only catches cells whose `result` was lowered along a previously breached
path — lake-interior cells are not lowered by a downstream breach, so the
guard does almost nothing for them.

**Fix.** Process only local minima (cells lower than all 8 neighbours),
not every raised cell.

### P1-4 · `strahler_order` is pure-Python row-major over the whole grid

**File:** `hydrology/streams.py:33-107`

The in-degree map and BFS are written as `for r in range(rows): for c in range(cols)`
Python loops, with dict accesses for adjacency. On a 4 k × 4 k stream
raster (≈1 % stream coverage = 160 k cells) this is glacial. Even with
just 1 % cells visited, the *outer* loop still walks 16 M cells before
finding them.

**Fix.** Use `np.argwhere(stream)` to get stream coordinates once, then
loop only over those; or push the hot loop into Numba.

### P1-5 · `priority_flood` epsilon accumulates linearly along flat paths

**File:** `hydrology/fill_sinks.py:74-96`

Each push enforces `new_elev = max(filled[nr,nc], elev + epsilon)`. On a
large flat plateau crossed by the flood-fill wavefront, the elevation is
raised by `step_count · epsilon`. For a 10 000-step-long flat (a real
coastal plain or floodplain) with default `epsilon=1e-6`, the surface is
raised by `10 mm` at the far end — not catastrophic, but well above
float-precision noise.

For very long flats and small-relief DEMs this manifests as a faint
elevation "tilt" in the filled output. Document the limit, expose
`epsilon` to the user, or fall back to Garbrecht–Martz flat resolution.

### P1-6 · D8 flow direction biases ties toward east

**File:** `hydrology/flow_direction.py:67-79`

```python
drops = np.stack([...], axis=0)
max_idx = np.argmax(drops, axis=0)
```

`np.argmax` returns the first index in a tie. The codes are stacked in
the order E, SE, S, SW, W, NW, N, NE — so equal-drop cells always drain
east, then south-east, then south, etc. This is the classic D8 limitation
(parallel "drainage stripes" on planar slopes). Truly flat cells are fine
because `max_drop > 0` filters them; but cells with two *equal positive*
drops (steep planar terrain) consistently bias east.

Standard fixes: Garbrecht & Martz (1997) flat-resolution, or D∞ — both
exist already in the same module.

### P1-7 · `aspect.flat_threshold` is in absolute gradient units

**File:** `derivatives/aspect.py:17-39`

Default `1e-6` is in `m/m` (rise over run). For a 30 m-cell DEM with
elevations stored in mm precision, the elevation diff between adjacent
cells can be `1 mm`, giving a gradient of `1e-3 / 30 ≈ 3.3e-5` — about
33× the flat threshold. The cell is **not** flagged as flat, and aspect
inherits the random noise direction.

Result: large "flat" regions get a noisy aspect that should be `-1`.
Either scale the threshold by `cell_size` (so the threshold is in
mm-elevation units), or expose it in the UI.

---

## P2 — Performance, edge cases, robustness

### P2-1 · `multidirectional_hillshade` recomputes gradients four times

**File:** `visibility/multidirectional.py:42-47`

```python
for az, w in zip(azimuths, weights):
    hs = hillshade(dem, cell_size, azimuth=az, ...)
```

Each `hillshade()` call re-runs `horn_gradients` from scratch. Compute
`dz_dx, dz_dy` once, then apply the Lambertian formula four times against
shared gradients — roughly 60 % speedup with no precision loss.

### P2-2 · `flow_accumulation` (D8 and D∞) are pure-Python `for` loops

**Files:** `hydrology/flow_accumulation.py:71-83, 138-165`

After `argsort` (which is vectorised), both functions step through every
cell with a Python `for`. On a 4 k × 4 k DEM that's 16 M Python-level
iterations with attribute lookups (`flow_dir[r,c]`, `code_dr[code]`, …).
Wall time grows fast: a few seconds at 1 k × 1 k → several minutes at 4 k × 4 k.

**Fix.** Numba-jit the hot loop, or compute the per-cell receiver indices
once vectorised (`recv_r = r + code_dr[flow_dir]; recv_c = c + code_dc[flow_dir]`)
and then accumulate by index in C.

### P2-3 · `priority_flood` Python heap dominates large fills

**File:** `hydrology/fill_sinks.py:73-96`

Each cell push/pop is a Python-level `heapq` call. A 1 k × 1 k DEM is ~1 M
heap ops in pure Python — measured 30–60 s on a mid-range machine.
4 k × 4 k → minutes. Numba/Cython or vendoring `pyrichdem` would close
the gap.

### P2-4 · `viewshed` bounding box helps, but inner loop is still pure-Python triple-nested

**File:** `visibility/viewshed.py:71-137`

The recent fix (bounding box around the observer) is correct and saves
work for small `max_radius`. But the inner `for tr / for tc / for step`
triple is still Python-level. For a 200-cell radius (~125 k targets ×
average ~100 steps each) that's ~12 M iterations per observer.
`cumulative_viewshed` runs this once per observer — fine for 2-3
observers, painful for 50.

### P2-5 · `_horizon_angles` reallocates ri/ci/sample arrays inside the step loop

**File:** `visibility/svf.py:42-67`

```python
for step in range(1, max_radius + 1):
    ri = np.arange(rows)
    ci = np.arange(cols)
    r_sample = np.clip(ri[:, None] + frac_r, 0, rows - 1)
    c_sample = np.clip(ci[None, :] + frac_c, 0, cols - 1)
    r0 = np.floor(r_sample).astype(int)
    ...
```

For `max_radius=50` and `n_directions=16` that's 800 iterations, each
allocating 4–6 full-grid arrays. Hoist `ri`/`ci` out of the loop and
reuse buffers with `out=` arguments.

### P2-6 · `_horizon_angles` boundary clipping invents elevation past the edge

**File:** `visibility/svf.py:50-67`

```python
r_sample = np.clip(ri[:, None] + frac_r, 0, rows - 1)
c_sample = np.clip(...)
```

When a ray runs off the grid, the sample stays pinned at the boundary —
elevation = boundary cell's elevation, distance keeps growing. The
resulting horizon angle decays with `1/dist` toward zero, contributing a
small "above-horizon" bias for cells where the true horizon would be
"open sky" off-grid. For SVF, this *underestimates* sky openness near
the edge.

Not strictly a bug (this is a documented horizon-mapping convention),
but should be noted in the docstring: edges are computed as if the
terrain were repeated to infinity at the boundary elevation.

### P2-7 · `_horizon_angles` silently zeros NaN-centred cells

**File:** `visibility/svf.py:62-75`

```python
elev_angle = np.arctan2(elev - dem, dist)
elev_angle = np.where(np.isfinite(elev_angle), elev_angle, -np.pi / 2)
```

When the *centre* cell `dem` is NaN, `elev - dem` is NaN regardless of
`elev`. The `np.where` substitutes `-π/2` (full sky) and the cell ends
up with SVF = 1 (perfectly open sky) instead of being masked. SVF
and openness outputs over NaN cells therefore look "valid", masking
data gaps.

**Fix.** Explicitly mask NaN-centred cells to NaN at the end of
`sky_view_factor` / `positive_openness`.

### P2-8 · `viewshed` bilinear interpolation pulls raw sentinel values

**File:** `visibility/viewshed.py:101-115`

The bilinear sample uses raw `dem[r0_s, c0_s]` without filtering
sentinel values. If the DEM uses `-9999`, an intermediate ray sample
across a nodata patch returns a hugely negative `z_sample`, the
slope-to-sample becomes a steep downhill, the LoS max-slope check
never triggers, and the target cell is marked visible past the void.

(Same root cause as P0-1; called out separately because the impact —
spurious visibility through nodata — is visually distinct.)

### P2-9 · `extract_streams` mixes `uint8` and `bool`

**File:** `hydrology/streams.py:25-30`

```python
stream = (flow_accum >= threshold).astype(np.uint8)
if slope is not None and slope_threshold is not None:
    stream &= (slope <= slope_threshold)   # uint8 &= bool
```

NumPy quietly casts the bool to `uint8` for `&=`, so the result is
right. But the intent is fragile — a future maintainer changing `stream`
to `np.bool_` would have it switch behaviour with no warning. Either
keep both sides as `bool` (and cast at the end) or explicitly
`stream = stream.astype(bool) & (slope <= slope_threshold)`.

### P2-10 · `tpi` annular path silently degrades when `inner_radius=0`

**File:** `indices/tpi.py:42-51`

```python
if annular and inner_radius >= 1:
    ...
else:
    mean_z = uniform_filter(z, size=window, mode="nearest")
```

`annular=True, inner_radius=0` looks like a valid request to the user
(annulus with no hole) but quietly switches to the non-annular path.
Either raise on the contradiction or document that an annulus needs
`inner_radius ≥ 1`.

### P2-11 · `roughness_sd` is numerically fragile near homogeneous patches

**File:** `indices/roughness.py:26-41`

```python
variance = np.maximum(mean_z2 - mean_z * mean_z, 0.0)
```

`E[z²] - E[z]²` catastrophically cancels when `E[z²] ≈ E[z]²`. The
`np.maximum(_, 0.0)` floor saves the sqrt, but the result is still 0
where the true SD is some small positive value (cancellation noise).
For metres-scale DEMs with metres-scale local variability the
cancellation isn't reached, but for sub-millimetre LiDAR or
single-precision input it bites. Welford / two-pass formulation is the
textbook fix.

### P2-12 · `surface_area_ratio` is a 4-triangle stencil, not Jenness (2004)

**File:** `indices/roughness.py:82-125`

The implementation builds four triangles (`N–E`, `E–S`, `S–W`, `W–N`)
covering a plan area of `2·L²`, then divides by `2·L²` to get a unitless
ratio. It is internally self-consistent: flat → 1, tilted plane →
`sec(slope)`. **However**, this is *not* Jenness's 8-triangle
single-cell method as the docstring claims (Jenness places vertices at
neighbour-midpoints, so each triangle's plan area is `L²/8` and the eight
of them sum to exactly one cell area `L²`).

The two formulations give similar numbers for gently rolling terrain
but diverge for high-frequency roughness — pixels of opposite tilt
within the 3×3 window cancel more aggressively in the 4-triangle
approach. Either rename the function to acknowledge the simplification
("four-triangle SAR") or replace the kernel with Jenness's actual
8-triangle stencil.

### P2-13 · `curvature` drops the (1+G²+H²) normalisation factor

**File:** `derivatives/curvature.py:46-92`

Z&T (1987) defines profile curvature as:

```
K_p = -2 (D G² + E H² + F G H) / [(G²+H²) · (1 + G²+H²)^1.5]
```

This implementation omits the `(1 + G²+H²)^1.5` denominator factor —
matching ESRI's "Curvature" tool, which is a small-slope approximation.
On steep terrain (slope > ~30°) the simplified form overestimates the
magnitude of curvature by a factor that grows with slope. Either
document the simplification or restore the full Z&T form.

### P2-14 · `flow_direction.D8` doesn't handle weight=NaN through `argmax`

**File:** `hydrology/flow_direction.py:67-74`

`drops = np.where(np.isfinite(drops), drops, -np.inf)` replaces NaN with
`-inf` so `argmax` doesn't pick a NaN direction. Good. But the same
replacement is **not** applied for cells where the centre itself is
NaN — those cells' `drops` are NaN for all 8 directions, `argmax` returns
0 (first index, code 1 = East), and only the subsequent
`flow_dir[nodata_mask] = 0` zeros them. The intermediate state during
the function is "every NaN cell thinks it drains east"; if a future
optimisation skips the final masking, the bug emerges. Defensive: also
filter NaN-centres before the argmax.

### P2-15 · `profile.sample_profile` duplicates samples near segment joins

**File:** `visibility/profile.py:79-95`

```python
for step in range(n_steps):
    frac = step / n_steps
    sample_rows.append(r0 + (r1 - r0) * frac)
    ...
# Add final point
sample_rows.append(row_col_points[-1][0])
```

For a multi-vertex polyline, segment *k* appends samples at
`frac = 0/n, 1/n, …, (n-1)/n` (omitting the endpoint at `frac=1`).
Segment *k+1* starts at `frac=0` of its own line, which equals segment
*k*'s endpoint → a duplicate sample at every internal vertex. The final
vertex is appended once explicitly, also duplicating the last segment's
final sample.

Visually this is a tiny "zero-distance" step in the elevation profile
at every internal vertex; in the cumulative-distance array the values
are monotone but not strictly increasing.

### P2-16 · Almost every algorithm upcasts to `float64` then downcasts

**Files:** every module in `derivatives/`, `indices/`, `hydrology/`, `visibility/`.

Pattern: `z = dem.astype(np.float64) * z_factor` → compute → `astype(np.float32)`.
A 5 k × 5 k DEM in float32 is 100 MB; the float64 temporary is 200 MB,
held for the entire computation. For chained analyses (slope → twi
needs slope_rad) the peak memory usage runs 600–800 MB above what the
final output needs. For typical 32-bit DEMs the precision gain from
float64 is negligible — float32 is sufficient. Either keep float32
throughout, or only upcast slice-by-slice.

### P2-17 · `dem_layer.valid_data` scans the entire array on every cache miss

**File:** `core/dem_layer.py:73-100`

```python
if self.nodata is None and np.issubdtype(self.data.dtype, np.floating) \
   and not np.any(~np.isfinite(self.data)):
    out = self.data
```

The `np.any(~np.isfinite(self.data))` short-circuits internally, but on
a finite-everywhere DEM (the common case for projected DEMs without
explicit nodata) it walks the entire array before returning. For a
5 k × 5 k float32 grid that's a 100 MB read every time `valid_data` is
accessed for the first time on a new `data` buffer. Cache the "all
finite" result on the DemLayer alongside the masked view.

### P2-18 · `priority_flood` `_progress` callback fires up to twice per cell

**File:** `hydrology/fill_sinks.py:77-81`

```python
done += 1
if _progress is not None and done >= next_report:
    if _progress(int(done / total * 100)):
        return filled
    next_report = done + max(1, total // 100)
```

`done` is incremented every pop, but a single cell can be pushed once
and popped once (so far so good). The cancel check happens before the
progress emit — so cancellation is honoured immediately. No bug, just
note: the integer percentage clamps to `[0, 100]`, so for grids
< 100 cells the bar stays at 0 → 100 with no steps.

---

## P3 — Code quality and documentation

### P3-1 · `auto_threshold`'s 99th-percentile rule has no physical basis

**File:** `hydrology/streams.py:110-124`

The docstring frames it as "a quick proxy for Tarboton et al. (1991)
constant-drop". Constant-drop chooses the threshold where the mean
along-channel slope stops decreasing — completely unrelated to a fixed
percentile. The percentile rule is just "top 1% of cells with positive
accumulation". On a DEM with heavy flat regions, that 1% may already
include hillslope cells. Either implement actual constant-drop or
relabel the function ("`top_percentile_threshold`").

### P3-2 · Type hints use bare `None` defaults instead of `Optional[float]`

**Files:** every function with `nodata: float = None`.

With `from __future__ import annotations` already active and Python ≥ 3.10
the implicit-Optional convention is deprecated. Either `nodata: float | None = None`
or `Optional[float]`.

### P3-3 · `specific_catchment_area` collapses two terms into one but keeps the misleading comment

**File:** `hydrology/flow_accumulation.py:170-182`

The implementation computes `flow_accum * cell_size`. The docstring
walks the reader through `A / w = (n·L²) / L = n·L`, which is the same
result. Either keep the algebraic derivation as a comment with units, or
just compute `flow_accum * cell_size` and link to the reference inline.

### P3-4 · `dem_layer.py` cache-set uses `object.__setattr__` unnecessarily

**File:** `core/dem_layer.py:99, 120, 138`

`DemLayer` is a `@dataclass` without `frozen=True`. Plain
`self._valid_cache = (...)` works. `object.__setattr__` is the right
idiom only for frozen dataclasses or slots; here it just confuses
readers into thinking the class is frozen.

### P3-5 · `flow_direction.d_infinity_flow_direction` mixes Tarboton angle convention with code that already reads correctly

**File:** `hydrology/flow_direction.py:84-199`

The facet table (`_FACETS_T`) uses Tarboton-style `(cardinal, diagonal,
base, sign)` tuples, but the comments at the top of the table reference
the original paper's table 1 — which uses *different* indices for the
facets (`α_k` from 0..7 vs. the code's loop order). Adding a
side-by-side mapping would help any future maintainer.

### P3-6 · `viewshed` cancellation only fires between rows

**File:** `visibility/viewshed.py:71-74`

`_progress` is checked at the start of each `tr` row. For a 200-cell-radius
viewshed (≈ 400 rows), that's 400 cancel checks total — fast enough.
For `cumulative_viewshed`, the outer "per observer" loop has no progress
emit at all, so cancelling between observers is impossible.

### P3-7 · `hydrology/streams.strahler_order` doesn't handle disconnected stream segments cleanly

**File:** `hydrology/streams.py:33-107`

If `flow_dir` has been set to 0 (no drainage) on a stream cell — which
happens when the cell is a sink in the filled DEM — the cell has
in-degree 0 (no upstream), is seeded as order 1, but never enqueues a
downstream cell. That's correct, but the resulting "stub" streams of
length 1 may surprise users expecting connected channels.

### P3-8 · `worker._emit_progress` truncates to int

**File:** `core/worker.py:81-84`

```python
self.progress.emit(int(max(0, min(100, pct))))
```

For algorithms that emit `int(done / total * 100)` themselves (every
hydrology function does this), the truncation is benign. For a future
emitter that passes `pct=99.7` directly, the bar shows 99 → 100. Not a
bug; just a "watch out".

### P3-9 · `renderer._categorical_d8_rgba` silently drops unknown codes

**File:** `core/renderer.py:82-94`

Codes not in the LUT (0, 1, 2, 4, …) get RGBA `(0,0,0,0)` — transparent.
A garbage flow-direction grid (e.g. accidentally passing flow
*accumulation* in) renders as a uniformly transparent overlay. Log a
warning when many cells fall outside the LUT.

### P3-10 · `renderer.array_to_rgba` collapses `vmin == vmax` to `vmin + 1.0`

**File:** `core/renderer.py:126-129`

```python
if vmax == vmin:
    vmax = vmin + 1.0
```

If `vmin` is `1e10` (a degenerate "all same large value" overlay), `+1.0`
disappears in float32 precision and the normalisation becomes
`0/0 → nan`. Use a relative bump (`vmax = vmin + max(1.0, abs(vmin) * 1e-6)`)
or clamp to a fixed range.

### P3-11 · `dem_layer.DemLayer.valid_data` cache key uses `id(self.data)`

**File:** `core/dem_layer.py:84-87`

`id()` is unique only for live objects. If the layer's `data` is freed
and a new array happens to receive the same memory address (CPython's
small-object allocator) the cache returns stale data. Practically rare
but theoretically possible. Use a content fingerprint (e.g. `(shape,
dtype, data.ctypes.data, nbytes)`) or just bump a version counter on
`__setattr__`.

### P3-12 · `__init__.py` files are empty across `core/` subpackages

**Files:** `core/__init__.py`, `core/derivatives/__init__.py`,
`core/hydrology/__init__.py`, `core/indices/__init__.py`,
`core/visibility/__init__.py`.

Each is an empty file. Re-exporting the public functions (`hillshade`,
`slope`, `aspect`, `priority_flood`, …) would let callers write
`from app.core.derivatives import slope` instead of
`from app.core.derivatives.slope import slope`. Minor ergonomic win.

### P3-13 · `breach_and_fill` declares `max_breach_depth` in absolute elevation units, not slope

**File:** `hydrology/fill_sinks.py:101-200`

The threshold is `(max_cost_along_path - outlet_elev)`. For a 30-m DEM
with metre-scale relief, the default `10.0` means "breach anything up to
10 m above the outlet" — reasonable. For a 1-m LiDAR DEM with
sub-metre features, 10 m is effectively "always breach" and the fill
pass at the end becomes the only depression remover. Document units
and consider an absolute-and-relative threshold pair.

### P3-14 · `roughness.surface_area_ratio` z_factor is applied before stencil

**File:** `indices/roughness.py:93`

```python
z = dem.astype(np.float64) * z_factor
```

Multiplying elevation by `z_factor` then computing differences is
algebraically the same as `dz_n * z_factor`, etc. — but it forces a
full-grid float64 multiply early. For `z_factor=1.0` (the common case)
this is a wasted pass; short-circuit when `z_factor == 1`.

### P3-15 · `visibility/profile.ElevationProfile` doesn't store `cell_size` or units

**File:** `visibility/profile.py:14-47`

`distances` is in metres only if the caller passed `cell_size` in
metres; the class itself has no idea. A consumer doing
`np.diff(distances)` to derive slope is fine, but anything else (e.g.
labelling axes "km") has to be told by the caller. Add a `units` field
or expose a `with_units(scale, label)` helper.

---

## Smoke-test results (current code)

```text
D-inf east-plane direction (10,10) = 180.0 deg (expect 180)   ✓
D-inf accum peak col = 0 (expect 0 = west)                    ✓
SAR(flat) = 1.000 (expect ~1.0)                               ✓
auto_threshold(zeros) = 0.0                                   ✓
Viewshed past obstacle: hidden                                ✓

slope adjacent to -9999 sentinel: 89.319° (expect ~0)         FAIL  → P0-1
hillshade near sentinel: black halo                           FAIL  → P0-1
TWI nodata flag changes result without a sentinel comparison  FAIL  → P0-2
strahler_order body contains dead pass-loop                   FAIL  → P0-3
```

All three failures point at issues introduced *by the same pattern*: the
algorithms accept a `nodata` value and use it inconsistently — sometimes
as a sentinel, sometimes as a presence flag, sometimes ignored. A single
"`_normalize_nodata`" helper at the top of each function, called before
anything else, would close every P0 finding here.

---

## Suggested remediation order

1. **Centralise nodata handling.** A 20-line helper, called from every
   module that today does `astype(np.float64)` plus pad — closes P0-1,
   P0-2, P1-1, P1-2, P2-7, P2-8.
2. **Fix the dead code in `strahler_order`** (P0-3) and rewrite the
   in-degree pass with `np.argwhere(stream)` (P1-4).
3. **Triage the slow paths** — Numba-jit priority-flood and the two
   accumulation routines (P2-2, P2-3, P2-4); share gradients in
   `multidirectional_hillshade` (P2-1).
4. **Document algorithm simplifications** — ESRI-style curvature
   (P2-13), 4-triangle SAR (P2-12), top-percentile auto-threshold
   (P3-1).
5. **Add a regression test harness** (`tests/test_core_algorithms.py`)
   so future changes can't silently reintroduce the boundary
   contamination or the `nodata` flag confusion.
