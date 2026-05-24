# DEM Analyst -- Benchmark results

Median wall-time (ms) of 3 runs per routine, per DEM size.  Refresh 
by re-running ``python -m tests.bench.run`` after each optimization 
commit so we have a measured trail of wins through Phases A -> C of 
OPTIMIZATION.md.

| Routine | 512^2 |
|---|---|
| hillshade | 20.4 |
| slope | 6.9 |
| aspect | 19.4 |
| profile_curv | 27.1 |
| tri | 16.1 |
| roughness_sd | 16.9 |
| vrm | 42.2 |
| multidirectional | 28.3 |
| svf | 2995.4 |
| priority_flood | 1444.3 |
| breach_and_fill | 19707.1 |
| d8_flow_dir | 90.7 |
| d_inf_flow_dir | 249.8 |
| d8_accum | 387.3 |
| d_inf_accum | 1159.1 |
