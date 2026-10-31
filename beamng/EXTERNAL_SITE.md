# Road network of an external site (trial)

The road part of the pipeline (network.py, network_surface.py, network_mesh.py, bridges.py) can run on a site
outside Switzerland whose network was exported by another tool, in the shape this pipeline reads (terrainWorker's
`export_magliasina.py`: `tlm/TLM_STRASSE.json`, `av/*.geojson` with `Genere`, `manifest.json`). First site:
Pfunds–Spiss (Tirol / Graubünden, EPSG:25832). Without the variables below every module behaves as before.

## Switches (config.py)

| Variable | Meaning |
|----------|---------|
| `MAGLIASO_EXTERNAL=1` | no Street View route, no Strada Cantonale corridor (`network_surface.cantonale_hold` returns nothing) |
| `MAGLIASO_CRS` | map CRS (default EPSG:2056); `lv95_*` functions and `geo` work in it |
| `MAGLIASO_ORIGIN`, `MAGLIASO_K` | local origin "E0,N0" and the CRS scale factor there |
| `MAGLIASO_AREA` | GeoJSON polygon of the area, in the map CRS (`area.polygon`) |
| `MAGLIASO_PONTI` | the bridge list (default `dati/ponti.json`); an external site keeps its own |
| `MAGLIASO_TER_SIZE`, `MAGLIASO_TER_SQUARE`, `MAGLIASO_TER_ORIGIN` | terrain block |
| `MAGLIASO_LEVEL_NAME` | level name |
| `MAGLIASO_EMB_MAX`, `MAGLIASO_EMB_SLOPE` | embankment of the edge steps (network_mesh.EMB_*; default 3 m at 1:1.5; prepare_external sets 1 m at 1:1) |

## Run

```bash
python prepare_external.py env EXPORT_DIR > external.env
```

```bash
source external.env && python prepare_external.py data EXPORT_DIR DGM_05.tif
```

```bash
source external.env && python landcover.py && python network.py && python network_surface.py && python roads_only.py
```

`prepare_external.py` copies the lines and polygons into `DATA`, builds `dtm05`, `dsm05` (the DTM when no DOM is
given) and `dtm2` from the 0.5 m model (NaN filled from the nearest cell; no Copernicus outside Switzerland) and an
empty bridge list. `roads_only.py` is the network part of `build_level.stage_roads` without the level: chunk DAEs,
the terrain carve, the bridge decks, and `tops.tif` / `tops_minus_dtm.tif` (1 m, map CRS) plus `report.json`.
It also writes the triangles of every chunk per material as `road_<tx>_<ty>.npz` (local coordinates) and the carve
and deck caps as `terrain_override.npz`: terrainWorker's `import_magliasina_roads.py` puts those into its own level.
Needs `opencv-python-headless` (lamps.py, imported through water.py by network.dry_ends).

## Pfunds–Spiss, first run (2026-10-30)

- network.py: 400 segments, 435 nodes (82 junctions), 48.9 km road + 23.9 km path (Swiss TLM ways), 37 bridges.
- network_surface.py: |z − DTM| median 1.2 cm, p99 0.53 m; 359 bridge stations; 1.8 % of the data rejected.
- roads_only.py (2026-10-30, later): seam ramps on the chunk meshes before writing (road_seams' step; the unpaved road
  and path surfaces count too): 2.75 km of seams with 1.5-30 cm steps (median 2 cm) -> 462 m left, 206 m of them over
  3 cm (corners where three pieces meet; a second pass is not run: in the Malcantone it stacked ramps up to 78 cm).
- roads_only.py, 69 s: 613 chunks, 1.30 M triangles (0.48 M network tops), 6,369 stone faces, 545,573 terrain
  vertices carved. Road top minus DTM on 1 m cells: median 0.003 m, p5/p95 −0.10/+0.12 m, p1 −0.59 m, p99 +2.79 m
  (bridge decks). The cells beyond ±0.3/0.5 m (about 2 % each) lie along road edges on steep slopes, where a 1 m
  cell holds road and bank.

## Edge steps (unpaved, paved_edges)

`roads_only.py` runs the two finish steps on a small level folder (`WORK/roads_only/level`: the road chunks in
`roads/surfaces`, `theTerrain.ter` with the terrain after the carve and its TerrainBlock, and the keep-out shapes)
so `network_mesh.unpaved_step` and `paved_edges_step` run unchanged. The keep-out comes from
`DATA/keep_out/<kind>.geojson` (guardrails, fences, walls, buildings, railway, water; terrainWorker's
`export_keep_out.py`) as flat triangles in the groups `keep_out_cells` reads. The seam ramps run first, so the ground
rises to the final faces. The chunks are read back from the folder (`road_*.dae` / `.npz`) and the terrain goes to
`terrain_final.npz` (z after the steps, dtm before anything; rows south → north).

Pfunds–Spiss (2026-10-31), keep-out: 731 buildings, 74 water features, no guard rails yet: paved outer edges over
the ground 0.3 m beyond them, median 0.20 m (81 % over 0.15 m) → 0.07 m (8 % over 0.15 m). With Magliasina's
embankment (3 m at 1:1.5) slopes steeper than 34° were filled 2-3 m deep up to 25 m from the road (31,525 px over
1.5 m in the 1.5 km test crop); with 1 m at 1:1 (prepare_external's default here): 3 px, raised median 0.30 m.
The terrain of the edge steps is for the game; a rockfall model needs the plain DGM.

## Not done

- `build_level.py` (the full level) is not adapted: stage_roads still meshes the corridor of `roadheight.py`, and
  the stages after it read the Street View corridor and `poses.json`. `ai_roads.build` still builds the Strada
  Cantonale route. No OSM extract for the site (`osm_surface` falls back to BELAGSART).
- No in-game test, no drive_test.py (needs the level).
