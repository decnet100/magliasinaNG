"""Bring a road network exported by another tool (terrainWorker's export_magliasina.py) into this pipeline.

Two calls. First, without the environment set, the variables the site needs (from the export's manifest.json):

    python prepare_external.py env EXPORT_DIR [MARGIN_M] > external.env      (then: source external.env)

Then, with them set, the data:

    python prepare_external.py data EXPORT_DIR DGM_05 [DOM_05]

* DATA/tlm/TLM_STRASSE.json and DATA/av/{LCSF,SOSF,SOLI,SOPT}.geojson from the export;
* DATA/ext_dgm05/ and DATA/ext_dom05/ with the 0.5 m terrain (and surface) model, already in the map CRS;
* WORK/dtm05.npz, dtm05_nodata.npy, dsm05.npz (the DTM where no DOM is given) on area.raster_bounds,
  WORK/dtm2.npz on area.block_bounds; NaN filled from the nearest cell (no Copernicus outside Switzerland);
* MAGLIASO_PONTI: an empty bridge list if there is none yet (bridges.py writes it on the first build).
The area (MAGLIASO_AREA) is the export's bbox less MARGIN_M (default 100 m), written beside the export.
After this, landcover.py, network.py, network_surface.py and roads_only.py run as usual.
"""
import json, os, shutil, sys


def env(export, margin=100.0):
    m = json.load(open(os.path.join(export, "manifest.json"), encoding="utf-8"))
    x0, y0, x1, y1 = m["bbox"]
    area = os.path.join(export, "area.geojson")
    ring = [[x0 + margin, y0 + margin], [x1 - margin, y0 + margin], [x1 - margin, y1 - margin],
            [x0 + margin, y1 - margin], [x0 + margin, y0 + margin]]
    json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
               "geometry": {"type": "Polygon", "coordinates": [ring]}}]}, open(area, "w"))
    e0, n0 = m["origin"]
    k = m.get("scale_factor_at_origin") or 1.0
    # terrain block: the bbox at 1 m (BeamNG wants a power of two)
    size = 1
    while size < max(x1 - x0, y1 - y0):
        size *= 2
    ter_x0 = (x0 - e0) / k
    ter_y0 = (y0 - n0) / k
    root = os.path.join(export, "magliasina_work")
    lines = {"MAGLIASO_EXTERNAL": "1", "MAGLIASO_CRS": m["crs"], "MAGLIASO_ORIGIN": f"{e0},{n0}", "MAGLIASO_K": repr(k),
             "MAGLIASO_AREA": area, "MAGLIASO_PONTI": os.path.join(root, "ponti.json"),
             "MAGLIASO_TER_SIZE": str(size), "MAGLIASO_TER_SQUARE": "1.0", "MAGLIASO_TER_ORIGIN": f"{ter_x0:.1f},{ter_y0:.1f}",
             "MAGLIASO_LEVEL_NAME": m.get("site", "external").replace("-", "_"), "MAGLIASO_ROOT": root,
             "MAGLIASO_DATASET": root, "MAGLIASO_BEAMNG_USER": os.path.join(root, "beamng_user"),
             "MAGLIASO_BEAMNG_GAME": "/nonexistent"}
    for k_, v in lines.items():
        print(f"export {k_}='{v}'")


def data(export, dgm, dom=None):
    import numpy as np
    from scipy import ndimage as ndi
    from config import DATA, PONTI, WORK
    from geo import Grid, mosaic
    from rasterio.warp import Resampling
    import area
    for sub in ("tlm", "av"):
        os.makedirs(os.path.join(DATA, sub), exist_ok=True)
    shutil.copy(os.path.join(export, "tlm", "TLM_STRASSE.json"), os.path.join(DATA, "tlm", "TLM_STRASSE.json"))
    for name in ("LCSF", "SOSF", "SOLI", "SOPT"):
        shutil.copy(os.path.join(export, "av", f"{name}.geojson"), os.path.join(DATA, "av", f"{name}.geojson"))
    for d, src in (("ext_dgm05", dgm), ("ext_dom05", dom)):
        if src:
            os.makedirs(os.path.join(DATA, d), exist_ok=True)
            dst = os.path.join(DATA, d, os.path.basename(src))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
    if not os.path.exists(PONTI):
        os.makedirs(os.path.dirname(PONTI), exist_ok=True)
        json.dump({"ponti": []}, open(PONTI, "w"))

    def filled(a):
        bad = np.isnan(a)
        if bad.any():
            idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
            a = a[tuple(idx)]
        return a, bad

    x0, y0, x1, y1 = area.raster_bounds(0.5)
    dtm = mosaic("ext_dgm05/*.tif", x0, y0, x1, y1, 0.5)[0]
    dtm, bad = filled(dtm)
    print("dtm05", dtm.shape, "filled %.4f" % bad.mean(), flush=True)
    Grid(dtm, x0, y1, 0.5).save(os.path.join(WORK, "dtm05.npz"))
    np.save(os.path.join(WORK, "dtm05_nodata.npy"), bad)
    if dom:
        dsm, _ = filled(mosaic("ext_dom05/*.tif", x0, y0, x1, y1, 0.5)[0])
    else:
        dsm = dtm
    Grid(np.asarray(dsm, np.float32), x0, y1, 0.5).save(os.path.join(WORK, "dsm05.npz"))
    del dtm, dsm
    bx0, by0, bx1, by1 = area.block_bounds(2.0)
    d2, _ = filled(mosaic("ext_dgm05/*.tif", bx0, by0, bx1, by1, 2.0, resampling=Resampling.average)[0])
    Grid(d2, bx0, by1, 2.0).save(os.path.join(WORK, "dtm2.npz"))
    print("dtm2", d2.shape, "done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "env":
        env(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 100.0)
    else:
        data(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
