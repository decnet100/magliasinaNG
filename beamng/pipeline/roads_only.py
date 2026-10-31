"""The road network meshes alone, without the level (an external site, or a quick check): the network part of
build_level.stage_roads.

Runs network_mesh (every tile), its terrain carve and bridges.build on the data of network.py and
network_surface.py, as stage_roads does, then road_seams' ramps (seam_ramps: the finish step of a level build), and
the finish steps unpaved and paved_edges on a small level folder (finish_edges: WORK/roads_only/level, only what they
read; keep-out shapes from DATA/keep_out/<kind>.geojson), but writes into WORK/roads_only/ instead of a level:
* road_<tx>_<ty>.dae - the meshes per 128 m chunk (Z-up COLLADA, local coordinates + origin), and .npz with the
  same triangles per material in local coordinates (materials, v<i>, t<i>, uv<i>), for a tool that writes its own;
* tops.tif - the height of the highest road top face on a 1 m grid, in the map CRS (NaN off the roads), and
  tops_minus_dtm.tif - that height minus the DTM, both for QGIS;
* terrain_final.npz - the terrain block after the carve and the edge steps (z), the DTM on the same vertices (dtm),
  x0, y0, square; rows south -> north;
* report.json - triangles, chunks, bridges, carved terrain vertices, and how far the tops sit from the DTM.
The Strada Cantonale corridor of the Malcantone build is not meshed here (exclude=None).

    python roads_only.py
"""
import json, os, time
from shutil import copyfile as shutil_copy
import numpy as np
import shapely
from config import CRS, WORK
import bng
import network_mesh
import road_mesh
import bridges
import terrain
from geo import Grid, save_geotiff, smoothed_sampler

CHUNK = 128.0
OUT = os.path.join(WORK, "roads_only")
TOP_RES = 1.0


SEAM_MATS = ("mp_road_asphalt", "mp_road_asphalt_fresh", "mp_road_sett", "mp_road_cobble",
             "mp_road_dirt", "mp_road_gravel", "mp_path_paved", "mp_path_dirt", "mp_hard_asphalt")


def seam_ramps(builders):
    """road_seams.py's step (the finish step of a level build) on the chunk meshes before they are written: every
    outline edge of a road face with a higher road face just beyond it (1.5-30 cm) raises the lower piece into a
    ramp (12 m per m of step, 0.75-3 m, cosine). Unlike the level step, the unpaved road and path surfaces count
    too (a way joining a road is one of them), and the sags of the AI lanes are not lifted (no AI roads here).
    Every vertex of the chunks at the place of a raised corner goes up with it (skirts, wall tops)."""
    import road_seams
    tops = []
    for mb in builders.values():
        for mat, p in mb.parts.items():
            if mat not in SEAM_MATS:
                continue
            V = np.concatenate(p[0])
            T = V[np.concatenate(p[3])]
            n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            up = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12) > 0.5
            tops.append(T[up])
    T = np.concatenate(tops)
    P = T.reshape(-1, 3)
    root = road_seams._weld(P)
    sx, snrm, sdz, sR, splane = road_seams.seams(T, root, road_seams.Faces(T))
    lift = road_seams.lifts(P, sx, snrm, sdz, sR, splane)
    Pi = P + np.column_stack([np.zeros((len(P), 2)), lift])
    left = road_seams.seams(Pi.reshape(-1, 3, 3), root, road_seams.Faces(Pi.reshape(-1, 3, 3)))[2]
    raised = lift > 0
    key = {tuple(k): v for k, v in zip(np.round(P[raised] * 1000).astype(np.int64).tolist(), lift[raised])}
    best = {}
    for k_, v in key.items():
        best[k_] = max(best.get(k_, 0.0), v)
    moved = 0
    for mb in builders.values():
        for p in mb.parts.values():
            for V in p[0]:
                k = np.round(V * 1000).astype(np.int64)
                dz = np.array([best.get(tuple(r), 0.0) for r in k.tolist()]) if best else np.zeros(len(V))
                if (dz > 0).any():
                    V[:, 2] += dz
                    moved += int((dz > 0).sum())
    sm = road_seams.SAMPLE
    rep = {"seam_m": round(len(sx) * sm, 1),
           "step_cm_p50": round(float(np.median(sdz)) * 100, 1) if len(sdz) else 0.0,
           "step_cm_max": round(float(sdz.max()) * 100, 1) if len(sdz) else 0.0,
           "seam_m_left": round(len(left) * sm, 1), "seam_m_left_over_3cm": round(float((left >= 0.03).sum()) * sm, 1),
           "vertices_raised": moved, "lift_cm_max": round(float(lift.max(initial=0)) * 100, 1)}
    print("seam ramps:", rep, flush=True)
    return rep



KEEP_FILES = {"guardrails": "roads/guardrails", "fences": "roads/fences", "walls": "walls", "buildings": "buildings",
              "railway": "railway", "water": "level_objects/Water"}


def keep_out_shapes(lv_dir):
    """DATA/keep_out/<kind>.geojson (map CRS; kinds as KEEP_FILES) as flat triangle shapes in the groups
    network_mesh.keep_out_cells reads: what marks a road's edge (no ground raised to the road there)."""
    from shapely.geometry import shape
    from config import DATA, lv95_to_local
    out = {}
    for kind, group in KEEP_FILES.items():
        f = os.path.join(DATA, "keep_out", f"{kind}.geojson")
        if not os.path.exists(f):
            continue
        tris = []
        for feat in json.load(open(f, encoding="utf-8"))["features"]:
            g = shape(feat["geometry"])
            g = shapely.transform(g, lambda c: np.column_stack(lv95_to_local(c[:, 0], c[:, 1])))
            if g.geom_type in ("LineString", "MultiLineString", "Point", "MultiPoint"):
                g = g.buffer(0.15)
            for t in getattr(shapely.constrained_delaunay_triangles(g), "geoms", []):
                tris.append(np.asarray(t.exterior.coords)[:3])
        if not tris:
            continue
        T = np.concatenate(tris)
        V = np.column_stack([T, np.zeros(len(T))])
        mb = bng.MeshBuilder()
        mb.add("keep_out", V)
        rel = f"art/shapes/keep_out/{kind}.dae"
        mb.write_dae(os.path.join(lv_dir, rel), name=f"keep_{kind}")
        out[group] = rel
        print("keep-out %s: %d triangles" % (kind, len(T) // 3), flush=True)
    return out


def finish_edges(builders, dtm, ov, cap, xs, ys):
    """Magliasina's finish steps unpaved and paved_edges (network_mesh.unpaved_step, paved_edges_step) on a small
    level folder that holds only what they read: the road chunks (roads/surfaces), the terrain as it is after the
    carve (theTerrain.ter + TerrainBlock), and the keep-out shapes (keep_out_shapes). They raise the ground to the
    road edges (embankments, no trench beside a road) except where something marks the edge, and lower the outer
    edges of the gravel and dirt faces onto the ground. Returns the final terrain heights (rows south -> north) and
    the folder; the chunks are read back from it."""
    import shutil, struct
    from config import LEVEL_NAME
    root = os.path.join(OUT, "level")
    shutil.rmtree(root, ignore_errors=True)
    lv_dir = os.path.join(root, "levels", LEVEL_NAME)
    os.makedirs(lv_dir)
    items = []
    for (tx, ty), mb in sorted(builders.items()):
        origin = np.array([(tx + 0.5) * CHUNK, (ty + 0.5) * CHUNK, 0.0])
        rel = f"art/shapes/roads/road_{tx:+03d}_{ty:+03d}.dae"
        mb.write_dae(os.path.join(lv_dir, rel), name=f"road_{tx}_{ty}", origin=origin)
        items.append(bng.tsstatic(f"/levels/{LEVEL_NAME}/{rel}", origin, collision=True, decal=True))
    # terrain after the carve, on the terrain block
    H = np.empty((len(ys), len(xs)), np.float64)
    for r0 in range(0, len(ys), 512):
        X, Y = np.meshgrid(xs, ys[r0:r0 + 512])
        H[r0:r0 + 512] = dtm.sample(X.ravel(), Y.ravel()).reshape(X.shape)
    base = H.astype(np.float32)                    # the DTM on the terrain vertices, before anything
    H = np.fmin(H, ov)
    H = np.minimum(H, cap)
    z0 = float(np.floor(np.nanmin(H))) - 1.0
    maxh = float(np.ceil(np.nanmax(H) - z0)) + 1.0
    q = np.clip(np.floor((np.nan_to_num(H, nan=z0) - z0) / maxh * 65535.0), 0, 65535).astype("<u2")
    with open(os.path.join(lv_dir, "theTerrain.ter"), "wb") as f:   # bng.write_ter's layout, heights floored
        f.write(bytes([9]) + struct.pack("<I", q.shape[0]) + q.tobytes() + bytes(q.size)
                + struct.pack("<I", 1) + bytes([5]) + b"Grass")
    groups = {"roads/surfaces": items,
              "level_objects/terrain": [{"name": "theTerrain", "class": "TerrainBlock", "position": [0, 0, z0],
                                         "maxHeight": maxh, "terrainFile": f"/levels/{LEVEL_NAME}/theTerrain.ter"}]}
    for group, rel in keep_out_shapes(lv_dir).items():
        groups[group] = [bng.tsstatic(f"/levels/{LEVEL_NAME}/{rel}", np.zeros(3), collision=False)]
    for group, objs in groups.items():
        d = os.path.join(lv_dir, "main", "MissionGroup", *group.split("/"))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "items.level.json"), "w") as f:
            f.write("".join(json.dumps(o) + "\n" for o in objs))
    t0 = time.time()
    network_mesh.unpaved_step(root)
    network_mesh.paved_edges_step(root, report=os.path.join(OUT, "paved_edges.json"))
    import network_mesh as pu
    q1 = pu.read_ter(open(os.path.join(lv_dir, "theTerrain.ter"), "rb").read())[1].astype(np.float64)
    changed = q1 != q.astype(np.float64)
    final = H.copy()
    final[changed] = z0 + q1[changed] / 65535.0 * maxh
    print("edge steps: %d terrain vertices raised, %.0f s" % (int(changed.sum()), time.time() - t0), flush=True)
    return final, base, root, lv_dir, int(changed.sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    dtm = Grid.load(os.path.join(WORK, "dtm05.npz"))
    ground = smoothed_sampler(dtm, 0.6)
    xs, ys = terrain.vertex_coords()
    ov = np.full((len(ys), len(xs)), np.nan, np.float32)
    cap = np.full((len(ys), len(xs)), np.inf, np.float32)
    builders, chunk_tops, stats = {}, {}, {"stone faces": 0}

    def add_surface(tx, ty, mat, uvt, V, T, S_, ground_, tops=None):
        mb = builders.setdefault((tx, ty), bng.MeshBuilder())
        mb.add(mat, V, uvs=V[:, :2] / uvt, tris=T)
        if tops is not None:
            tops.append(V[T])
        kerb, wall = road_mesh.skirt_bands(V, T, road_mesh.skirt_depth(V, T, S_, ground_))
        mb.add(mat, kerb, uvs=bng.box_uvs_soup(kerb, uvt), normals=bng.flat_normals_soup(kerb))
        if len(wall):
            mb.add("mp_road_wall", wall, uvs=bng.box_uvs_soup(wall, 1.6), normals=bng.flat_normals_soup(wall))
            stats["stone faces"] += len(wall) // 6

    def on_tops(cx, cy, tri):
        chunk_tops.setdefault((cx, cy), []).append(tri)

    def on_carve(r, c, z):
        ov[r, c] = np.fmin(ov[r, c], z)

    def on_cap(r, c, z):
        cap[r, c] = np.minimum(cap[r, c], z)

    def on_deck(x, y, mat, uvt, soup, kind):
        tx, ty = int(np.floor(x / CHUNK)), int(np.floor(y / CHUNK))
        mb = builders.setdefault((tx, ty), bng.MeshBuilder())
        V = soup.reshape(-1, 3)
        uv = V[:, :2] / uvt if kind == "top" else bng.box_uvs_soup(V, uvt)
        mb.add(mat, V, uvs=uv, normals=bng.flat_normals_soup(V))
        if kind == "top":
            chunk_tops.setdefault((int(round(x // CHUNK)), int(round(y // CHUNK))), []).append(V.reshape(-1, 3, 3))

    t0 = time.time()
    net = network_mesh.Network(exclude=None)
    try:
        net.assign_surfaces()
    except Exception as e:                       # OSM surfaces: none outside the Malcantone extract
        print("assign_surfaces skipped:", e, flush=True)
    tiles = network_mesh.tiles()
    ntri = 0
    for k, (X0, Y0) in enumerate(tiles):
        ntri += network_mesh.mesh_tile(net, dtm, X0, Y0, xs, ys, add_surface, on_tops)
        if k % 10 == 0:
            print("  tiles %d/%d, %d triangles, %.0f s" % (k + 1, len(tiles), ntri, time.time() - t0), flush=True)
    per = int(round(network_mesh.TILE / CHUNK))
    for X0, Y0 in tiles:
        cx0, cy0 = int(round(X0 / CHUNK)), int(round(Y0 / CHUNK))
        near = [t for cx in range(cx0 - 1, cx0 + per + 1) for cy in range(cy0 - 1, cy0 + per + 1)
                for t in chunk_tops.get((cx, cy), [])]
        if near:
            network_mesh.carve_tile(net, X0, Y0, xs, ys, road_mesh.TriSurface(np.concatenate(near)), on_carve)
    print("carve done, %.0f s" % (time.time() - t0), flush=True)
    print("bridges ...", flush=True)
    recs = bridges.build(net, lambda x, y: dtm.sample(x, y), on_deck, on_cap, xs, ys, exclude=None)
    bridges.save_records(recs)
    # the ramps first (the level build runs road_seams after paved_edges): the ground then rises to the final faces
    seams = seam_ramps(builders)
    print("bridges %d, %d chunks, %.0f s" % (len(recs), len(builders), time.time() - t0), flush=True)
    final, base, root, lv_dir, n_raised = finish_edges(builders, dtm, ov, cap, xs, ys)
    # the chunks as the finish steps left them: DAEs and per material triangles (local coordinates)
    import optimize_level
    for f in sorted(os.listdir(os.path.join(lv_dir, "art", "shapes", "roads"))):
        text = open(os.path.join(lv_dir, "art", "shapes", "roads", f), encoding="utf-8").read()
        tx, ty = (int(v) for v in f[5:-4].split("_"))
        origin = np.array([(tx + 0.5) * CHUNK, (ty + 0.5) * CHUNK, 0.0])
        V, N, UV, C, parts, node = optimize_level.parse(text)
        arrays, mats = {}, []
        for i, (mat, idx) in enumerate(parts):
            mats.append(mat)
            vi = idx[:, 0]
            arrays[f"v{i}"] = (V[vi] + origin).astype(np.float64)
            arrays[f"t{i}"] = np.arange(len(vi), dtype=np.int32).reshape(-1, 3)
            arrays[f"uv{i}"] = (UV[idx[:, 1]] if idx.shape[1] > 1 else UV[vi]).astype(np.float32)
        np.savez_compressed(os.path.join(OUT, f[:-4] + ".npz"), materials=np.array(mats), **arrays)
        shutil_copy(os.path.join(lv_dir, "art", "shapes", "roads", f), os.path.join(OUT, f))

    # top heights on a grid, for a look in QGIS
    tops = [t for v in chunk_tops.values() for t in v]
    surf = road_mesh.TriSurface(np.concatenate(tops))
    x0, y0, x1, y1 = (min(t[:, :, 0].min() for t in tops), min(t[:, :, 1].min() for t in tops),
                      max(t[:, :, 0].max() for t in tops), max(t[:, :, 1].max() for t in tops))
    gx = np.arange(x0, x1, TOP_RES) + TOP_RES / 2
    gy = np.arange(y1, y0, -TOP_RES) - TOP_RES / 2
    top = np.full((len(gy), len(gx)), np.nan, np.float32)
    diff = np.full((len(gy), len(gx)), np.nan, np.float32)
    for r0 in range(0, len(gy), 256):            # bands: the whole grid at once does not fit in memory
        X, Y = np.meshgrid(gx, gy[r0:r0 + 256])
        h = np.asarray(surf.height(X.ravel(), Y.ravel(), mode="high"), np.float32).reshape(X.shape)
        top[r0:r0 + 256] = h
        ok = np.isfinite(h)
        if ok.any():
            band = diff[r0:r0 + 256]
            band[ok] = h[ok] - dtm.sample(X[ok], Y[ok]).astype(np.float32)
    save_geotiff(os.path.join(OUT, "tops.tif"), top[None], x0, y1, TOP_RES)
    save_geotiff(os.path.join(OUT, "tops_minus_dtm.tif"), diff[None], x0, y1, TOP_RES)
    d = diff[np.isfinite(diff)]
    rep = {"crs": CRS, "chunks": len(builders), "triangles": int(sum(mb.triangle_count() for mb in builders.values())),
           "network_triangles": int(ntri), "stone_faces": stats["stone faces"], "bridges": len(recs),
           "carved_terrain_vertices": int(np.isfinite(ov).sum()), "capped_terrain_vertices": int(np.isfinite(cap).sum()),
           "road_cells": int(d.size), "tops_minus_dtm_m": {q: round(float(np.percentile(d, q)), 3) for q in (1, 5, 50, 95, 99)},
           "seam_ramps": seams, "edge_steps_terrain_vertices_raised": n_raised, "seconds": round(time.time() - t0)}
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "terrain_override.npz"), ov=ov, cap=cap)
    # the terrain block after the carve and the edge steps (rows south -> north, xs/ys of terrain.vertex_coords):
    # an importer takes it where it differs from the DTM
    np.savez_compressed(os.path.join(OUT, "terrain_final.npz"), z=final.astype(np.float32), dtm=base, x0=xs[0],
                        y0=ys[0], square=xs[1] - xs[0])
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
