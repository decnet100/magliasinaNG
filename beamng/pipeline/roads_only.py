"""The road network meshes alone, without the level (an external site, or a quick check): the network part of
build_level.stage_roads.

Runs network_mesh (every tile), its terrain carve and bridges.build on the data of network.py and
network_surface.py, as stage_roads does, but writes into WORK/roads_only/ instead of a level:
* road_<tx>_<ty>.dae - the meshes per 128 m chunk (Z-up COLLADA, local coordinates + origin), and .npz with the
  same triangles per material in local coordinates (materials, v<i>, t<i>, uv<i>), for a tool that writes its own;
* tops.tif - the height of the highest road top face on a 1 m grid, in the map CRS (NaN off the roads), and
  tops_minus_dtm.tif - that height minus the DTM, both for QGIS;
* report.json - triangles, chunks, bridges, carved terrain vertices, and how far the tops sit from the DTM.
The Strada Cantonale corridor of the Malcantone build is not meshed here (exclude=None).

    python roads_only.py
"""
import json, os, time
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
    for (tx, ty), mb in sorted(builders.items()):
        # the triangles as they are (local coordinates), for a tool that writes its own meshes (terrainWorker)
        arrays, mats = {}, []
        for i, (mat, p) in enumerate(mb.parts.items()):
            mats.append(mat)
            arrays[f"v{i}"] = np.concatenate(p[0]).astype(np.float64)
            arrays[f"t{i}"] = np.concatenate(p[3]).astype(np.int32)
            arrays[f"uv{i}"] = np.concatenate(p[2]).astype(np.float32)
        np.savez_compressed(os.path.join(OUT, f"road_{tx:+03d}_{ty:+03d}.npz"), materials=np.array(mats), **arrays)
        origin = np.array([(tx + 0.5) * CHUNK, (ty + 0.5) * CHUNK, 0.0])
        mb.write_dae(os.path.join(OUT, f"road_{tx:+03d}_{ty:+03d}.dae"), name=f"road_{tx}_{ty}", origin=origin)

    print("bridges %d, writing %d chunks, %.0f s" % (len(recs), len(builders), time.time() - t0), flush=True)
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
           "seconds": round(time.time() - t0)}
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "terrain_override.npz"), ov=ov, cap=cap)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
