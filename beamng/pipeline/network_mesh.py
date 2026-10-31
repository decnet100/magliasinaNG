"""Drivable meshes of the whole road and path network (v2.0), on the heights of network_surface.py.

Drivable polygons (away from the Strada Cantonale corridor, which keeps the v1.1 meshes):
- the surveyed carriageways, sidewalks, islands (MU 'strada_sentiero', 'marciapiede',
  'spartitraffico') and the hard surfaces that touch them near the network ('altro_rivestimento_duro':
  yards, car parks, squares);
- a strip of the line's width along every swissTLM3D line outside those (forest roads, farm
  tracks, paths, mule tracks, stairs), asphalt or gravel after the line's surface;
- not the bridges: their decks are meshed by bridges.py.
Heights, per 0.5 m cell of a tile: every polygon belongs to the lines that run in it (a strip to
its own line and the lines at its ends); a cell takes the height of those lines at its projection
on them (linear between stations) and across the cross slope, blended between lines that agree
within BLEND_DZ; where they disagree (two roads at different levels, a wall between) the nearest
line wins and a step remains. v2.2: lines that meet at a node near the cell are one junction and
always blend (fading out between JUNCTION_R0 and JUNCTION_R1 m from the node), so a side road does
not end in a step against the road it joins. Cells far from any line (squares, car parks) get the
smoothed DTM corrected harmonically to meet the roads around them. v2.2: neighbouring cells of
different polygons (carriageway, sidewalk, yard) are one surface where their heights agree, and
where the height passes from one line to another, from a polygon to the next or to a filled
square, the cells within SEAM_R m of the seam are relaxed (Laplace over the linked cells), so the
passage is a ramp and not a kerb a car hits.
The cells of a tile make a surface_fit.Surface, so the meshing, the skirts and stone faces at the
steps and the terrain carve are the ones of v1.1 (road_mesh.py). Tiles of TILE m (whole 128 m
chunks) with MARGIN m of context.
"""
import os, pickle
import numpy as np
import shapely
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from rasterio import features
from rasterio.transform import Affine
from config import WORK
from geo import Grid
import area
import network
import network_surface
import surface_fit
import json, os, re, shutil, struct, sys, tempfile, time, zipfile
from concurrent.futures import ProcessPoolExecutor
import bng
import optimize_level
import osm_surface
import road_mesh
from config import LEVEL_NAME, TER_X0, TER_Y0, TER_SQUARE
import argparse, json, os, sys, zipfile
import lamps as pl
import network_mesh as pu
import walls as pw
from config import LEVEL_NAME

TILE = 512.0
MARGIN = 24.0
RES = 0.5
CHUNK = 128.0
STEP_Z = 0.30          # m, neighbouring cells further apart: a step (wall)
BLEND_DZ = 0.40        # m, lines whose heights at a cell agree within this are blended
JUNCTION_R0 = 8.0      # m from the node where two lines meet: within, they blend whatever their heights
JUNCTION_R1 = 16.0     # m ... fading out up to here
SEAM_R = 4.0           # m around a seam between the heights of two lines or two polygons: relaxed
SEAM_DZ = 0.01         # m, a smaller difference between the heights of two lines or polygons is no seam
JUNCTION_BLEND = False # lines meeting at a node blend whatever their heights (JUNCTION_R0/R1)
JUNCTION_DZ = 1.0      # m, the cells of two lines meeting at a node within JUNCTION_R0 m are one surface up to this
# v3.0 fix round: two neighbouring cells that take their heights from one line at points more than FOLD_GAP m apart
# along it (the inside of a hairpin, where the nearest point of the line jumps from one leg to the other) are not one
# surface by that alone: up to v3.0 they were linked whatever their heights (as the steps of a stair), and the mesh
# rose by the whole height between the legs from one cell to the next, a step of up to a metre across the lane after
# the hairpins. Now, up to FOLD_DZ apart, the link is a seam spread by relax_seams (farther apart they stay as they were).
FOLD_GAP = 3.0
FOLD_DZ = 1.5
# v3.0 fix round: a path line (a "1m Weg" ending at a road, a flight of steps beside it) gives no height to a cell that
# a road line also reaches: up to v3.0 the line nearest the cell in half widths won, and a path, with its half width
# of 0.5 m, won the cells right over it inside the carriageway, at its own height (0.3-2 m off the road's). Its score
# is raised by this much (half widths).
PATH_YIELD = 10.0
# v3.0.40: a survey polygon that only paths reach (a flight of steps, a footpath's polygon beside a road) also takes the
# heights of the road lines whose carriageway runs into it, on that carriageway only (CARR_REACH m past its edge): the
# polygon of the steps up from Via Cantonale (Pura, 1980 3337; 1132 -285) gave the edge of the lane the steps' height,
# 1-2.5 m above the road, and a car's wheels met a wall inside the lane. Off the carriageway the path keeps its own.
CARR_REACH = 0.25
# v3.0 fix round: the cells within BRIDGE_HEAD m of the end of a bridge (its head, past the end of the line leading to it)
# keep their line's height: they were left to the ground of the gully under the bridge, 2-3 m below the deck (above Vezio).
BRIDGE_HEAD = 8.0
STRIP_BLEND_M = 2.0    # m outside a road's carriageway where its weight in a junction strip (strip_height) is a quarter
STEP_DZ_PIN = 0.05     # m, cells pinned to two different strips that differ more are left free (_pin_strips)
MAIN_CLASSES = ("Autobahn", "Autostrasse", "10m Strasse", "8m Strasse", "6m Strasse")
REACH = 6.0            # m beyond its half width a station still gives heights (junction corners)
NEAR_NET = 25.0        # m, hard surfaces (yards, car parks) farther from any line are left out
FILL_OFF = 2.0         # m, largest offset of a square or car park from the smoothed ground
OFF_LINE = 1.5         # half widths from its line: beyond, a cell far from the ground keeps to the ground
FILLED = -2            # line code of the cells of squares and car parks (harmonic fill)
SUNK = 0.75            # m, deepest a survey surface may lie under the lowest bare ground within 1 m
AV_CLASSES = {"strada_sentiero": "road", "marciapiede": "sidewalk", "spartitraffico": "island",
              "altro_rivestimento_duro": "hard"}


class Network:
    """Stations with their heights, and the drivable polygons with the lines they belong to."""

    def __init__(self, exclude=None):
        self.segs, st, node_pos = network.load()
        self.node_xy = np.asarray(node_pos, np.float64)
        ns = network_surface.load()
        self.x, self.y, self.w = st["x"], st["y"], st["width"]
        self.z_tlm = st["z_tlm"]
        self.seg = st["seg"]
        self.arc = st["s"]                    # m along its line of every station
        self.z, self.g, self.c = ns["z"], ns["g"], ns["c"]
        self.nx, self.ny = ns["nx"], ns["ny"]
        self.n0 = np.array([sg["nodes"][0] for sg in self.segs], np.int64)     # end nodes of every line
        self.n1 = np.array([sg["nodes"][-1] for sg in self.segs], np.int64)
        self.tree = cKDTree(np.column_stack([self.x, self.y]))
        self.bridge = np.array([s["bridge"] for s in self.segs])
        self.main = np.array([s["kind"] == "road" and s["class"] in MAIN_CLASSES for s in self.segs])
        self.path = np.array([s["kind"] == "path" for s in self.segs])
        bn = sorted({nd for s in self.segs if s["bridge"] for nd in s["nodes"]})
        self.bridge_ends = cKDTree(self.node_xy[bn]) if bn else None          # the end nodes of the bridges
        self.polys = []                       # dicts: geom, cls, surface, segs
        self.deck_tops = []                   # bridge deck top triangles (bridges.py)
        self.deck_feet = []                   # (kind, footprint) of every deck built (bridges.py)
        self._build_polygons(exclude)

    # ------------------------------------------------------------------ polygons
    def _build_polygons(self, exclude):
        segs = self.segs
        keep = area.polygon().buffer(network.CLIP)
        # bridge decks (bridges.py, as wide as the deck; survey polygons 1 m wider, so no sliver of
        # them is left beside a deck without a line to take its height from) and the corridor stay out
        import bridges
        decks, decks_av, node_decks, deck_hw = [], [], {}, []
        # v3.1: the decks on the survey's outline (bridges.survey_decks) as they are, the survey's roads up to them
        survey = {i: foot for foot, ids in bridges.survey_decks(segs, self.x, self.y, self.w) for i in ids}
        for s in segs:
            if not s["bridge"]:
                continue
            L = shapely.LineString(np.column_stack([self.x[s["first"]:s["first"] + s["n"]],
                                                    self.y[s["first"]:s["first"] + s["n"]]]))
            hw = bridges.half_width({"width": self.w}, s)
            for nd in s["nodes"]:
                node_decks.setdefault(nd, []).append(len(decks))
            deck_hw.append(hw)
            if s["id"] in survey:
                decks.append(survey[s["id"]])
                decks_av.append(survey[s["id"]])
                continue
            decks.append(L.buffer(hw, cap_style="flat"))
            decks_av.append(L.buffer(hw + 1.0, cap_style="flat"))
        ex = [exclude] if exclude is not None else []
        holes_av = shapely.union_all(decks_av + ex) if decks_av or ex else shapely.Polygon()
        ex_geom = shapely.union_all(ex) if ex else shapely.Polygon()
        shapely.prepare(ex_geom)
        # lines as strips (for the association and the parts without survey)
        lines = []
        for s in segs:
            a, n = s["first"], s["n"]
            lines.append(shapely.LineString(np.column_stack([self.x[a:a + n], self.y[a:a + n]])))
        ltree = shapely.STRtree(lines)
        av = pickle.load(open(os.path.join(WORK, "av_local.pkl"), "rb"))
        dry = network.LakeCut(av)             # v3.0.40: no surface under the lake (a slipway at Caslano)
        road_like = []
        for cls, kind in AV_CLASSES.items():
            for g, _ in av["LCSF"].get(cls, []):
                if not g.intersects(keep):
                    continue
                g = g.intersection(keep)
                if kind == "hard" and ltree.query(g, predicate="dwithin", distance=NEAR_NET).size == 0:
                    continue
                g = g.difference(holes_av) if not holes_av.is_empty and g.intersects(holes_av) else g
                g = dry(g)
                for p in shapely.get_parts(g):
                    if p.geom_type == "Polygon" and p.area >= 0.5:
                        road_like.append((p, kind))
        av_tree = shapely.STRtree([p for p, _ in road_like]) if road_like else None
        hw_line = [0.5 * float(np.median(self.w[s["first"]:s["first"] + s["n"]])) for s in segs]
        for p, kind in road_like:
            near = ltree.query(p, predicate="dwithin", distance=1.0)
            q = {"geom": p, "cls": kind, "surface": "hard", "segs": set(int(i) for i in near)}
            if kind == "road" and q["segs"] and all(segs[i]["kind"] == "path" for i in q["segs"]):
                carr = [int(i) for i in ltree.query(p, predicate="dwithin", distance=8.0)
                        if segs[i]["kind"] != "path" and not segs[i]["bridge"] and int(i) not in q["segs"]
                        and lines[i].distance(p) < hw_line[i]]
                if carr:
                    q["carr_segs"] = set(carr)
            self.polys.append(q)
        # strips of the lines outside the survey
        node_segs = {}
        for s in segs:
            for nd in s["nodes"]:
                node_segs.setdefault(nd, set()).add(s["id"])
        for s, L in zip(segs, lines):
            if s["bridge"]:
                continue
            a, n = s["first"], s["n"]
            hw = 0.5 * float(np.median(self.w[a:a + n]))
            strip = L.buffer(hw, cap_style="round", quad_segs=4)
            if av_tree is not None:
                near = av_tree.query(strip, predicate="intersects")
                if len(near):
                    strip = strip.difference(shapely.union_all([road_like[i][0] for i in near]))
            # the decks of the bridges it leads onto, where the two overlap at the junction only (a
            # path that starts on a bridge and goes down under it keeps its way under), not the
            # decks that pass over it
            holes = []
            for nd in s["nodes"]:
                for i in node_decks.get(nd, []):
                    near_node = shapely.Point(self.node_xy[nd]).buffer(deck_hw[i] + hw + 0.5)
                    holes.append(decks[i].intersection(near_node))
            for hole in holes + ([ex_geom] if not ex_geom.is_empty else []):
                if not hole.is_empty and strip.intersects(hole):
                    strip = strip.difference(hole)
            strip = dry(strip.intersection(keep))
            own = {s["id"]} | node_segs.get(s["nodes"][0], set()) | node_segs.get(s["nodes"][1], set())
            for p in shapely.get_parts(strip):
                if p.geom_type == "Polygon" and p.area >= 0.5:
                    self.polys.append({"geom": p, "cls": "strip_" + s["kind"], "surface": s["surface"], "segs": own,
                                       "own": s["id"]})
        # polygons of the lines of the survey polygons must not be blended with bridges' lines
        for p in self.polys:
            p["segs"] = {k for k in p["segs"] if not segs[k]["bridge"]}
        self.ptree = shapely.STRtree([p["geom"] for p in self.polys])
        n_seg = len(segs)
        rows = np.concatenate([np.full(len(p["segs"]), i) for i, p in enumerate(self.polys)]) if self.polys else []
        cols = np.concatenate([np.array(sorted(p["segs"]), int) for p in self.polys]) if self.polys else []
        self.assoc = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(self.polys), n_seg))
        cr = [(i, k) for i, p in enumerate(self.polys) for k in sorted(p.get("carr_segs", ()))]
        self.assoc_carr = sp.csr_matrix((np.ones(len(cr), bool), ([a for a, _ in cr], [b for _, b in cr])),
                                        shape=(len(self.polys), n_seg))
        print("path polygons with the carriageway of a road: %d (%d road lines)"
              % (sum(1 for p in self.polys if "carr_segs" in p), len(cr)), flush=True)
        print("surveyed polygons and strips cut at the lake shore: %d" % dry.cut, flush=True)
        area_by = {}
        for p in self.polys:
            area_by[p["cls"]] = area_by.get(p["cls"], 0) + p["geom"].area / 1e4
        print("drivable polygons %d, ha %s" % (len(self.polys), {k: round(v, 1) for k, v in area_by.items()}), flush=True)

    def assign_surfaces(self):
        """v2.4: the surface of every polygon (osm_surface.py): the lines' strips that of their line,
        the survey roads that of the lines and OSM ways in them (in parts where they differ), the
        yards and squares that of the OSM areas over them; sidewalks and islands stay paved.
        p["surface"] is the main one, p["zones"] the parts [(surface, geometry)] where there are several."""
        import osm_surface
        self.line_cat, st = osm_surface.line_categories(self)
        area_by = {}
        for p in self.polys:
            zones = None
            if p["cls"].startswith("strip"):
                p["surface"] = self.line_cat[p["own"]]
            elif p["cls"] == "road":
                zones = osm_surface.polygon_zones(self, p, self.line_cat)
            elif p["cls"] == "hard":
                zones = osm_surface.area_zones(p)
            if zones is not None:
                p["surface"] = zones[-1][0]
                if len(zones) > 1:
                    p["zones"] = zones
            for c, g in p.get("zones") or [(p["surface"], p["geom"])]:
                area_by[(p["cls"], c)] = area_by.get((p["cls"], c), 0.0) + g.area / 1e4
        st["ha"] = {f"{k[0]}:{k[1]}": round(v, 2) for k, v in sorted(area_by.items())}
        st["polygons_in_parts"] = sum(1 for p in self.polys if "zones" in p)
        print("surfaces (OSM, swissTLM3D):", st, flush=True)
        return st

    # ------------------------------------------------------------------ heights
    def heights(self, X, Y, P, with_seg=False, own=None, with_u=False):
        """Heights at points (X, Y) inside polygons P (index into self.polys); NaN where no line
        of the polygon is near (squares, car parks). with_seg: also the line that gave each height
        (-1 for none).
        Every line near a point gives the height of its surface there: the point projected on the
        line (between two stations, linear), plus the cross slope times the offset; beyond the end
        of a line it carries on along the end's grade. The line nearest the point (relative to its
        half width) wins; lines that agree with it within BLEND_DZ are blended (junctions).
        own: only this line (a strip along its line: a path beside a higher road keeps its own
        height up to its edges). with_u: also the distance of each point from the winning line in
        half widths (0 on the axis, 1 at the edge of its carriageway) and how far past the end of
        that line the point lies (m, 0 alongside it).
        A line gives heights only within REACH m of its carriageway, measured as the true distance:
        the offset across the line alone would let a point far ahead of a bend, or far beyond the
        end, take the height of a line it is nowhere near (a square beside a steep path)."""
        n = len(X)
        out = np.full(n, np.nan)
        segs_out = np.full(n, -1, np.int64)
        u_out = np.full(n, np.inf)
        e_out = np.zeros(n)
        a_out = np.full(n, np.nan)
        if n == 0:
            return (out, segs_out, u_out, e_out, a_out) if with_u else ((out, segs_out) if with_seg else out)
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        k = 12
        _, j = self.tree.query(np.column_stack([X, Y]), k=k, distance_upper_bound=12.0 + REACH)
        valid = j < len(self.x)
        jj = np.where(valid, j, 0)
        seg = self.seg[jj]
        ok = valid & np.asarray(self.assoc[np.repeat(P, k), seg.ravel()]).reshape(n, k)
        carr = np.zeros_like(ok)
        if own is None and getattr(self, "assoc_carr", None) is not None and self.assoc_carr.nnz:
            carr = valid & ~ok & np.asarray(self.assoc_carr[np.repeat(P, k), seg.ravel()]).reshape(n, k)
            ok |= carr
        if own is not None:
            ok &= seg == own
        N = len(self.x)
        Xb, Yb = X[:, None], Y[:, None]
        best_d = np.full((n, k), np.inf)
        h = np.full((n, k), np.nan)
        off = np.zeros((n, k))
        ext = np.zeros((n, k))
        hw = np.ones((n, k))
        arc = np.zeros((n, k))
        for da, db in ((-1, 0), (0, 1)):
            a = np.clip(jj + da, 0, N - 1)
            b = np.clip(jj + db, 0, N - 1)
            good = ok & (self.seg[a] == seg) & (self.seg[b] == seg) & (a != b)
            vx, vy = self.x[b] - self.x[a], self.y[b] - self.y[a]
            L2 = np.maximum(vx * vx + vy * vy, 1e-9)
            s_raw = ((Xb - self.x[a]) * vx + (Yb - self.y[a]) * vy) / L2
            s = np.clip(s_raw, 0.0, 1.0)
            qx, qy = self.x[a] + s * vx, self.y[a] + s * vy
            nx = self.nx[a] + s * (self.nx[b] - self.nx[a])
            ny = self.ny[a] + s * (self.ny[b] - self.ny[a])
            nn = np.maximum(np.hypot(nx, ny), 1e-9)
            nx, ny = nx / nn, ny / nn
            t = (Xb - qx) * nx + (Yb - qy) * ny
            along = (Xb - qx) * ny - (Yb - qy) * nx            # tangent (Ny, -Nx)
            # past the first / last station of the line: carry on along its grade
            a_first = (a == 0) | (self.seg[np.clip(a - 1, 0, N - 1)] != seg)
            b_last = (b == N - 1) | (self.seg[np.clip(b + 1, 0, N - 1)] != seg)
            before, after = (s_raw < 0) & a_first, (s_raw > 1) & b_last
            e = np.where(before | after, along, 0.0)
            ge = np.where(s_raw < 0, self.g[a], self.g[b])
            z = self.z[a] + s * (self.z[b] - self.z[a]) + (self.c[a] + s * (self.c[b] - self.c[a])) * t + ge * e
            d = np.hypot(Xb - qx, Yb - qy)
            # a clamp at a middle station: the neighbouring piece has the same point, let it win
            d = np.where(((s_raw < 0) & ~a_first) | ((s_raw > 1) & ~b_last), d + 1e-3, d)
            take = good & (d < best_d)
            best_d = np.where(take, d, best_d)
            h = np.where(take, z, h)
            off = np.where(take, t, off)
            ext = np.where(take, e, ext)
            hw = np.where(take, 0.5 * (self.w[a] + s * (self.w[b] - self.w[a])), hw)
            arc = np.where(take, self.arc[a] + s * (self.arc[b] - self.arc[a]), arc)
        ok &= np.isfinite(h) & (np.abs(off) <= hw + REACH) & (best_d <= hw + REACH)
        ok &= ~carr | ((np.abs(off) <= hw + CARR_REACH) & (best_d <= hw + CARR_REACH))
        # one entry per line and point (neighbouring stations of a line give the same projection)
        order = np.argsort(np.where(ok, best_d, np.inf), axis=1)
        seg_o = np.take_along_axis(seg, order, 1)
        ok_o = np.take_along_axis(ok, order, 1)
        dup = np.zeros((n, k), bool)
        for c in range(1, k):
            dup[:, c] = (seg_o[:, :c] == seg_o[:, c:c + 1]).any(1)
        ok_o &= ~dup
        h_o = np.take_along_axis(h, order, 1)
        u = np.abs(np.take_along_axis(off, order, 1)) / np.maximum(np.take_along_axis(hw, order, 1), 0.3)
        e_o = np.abs(np.take_along_axis(ext, order, 1))
        score = np.where(ok_o, u + 0.05 * e_o + PATH_YIELD * self.path[seg_o], np.inf)
        best = np.argmin(score, axis=1)
        has = ok_o.any(1)
        zb = h_o[np.arange(n), best]
        wt = np.where(ok_o, np.exp(-2.0 * u ** 2) / (1.0 + e_o), 0.0)
        # lines meeting the winning line at a node near the point: one junction, blended whatever their
        # heights (fading out from JUNCTION_R0 to JUNCTION_R1 m from the node); the others only where they
        # agree with it within BLEND_DZ (a road passing above or below)
        sw = seg_o[np.arange(n), best]
        a0, a1 = self.n0[sw][:, None], self.n1[sw][:, None]
        b0, b1 = self.n0[seg_o], self.n1[seg_o]
        shared = np.where((a0 == b0) | (a0 == b1), a0, np.where((a1 == b0) | (a1 == b1), a1, -1))
        sn = np.maximum(shared, 0)
        dn = np.where(shared >= 0, np.hypot(Xb - self.node_xy[sn, 0], Yb - self.node_xy[sn, 1]), np.inf)
        fj = np.clip((JUNCTION_R1 - dn) / (JUNCTION_R1 - JUNCTION_R0), 0.0, 1.0)
        agree = np.abs(np.nan_to_num(h_o) - zb[:, None]) <= BLEND_DZ
        wt = wt * np.where(agree, 1.0, fj if JUNCTION_BLEND else 0.0)
        wt[np.arange(n), best] = np.where(has, np.maximum(wt[np.arange(n), best], 1e-12), 0.0)
        ws = wt.sum(1)
        if getattr(self, "explain", False):           # fix_probe.py: the candidates of every point
            self.explained = dict(seg=seg_o, h=h_o, u=u, e=e_o, wt=wt, ok=ok_o, best=best)
        good = has & (ws > 0)
        out[good] = (wt * np.nan_to_num(h_o)).sum(1)[good] / ws[good]
        segs_out[good] = seg_o[np.arange(n), best][good]
        u_out[good] = u[np.arange(n), best][good]
        e_out[good] = e_o[np.arange(n), best][good]
        a_out[good] = np.take_along_axis(arc, order, 1)[np.arange(n), best][good]
        if with_u:
            return out, segs_out, u_out, e_out, a_out
        return (out, segs_out) if with_seg else out

    def _junction_links(self, Ga, Gb, xs, ys):
        """Pairs of neighbouring cells (lines Ga, Gb) whose lines meet at a node within JUNCTION_R0 m of the
        pair (xs: x of the pairs' columns, ys: y of their rows)."""
        out = np.zeros(Ga.shape, bool)
        cand = (Ga >= 0) & (Gb >= 0) & (Ga != Gb)
        if not cand.any():
            return out
        r, c = np.nonzero(cand)
        a, b = Ga[r, c], Gb[r, c]
        a0, a1, b0, b1 = self.n0[a], self.n1[a], self.n0[b], self.n1[b]
        shared = np.where((a0 == b0) | (a0 == b1), a0, np.where((a1 == b0) | (a1 == b1), a1, -1))
        ok = shared >= 0
        px, py = xs[c], ys[r]
        sn = np.maximum(shared, 0)
        d = np.hypot(px - self.node_xy[sn, 0], py - self.node_xy[sn, 1])
        out[r[ok & (d < JUNCTION_R0)], c[ok & (d < JUNCTION_R0)]] = True
        return out

    def _pin_strips(self, box, x0, y1, W, H, m, Z, G):
        """Mask of the cells (m: the polygons' cells that may) beside a strip that take its height (Z set in place)."""
        pin = np.zeros((H, W), bool)
        sids = [int(i) for i in self.ptree.query(box, predicate="intersects") if self.polys[i]["cls"].startswith("strip")]
        if not sids:
            return pin
        tr = Affine(RES, 0, x0, 0, -RES, y1)
        sown = features.rasterize([(self.polys[i]["geom"], i) for i in sids], out_shape=(H, W), transform=tr, fill=-1,
                                  dtype=np.int32, all_touched=True)
        near = sown.copy()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.full((H, W), -1, np.int32)
            sh[max(dr, 0):H + min(dr, 0), max(dc, 0):W + min(dc, 0)] = sown[max(-dr, 0):H + min(-dr, 0), max(-dc, 0):W + min(-dc, 0)]
            near = np.where(near >= 0, near, sh)
        cand = m & (near >= 0)
        if not cand.any():
            return pin
        rr, cc = np.nonzero(cand)
        X, Y = x0 + (cc + 0.5) * RES, y1 - (rr + 0.5) * RES
        Z_own = Z.copy()
        pid_of = np.full((H, W), -1, np.int64)
        for pid in np.unique(near[rr, cc]):
            k = near[rr, cc] == pid
            line = self.polys[int(pid)]["own"]
            g = G[rr[k], cc[k]]
            gi = np.maximum(g, 0)
            meet = (g == line) | (g < 0) | (self.n0[gi] == self.n0[line]) | (self.n0[gi] == self.n1[line]) \
                | (self.n1[gi] == self.n0[line]) | (self.n1[gi] == self.n1[line])
            zs = self.strip_height(int(pid))(X[k], Y[k])
            ok = meet & np.isfinite(zs) & (np.abs(zs - Z[rr[k], cc[k]]) < JUNCTION_DZ)
            Z[rr[k][ok], cc[k][ok]] = zs[ok]
            pin[rr[k][ok], cc[k][ok]] = True
            pid_of[rr[k][ok], cc[k][ok]] = pid
        # where two strips meet (a junction of two lines both meshed as strips), neighbouring cells pinned to different
        # strips at different heights keep their own: pinned, the step between them stayed (0.47 m under the church of
        # Arosio); free, relax_seams spreads it
        clash = np.zeros((H, W), bool)
        for a, b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                     ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
            c = pin[a] & pin[b] & (pid_of[a] != pid_of[b]) & (np.abs(Z[a] - Z[b]) > STEP_DZ_PIN)
            clash[a] |= c
            clash[b] |= c
        Z[clash] = Z_own[clash]
        pin &= ~clash
        return pin

    def strip_height(self, pid):
        """Height function of a strip polygon: the surface of its own line (nearest station of its
        lines where that finds none)."""
        own = self.polys[pid].get("own")
        others = sorted(k for k in self.polys[pid]["segs"] if k != own and not self.path[k]) if own is not None else []

        def fn(x, y, *_):
            x = np.atleast_1d(np.asarray(x, np.float64)); y = np.atleast_1d(np.asarray(y, np.float64))
            z = self.heights(x, y, np.full(len(x), pid), own=own)
            if others and not self.path[own]:
                # v3.0 fix round: near the end nodes of its line a strip takes the weighted mean of its line's surface
                # and those of the roads meeting there, each weighted by how far the point is outside its carriageway (the
                # same weights in every strip of the junction, so where two strips overlap they agree). Up to v3.0 each
                # kept its own line's height and two strips meeting at a junction overlapped 0.6 m apart (under the
                # church of Arosio). Full within JUNCTION_R0 m of the node, fading out at JUNCTION_R1.
                ends = self.node_xy[[self.n0[own], self.n1[own]]]
                d = np.min(np.hypot(x[:, None] - ends[:, 0], y[:, None] - ends[:, 1]), axis=1)
                f = np.clip((JUNCTION_R1 - d) / (JUNCTION_R1 - JUNCTION_R0), 0.0, 1.0)
                k = (f > 0) & np.isfinite(z)
                if k.any():
                    Pk = np.full(int(k.sum()), pid)
                    num, den = np.zeros(int(k.sum())), np.zeros(int(k.sum()))
                    for line in [own] + others:
                        zl, _, ul, el, _ = self.heights(x[k], y[k], Pk, own=line, with_u=True)
                        sg = self.segs[line]
                        hw = max(0.5 * float(np.median(self.w[sg["first"]:sg["first"] + sg["n"]])), 0.3)
                        dm = np.hypot(np.maximum(ul * hw - hw, 0.0), el)       # m outside its carriageway
                        wl = np.where(np.isfinite(zl), 1.0 / (1.0 + (dm / STRIP_BLEND_M) ** 2) ** 2, 0.0)
                        num += wl * np.nan_to_num(zl)
                        den += wl
                    zb = np.where(den > 1e-9, num / np.maximum(den, 1e-12), z[k])
                    z[k] = z[k] + f[k] * (zb - z[k])
            bad = np.isnan(z)
            if bad.any():
                segs = np.array(sorted(self.polys[pid]["segs"]), int)
                _, j = self.tree.query(np.column_stack([x[bad], y[bad]]), k=8)
                j = np.minimum(j, len(self.x) - 1)
                allowed = np.isin(self.seg[j], segs)
                first = np.where(allowed.any(1), allowed.argmax(1), 0)
                z[bad] = self.z[j[np.arange(len(j)), first]]
            return z
        return fn

    # ------------------------------------------------------------------ tiles
    def tile_surface(self, x0, y0, x1, y1, dtm):
        """surface_fit.Surface of the drivable cells of the window, and the polygon indices in it."""
        box = shapely.box(x0, y0, x1, y1)
        ids = self.ptree.query(box, predicate="intersects")
        ids = np.array([i for i in ids if not self.polys[i]["cls"].startswith("strip")], int)
        W, H = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
        if len(ids) == 0:
            return None, ids
        tr = Affine(RES, 0, x0, 0, -RES, y1)
        shapes = [(self.polys[i]["geom"], int(i)) for i in ids]
        owner = features.rasterize(shapes, out_shape=(H, W), transform=tr, fill=-1, dtype=np.int32)
        # a thin polygon at an angle is a chain of cells touching only at their corners: the cells
        # it merely touches join it (where no other polygon has them), so its cells are linked
        touched = features.rasterize(shapes, out_shape=(H, W), transform=tr, fill=-1, dtype=np.int32,
                                     all_touched=True)
        free = (owner < 0) & (touched >= 0)
        owner[free] = touched[free]
        m = owner >= 0
        if not m.any():
            return None, ids
        rr, cc = np.nonzero(m)
        X = x0 + (cc + 0.5) * RES
        Y = y1 - (rr + 0.5) * RES
        Z = np.full((H, W), np.nan)
        G = np.full((H, W), -1, np.int64)                   # line that gave the height
        U = np.full((H, W), np.inf)
        E = np.zeros((H, W))
        A = np.full((H, W), np.nan)                          # m along the line that gave the height
        Z[rr, cc], G[rr, cc], U[rr, cc], E[rr, cc], A[rr, cc] = self.heights(X, Y, owner[rr, cc], with_seg=True, with_u=True)
        # the smoothed DTM on the cells of the window (sampled: the window may reach past the edge
        # of the DTM raster, near the corners of the area)
        Xc = x0 + (np.arange(W) + 0.5) * RES
        Yc = y1 - (np.arange(H) + 0.5) * RES
        Ds = dtm.sample(np.tile(Xc, H), np.repeat(Yc, W)).reshape(H, W).astype(np.float64)
        if np.isnan(Ds).any():
            Ds = np.where(np.isnan(Ds), np.nanmean(Ds), Ds)
        Dlo = ndi.minimum_filter(Ds, size=5) - SUNK             # lowest bare ground within 1 m
        Ds = ndi.gaussian_filter(Ds, 4.0)
        # a survey polygon reaching beyond the carriageway of its line onto ground far above or
        # below it (a terrace, a yard behind a wall), or past the end of its line where the line's
        # grade carried on leaves the ground: that part keeps to the ground
        at_bridge = np.zeros((H, W), bool)
        if self.bridge_ends is not None:
            dq, _ = self.bridge_ends.query(np.column_stack([np.tile(Xc, H), np.repeat(Yc, W)]), distance_upper_bound=BRIDGE_HEAD)
            at_bridge = (G >= 0) & np.isfinite(dq).reshape(H, W)
        off = m & ((U > OFF_LINE) | (E > 1.0)) & (np.abs(Z - Ds) > FILL_OFF) & ~at_bridge
        if getattr(self, "explain", False):           # fix_probe.py: why a cell left its line
            self.tile_explained = dict(Z=Z.copy(), G=G.copy(), U=U.copy(), E=E.copy(), A=A.copy(), Ds=Ds.copy(),
                                       Dlo=Dlo.copy(), off=off.copy(), at_bridge=at_bridge.copy())
        Z[off] = np.nan
        G[off] = -1
        # squares and car parks: smoothed DTM, corrected to meet the roads around
        hole = m & np.isnan(Z)
        if hole.any():
            Z = harmonic_fill(Z, m, Ds)
            G[hole] = FILLED                  # one smooth field: its cells are one surface however steep
        # no paved surface under the ground: a cell given a height far below the bare ground (a line
        # of the polygon that runs lower, a hairpin's lower leg, the fill pulled down by it) comes up
        # to SUNK under it, as a surface of its own with a step to the rest
        low = m & (Z < Dlo)
        Z[low] = Dlo[low]
        G[low] = FILLED
        # v3.0 fix round: the cells beside the strip of a line (its mesh outside the survey polygons, at the line's
        # own height: strip_height) take the strip's height and keep it, where their own line is the strip's or
        # meets it at a node (or they are a filled square) and the two differ by less than JUNCTION_DZ; the seam
        # moves into the polygon and is spread there. Up to v3.0 a strip met a polygon whose cells had been given
        # another line's height, or relaxed towards a junction, with a step (0.3-0.6 m: Astano, Arosio).
        main_cell = (G >= 0) & self.main[np.maximum(G, 0)] & (U <= 1.0)
        pin = self._pin_strips(box, x0, y1, W, H, m & ~main_cell, Z, G)     # a main road does not bend to a strip
        if getattr(self, "explain", False):
            self.tile_explained.update(pin=pin.copy(), Z_pinned=Z.copy(), G_final=G.copy())
        # neighbouring cells of a polygon are one surface when their heights come from the same line
        # (continuous however steep the line: stairs, mule tracks) or agree within STEP_Z (junctions,
        # squares); a step (a wall) is left between lines at different heights
        # v2.2: cells of neighbouring polygons too (the carriageway and its sidewalk, a yard beside the road)
        LR = np.zeros(owner.shape, bool)
        LD = np.zeros(owner.shape, bool)
        LR[:, :-1] = m[:, :-1] & m[:, 1:]
        LD[:-1, :] = m[:-1, :] & m[1:, :]
        same_r = (G[:, :-1] == G[:, 1:]) & (G[:, :-1] != -1) & (owner[:, :-1] == owner[:, 1:])
        same_d = (G[:-1, :] == G[1:, :]) & (G[:-1, :] != -1) & (owner[:-1, :] == owner[1:, :])
        fold_r = same_r & (G[:, :-1] >= 0) & (np.abs(A[:, :-1] - A[:, 1:]) > FOLD_GAP) \
            & (np.abs(Z[:, :-1] - Z[:, 1:]) < FOLD_DZ)
        fold_d = same_d & (G[:-1, :] >= 0) & (np.abs(A[:-1, :] - A[1:, :]) > FOLD_GAP) \
            & (np.abs(Z[:-1, :] - Z[1:, :]) < FOLD_DZ)
        # two lines that meet at a node near the cells are one junction: their cells are linked up to
        # JUNCTION_DZ apart (the step between them is spread by relax_seams, not left as a kerb)
        jun_r = self._junction_links(G[:, :-1], G[:, 1:], x0 + (np.arange(W - 1) + 1.0) * RES,
                                     y1 - (np.arange(H) + 0.5) * RES)
        jun_d = self._junction_links(G[:-1, :], G[1:, :], x0 + (np.arange(W) + 0.5) * RES,
                                     y1 - (np.arange(H - 1) + 1.0) * RES)
        LR[:, :-1] &= same_r | (np.abs(Z[:, :-1] - Z[:, 1:]) < STEP_Z) | (jun_r & (np.abs(Z[:, :-1] - Z[:, 1:]) < JUNCTION_DZ))
        LD[:-1, :] &= same_d | (np.abs(Z[:-1, :] - Z[1:, :]) < STEP_Z) | (jun_d & (np.abs(Z[:-1, :] - Z[1:, :]) < JUNCTION_DZ))
        # the carriageways of the main roads keep their surface: the side roads, yards and paths meeting
        # them take up the difference (a main road does not bend to a side road)
        pin_r, pin_d = pin[:, :-1] != pin[:, 1:], pin[:-1, :] != pin[1:, :]
        Z = relax_seams(Z, G, owner, LR, LD, main_cell | pin, (fold_r | pin_r, fold_d | pin_d), pin)
        return surface_fit.Surface(owner, np.nan_to_num(Z), LR, LD, x0, y1, RES), ids


def relax_seams(Z, G, owner, LR, LD, fixed_cells=None, folds=None, pinned=None):
    """Z with the jumps at the seams spread over SEAM_R m. A seam is a link between two cells whose heights
    come from different lines, or that lie in different polygons, and differ by more than SEAM_DZ. Around
    the seams (cells within SEAM_R m, linked) the heights are solved again keeping the height difference
    of every link but the seams', which becomes zero (a gradient-domain edit: the crown, the grade and
    the vertical curves of the roads stay, only the jump is spread smoothly over the zone); the cells
    outside the zone keep their heights, and so do fixed_cells (the carriageways of the main roads) unless
    the seam is between two of them. folds: (right, down) link masks of neighbouring cells of one line that are
    seams too (FOLD_GAP, the cells beside a strip). pinned: cells that keep their heights in any case."""
    fr_, fd_ = folds if folds is not None else (np.zeros(LR[:, :-1].shape, bool), np.zeros(LD[:-1, :].shape, bool))
    jr = LR[:, :-1] & ((G[:, :-1] != G[:, 1:]) | (owner[:, :-1] != owner[:, 1:]) | fr_) & \
        (np.abs(Z[:, :-1] - Z[:, 1:]) > SEAM_DZ)
    jd = LD[:-1, :] & ((G[:-1, :] != G[1:, :]) | (owner[:-1, :] != owner[1:, :]) | fd_) & \
        (np.abs(Z[:-1, :] - Z[1:, :]) > SEAM_DZ)
    seam = np.zeros(Z.shape, bool)
    seam[:, :-1] |= jr
    seam[:, 1:] |= jr
    seam[:-1, :] |= jd
    seam[1:, :] |= jd
    if not seam.any():
        return Z
    linked = np.zeros(Z.shape, bool)
    linked[:, :-1] |= LR[:, :-1]
    linked[:, 1:] |= LR[:, :-1]
    linked[:-1, :] |= LD[:-1, :]
    linked[1:, :] |= LD[:-1, :]
    zone = ndi.binary_dilation(seam, iterations=int(round(SEAM_R / RES))) & linked & np.isfinite(Z)
    if fixed_cells is not None:
        both = np.zeros(Z.shape, bool)                  # seams between two fixed cells
        fr = jr & fixed_cells[:, :-1] & fixed_cells[:, 1:]
        fd = jd & fixed_cells[:-1, :] & fixed_cells[1:, :]
        both[:, :-1] |= fr
        both[:, 1:] |= fr
        both[:-1, :] |= fd
        both[1:, :] |= fd
        free = ndi.binary_dilation(both, iterations=int(round(SEAM_R / RES))) if both.any() else both
        zone &= ~fixed_cells | free
    if pinned is not None:                              # pinned cells keep their heights whatever the seams around
        zone &= ~pinned
    if fixed_cells is not None or pinned is not None:
        if not zone.any():
            return Z
    ur, uc = np.nonzero(zone)
    idx = -np.ones(Z.shape, np.int64)
    idx[ur, uc] = np.arange(len(ur))
    H, W = Z.shape
    rows, cols, vals = [], [], []
    b = np.zeros(len(ur))
    diag = np.zeros(len(ur))
    # the four links of every cell of the zone: (neighbour offset, link present, target difference z_i - z_j)
    Zc = np.nan_to_num(Z)
    for dr_, dc_ in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        r2, c2 = ur + dr_, uc + dc_
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        r2c, c2c = np.clip(r2, 0, H - 1), np.clip(c2, 0, W - 1)
        if dc_ == 1:
            lk, jump = LR[ur, uc], _pad(jr, 1)[ur, uc]
        elif dc_ == -1:
            lk, jump = LR[ur, np.maximum(uc - 1, 0)] & (uc > 0), _pad(jr, 1)[ur, np.maximum(uc - 1, 0)] & (uc > 0)
        elif dr_ == 1:
            lk, jump = LD[ur, uc], _pad(jd, 0)[ur, uc]
        else:
            lk, jump = LD[np.maximum(ur - 1, 0), uc] & (ur > 0), _pad(jd, 0)[np.maximum(ur - 1, 0), uc] & (ur > 0)
        lk = lk & ok
        g = np.where(jump, 0.0, Zc[ur, uc] - Zc[r2c, c2c])            # keep the difference, but not a jump
        diag += lk
        b += np.where(lk, g, 0.0)
        nb_unk = lk & zone[r2c, c2c]
        rows.append(np.flatnonzero(nb_unk))
        cols.append(idx[r2c, c2c][nb_unk])
        vals.append(-np.ones(int(nb_unk.sum())))
        nb_kn = lk & ~zone[r2c, c2c]
        b[nb_kn] += Zc[r2c, c2c][nb_kn]
    fixed = diag == 0                                   # a cell without links keeps its height
    diag = np.where(fixed, 1.0, diag + 1e-4)            # (a weak pull to its own height: no free piece)
    b = np.where(fixed, Zc[ur, uc], b + 1e-4 * Zc[ur, uc])
    A = sp.csr_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(len(ur))]),
                                                        np.concatenate(cols + [np.arange(len(ur))]))),
                      shape=(len(ur), len(ur)))
    out = Z.copy()
    out[ur, uc] = spsolve(A.tocsc(), b)
    return out


def _pad(J, axis):
    """A link mask (H, W-1) or (H-1, W) padded with False to (H, W)."""
    if axis == 1:
        return np.concatenate([J, np.zeros((J.shape[0], 1), bool)], axis=1)
    return np.concatenate([J, np.zeros((1, J.shape[1]), bool)], axis=0)


def harmonic_fill(Z, m, D):
    """Fill the NaN cells of Z inside m: D plus the harmonic interpolation of Z - D from the known
    cells (4-neighbour Laplace over m)."""
    unk = m & np.isnan(Z)
    known = m & ~np.isnan(Z)
    # at most FILL_OFF m off the smoothed ground: a yard beside a road in a deep cutting stays on
    # the ground with a step (a wall) to the road, not in a pit
    R = np.where(known, np.clip(Z - D, -FILL_OFF, FILL_OFF), 0.0)
    idx = -np.ones(Z.shape, np.int64)
    ur, uc = np.nonzero(unk)
    idx[ur, uc] = np.arange(len(ur))
    rows, cols, vals = [], [], []
    b = np.zeros(len(ur))
    diag = np.zeros(len(ur))
    H, W = Z.shape
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        r2, c2 = ur + dr, uc + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        r2c, c2c = np.clip(r2, 0, H - 1), np.clip(c2, 0, W - 1)
        inm = ok & m[r2c, c2c]
        diag += inm
        nb_unk = inm & unk[r2c, c2c]
        rows.append(np.flatnonzero(nb_unk)); cols.append(idx[r2c, c2c][nb_unk]); vals.append(-np.ones(nb_unk.sum()))
        nb_kn = inm & known[r2c, c2c]
        b[nb_kn] += R[r2c, c2c][nb_kn]
    diag = np.maximum(diag, 1) + 1e-6
    A = sp.csr_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(len(ur))]),
                                                        np.concatenate(cols + [np.arange(len(ur))]))),
                      shape=(len(ur), len(ur)))
    r = spsolve(A.tocsc(), b) if len(ur) else np.zeros(0)
    out = Z.copy()
    out[ur, uc] = D[ur, uc] + r
    return out


# ---------------------------------------------------------------------- meshes and carve
# class -> (material group of osm_surface.MATS, mesh cell m, uv tile m of the asphalt); the unpaved and
# stone surfaces (v2.4) tile over surface_textures.TILE_M m
CLASS_MESH = {"road": ("road", 3.0, 4.0), "strip_road": ("road", 3.0, 4.0), "hard": ("hard", 2.0, 4.0),
              "strip_path": ("path", 2.0, 4.0)}            # v3.0: 4 m, italy_mats.ASPHALT_TILE (up to v2.8 1.25)
FIXED = {"sidewalk": ("mp_sidewalk", 2.0, 4.0), "island": ("mp_island", 2.0, 2.5)}


def material(cls, surface):
    """(material, mesh cell m, uv tile m) of a polygon class with a surface of osm_surface."""
    if cls in FIXED:
        return FIXED[cls]
    import osm_surface
    import surface_textures
    group, cell, uvt = CLASS_MESH[cls]
    if surface == "natural":                         # swissTLM3D's own, before assign_surfaces
        surface = "gravel" if group == "road" else "dirt"
    surface = surface if surface in osm_surface.MATS[group] else "hard"
    return osm_surface.MATS[group][surface], cell, (uvt if surface == "hard" else surface_textures.TILE_M)


def tiles():
    """South-west corners of the TILE m tiles over the network (multiples of TILE)."""
    x0, y0, x1, y1 = area.bounds(network.CLIP + 5)
    return [(tx * TILE, ty * TILE) for tx in range(int(np.floor(x0 / TILE)), int(np.floor(x1 / TILE)) + 1)
            for ty in range(int(np.floor(y0 / TILE)), int(np.floor(y1 / TILE)) + 1)]


def mesh_tile(net, dtm, X0, Y0, xs, ys, on_mesh, on_tops):
    """Meshes of the chunks of one tile -> on_mesh(tx, ty, material, uv tile, V, T, S, ground);
    their top faces -> on_tops(tx, ty, triangles) (for carve_tile, once every tile is meshed)."""
    import road_mesh
    X1, Y1 = X0 + TILE, Y0 + TILE
    core = shapely.box(X0, Y0, X1, Y1)
    ids = net.ptree.query(core, predicate="intersects")
    if len(ids) == 0:
        return 0
    S, _ = net.tile_surface(X0 - MARGIN, Y0 - MARGIN, X1 + MARGIN, Y1 + MARGIN, dtm)
    if S is None:                    # only strips here: an empty surface for the skirts
        own = -np.ones((4, 4), np.int32)
        S = surface_fit.Surface(own, np.zeros((4, 4)), np.zeros((4, 4), bool), np.zeros((4, 4), bool),
                                X0 - MARGIN, Y0 - MARGIN + 2.0, RES)
    sub, _, _ = dtm.window(X0 - MARGIN, Y0 - MARGIN, X1 + MARGIN, Y1 + MARGIN, pad=4)
    lo = Grid(ndi.minimum_filter(np.asarray(sub.a, np.float32), size=5) - SUNK, sub.x_min, sub.y_max, sub.res)
    sub.a = ndi.gaussian_filter(np.asarray(sub.a, np.float32), 0.6)
    ground = lambda x, y: sub.sample(x, y)
    n_tri = 0

    def emit(cx, cy, mat, uvt, V, T):
        on_mesh(cx, cy, mat, uvt, V, T, S, ground)
        tri = V[T]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        on_tops(cx, cy, tri[nrm[:, 2] / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12) > 0.5])
    for cx in range(int(round(X0 / CHUNK)), int(round(X1 / CHUNK))):
        for cy in range(int(round(Y0 / CHUNK)), int(round(Y1 / CHUNK))):
            cbox = shapely.box(cx * CHUNK, cy * CHUNK, (cx + 1) * CHUNK, (cy + 1) * CHUNK)
            for pid in net.ptree.query(cbox, predicate="intersects"):
                p = net.polys[pid]
                piece = road_mesh.polygonal(p["geom"].intersection(cbox))
                if piece.is_empty or piece.area < 0.05:
                    continue
                # v2.4: the parts of the polygon with another surface (osm_surface.py) are meshed apart,
                # on the same heights: the facets of the whole piece, then each cut into the zones
                zones = p.get("zones") or [(p["surface"], None)]
                if p["cls"].startswith("strip"):                 # on its line, no raster
                    for surf, zg in zones:
                        zpiece = piece if zg is None else road_mesh.polygonal(piece.intersection(zg))
                        if zpiece.is_empty or zpiece.area < 1e-3:
                            continue
                        mat, cell, uvt = material(p["cls"], surf)
                        V, T = road_mesh.mesh_polygon(zpiece, net.strip_height(int(pid)), cell=cell)
                        if len(T) and np.isfinite(V[:, 2]).all():
                            emit(cx, cy, mat, uvt, V, T)
                            n_tri += len(T)
                    continue
                # the vertices too: no deeper than SUNK under the bare ground (an edge extrapolated
                # from far cells of its surface)
                zf = (lambda pid_: (lambda x, y, comp: np.maximum(S.height(x, y, pid=pid_, comp=int(comp)),
                                                                  lo.sample(x, y))))(int(pid))
                kf_all = (lambda pid_: (lambda x, y: S.surfaces_at_polygon(x, y, pid_)))(int(pid))
                for part, comp in road_mesh.split_by_surface(piece, S, int(pid), cut=cbox.exterior):
                    kf = kf_all if comp is None else (lambda x, y, c=comp: np.full(np.shape(x), c))
                    for surf, zg in zones:
                        zpart = part if zg is None else road_mesh.polygonal(part.intersection(zg))
                        if zpart.is_empty or zpart.area < 1e-3:
                            continue
                        mat, cell, uvt = material(p["cls"], surf)
                        for comp_, V, T in road_mesh.mesh_polygon_surfaces(zpart, zf, kf, cell=cell):
                            if len(T) and np.isfinite(V[:, 2]).all():
                                emit(cx, cy, mat, uvt, V, T)
                                n_tri += len(T)
    return n_tri


def carve_tile(net, X0, Y0, xs, ys, surf, on_carve):
    """Terrain carve of one tile: every vertex whose terrain triangles reach a network mesh drops
    0.1 m under the lowest top face within one terrain step of it (road_mesh.carve_window; surf:
    road_mesh.TriSurface of the meshes as built, those of the neighbouring tiles too), so the terrain
    stays under every face, between its vertices too -> on_carve(rows, cols, z)."""
    import road_mesh
    X1, Y1 = X0 + TILE, Y0 + TILE
    ids = net.ptree.query(shapely.box(X0, Y0, X1, Y1), predicate="intersects")
    c0 = int(np.searchsorted(xs, X0)); c1 = int(np.searchsorted(xs, X1))
    r0 = int(np.searchsorted(ys, Y0)); r1 = int(np.searchsorted(ys, Y1))
    if len(ids) == 0 or c1 <= c0 or r1 <= r0:
        return
    sq = xs[1] - xs[0]
    tr = Affine(sq, 0, xs[c0] - 0.5 * sq, 0, sq, ys[r0] - 0.5 * sq)          # row 0 = south
    # the vertices within one step (in x and in y) of a mesh: their triangles can reach over it
    mask = features.rasterize([(net.polys[i]["geom"].buffer(sq), 1) for i in ids], out_shape=(r1 - r0, c1 - c0),
                              transform=tr, fill=0, dtype=np.uint8, all_touched=True).astype(bool)
    rr, cc = np.nonzero(mask)
    if not len(rr):
        return
    zc = road_mesh.carve_window(surf, xs, ys, r0 + rr, c0 + cc)
    ok = np.isfinite(zc)
    on_carve(r0 + rr[ok], c0 + cc[ok], zc[ok] - 0.10)


# --------------------------------------------------------------------------------------------------
# Dirt and gravel roads and paths flush with the ground (v2.7), in the built level.
#
# Up to v2.6 the terrain under every road mesh was carved 0.1 m under the lowest face within one
# terrain step (1.5 m) of each vertex (network_mesh.carve_tile): the terrain stays under the faces
# whatever the grade, but along a dirt track it leaves a trench 1.5 m wide on both sides and the
# track stands on it like a slab, its edge face showing 0.3-0.4 m above the ground (median; asphalt
# has kerbs and walls, an unpaved surface has none).
#
# Here the terrain around the unpaved meshes (osm_surface: gravel and dirt on roads, paths, yards)
# is raised to their surface instead:
# - every terrain vertex within SHOULDER m of an unpaved top face aims at the height of the nearest
#   point of the face, EPS under it; from there to BLEND m the aim fades back to the ground, only
#   where the ground is lower (an uphill bank stays as it is); no vertex goes down;
# - a vertex more than TALL m under that surface is left alone (a road on a wall or a bridge);
# - then the aims are lowered as little as needed to keep every terrain triangle under every road
#   face (unpaved by EPS, the others by EPS_PAVED), on the faces sampled SUB x SUB per terrain
#   square, for both ways the square may be split into triangles: a few passes spread each excess
#   on the corners by their weight, the last ones lower all three corners of an offending triangle
#   by the whole excess (never under the terrain as it was, which was safe). The blocks are solved
#   twice, the second time with the vertices of the neighbouring blocks as the first time left them.
# Then the outer edges of the unpaved meshes go down onto the raised terrain (EDGE_UP over it, at
# most EDGE_DROP m down), with the top of their edge faces: the edge between track and ground is
# 1-2 cm instead of 0.3 m. The edges against another surface (asphalt at a junction) and the seams
# between two tiles or two polygons stay where they are. The materials and grip, the AI roads and
# everything else are copied as they are.
#
# A finishing step of build_level.py (FINISH), on the built level; alone: python build_level.py --finish unpaved
# --------------------------------------------------------------------------------------------------

UNPAVED = tuple(m for g in osm_surface.MATS.values() for c, m in g.items() if c in ("gravel", "dirt"))
EPS = 0.04            # m, terrain under an unpaved face (above the 2.2 cm height step of the .ter)
EPS_PAVED = 0.05      # m, under the other faces (they were carved 0.1 m under the lowest face near)
SHOULDER = 1.5        # m from an unpaved face: the terrain aims at its height
BLEND = 4.5           # m ... fading back to the ground up to here
TALL = 1.0            # m, ground further under the face: a wall or a bridge, not a trench
EMB_FLAT = 0.5        # m (v3.0) of ground at the road's height beside a road higher than TALL over the ground
EMB_SLOPE = float(os.environ.get("MAGLIASO_EMB_SLOPE") or 0.67)   # then down 1:1.5 (an embankment) ...
EMB_MAX = float(os.environ.get("MAGLIASO_EMB_MAX") or 3.0)        # ... where that fills the ground by no more than this
# (deeper: a wall or a bridge). An external site on steep alpine slopes sets both lower (EXTERNAL_SITE.md): read
# from the environment, so the spawned workers of unpaved_step get them too
SUB = 5               # face samples per terrain step
PASSES = 12           # spreading passes before the last, safe one
EDGE_UP = 0.01        # m, an outer edge vertex of an unpaved face over the terrain
EDGE_DROP = 0.25      # m, deepest an edge vertex goes down; further over the terrain: a wall or a bridge
OUT = 0.10            # m beyond an edge: no road face there, the edge is the outer one
BLOCK = 256           # terrain vertices per block side
PAD = 4               # vertices of context around a block (BLEND / TER_SQUARE, rounded up)


def read_items(zi, name):
    return [json.loads(l) for l in zi.read(name).decode("utf-8").splitlines() if l.strip()]


def top_faces(zi, objs, mats=None):
    """(faces of the materials `mats` (default UNPAVED), all) top faces (k, 3, 3) of the TSStatic shapes
    `objs` of the zip."""
    mats = UNPAVED if mats is None else mats
    un, al = [], []
    for o in objs:
        sn = o.get("shapeName", "")
        if o.get("class") != "TSStatic" or not sn.startswith("/levels/"):
            continue
        name = sn.lstrip("/")
        if name not in zi.NameToInfo:
            continue
        V, _, _, _, parts, _ = optimize_level.parse(zi.read(name).decode("utf-8"))
        V = V + np.asarray(o.get("position", [0, 0, 0]), np.float64)
        for mat, idx in parts:
            t = V[idx[:, 0]].reshape(-1, 3, 3)
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            t = t[n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12) > 0.5]
            al.append(t)
            if mat in mats:
                un.append(t)
    return np.concatenate(un), np.concatenate(al)


def read_ter(data):
    n = struct.unpack("<I", data[1:5])[0]
    q = np.frombuffer(data[5:5 + 2 * n * n], "<u2").reshape(n, n)
    return n, q


def corner_weights(u, v):
    """Corners (00, 01, 10, 11 as 0..3: row offset * 2 + column offset) and barycentric weights of
    the points (u along x, v along y, 0..1 in a square) in the triangles of both splits:
    two (k, 3) index and weight arrays per split."""
    out = []
    a = u >= v                                       # split 00-11
    ia = np.where(a[:, None], [0, 1, 3], [0, 2, 3])
    wa = np.where(a[:, None], np.column_stack([1 - u, u - v, v]), np.column_stack([1 - v, v - u, u]))
    out.append((ia, wa))
    b = u + v <= 1                                   # split 01-10
    ib = np.where(b[:, None], [0, 1, 2], [3, 1, 2])
    wb = np.where(b[:, None], np.column_stack([1 - u - v, u, v]), np.column_stack([u + v - 1, 1 - v, 1 - u]))
    out.append((ib, wb))
    return out


# faces of the whole level, set in main() and in every worker (pool(): Windows spawns the workers)
UN = ALL = None
Z0 = MAXH = None
WORKERS = int(os.environ.get("PATCH_WORKERS", min(os.cpu_count(), 4)))   # each holds the faces of the level (~1.5 GB)


_BED = _RD = None
RAIL_CLEAR = 8.0      # m from the axis of a track where the edges of the roads raise no ground


def rail_dist():
    """f(x, y): distance to the nearest track axis (railway.tracks, bridges included), once per process."""
    global _RD
    if _RD is None:
        import railway
        from scipy.spatial import cKDTree
        P = [railway.resample(Q, 0.5)[0][:, :2] for _, Q in railway.tracks()]
        if not P:
            _RD = lambda x, y: np.full(np.shape(x), np.inf)
        else:
            tree = cKDTree(np.concatenate(P))
            _RD = lambda x, y: tree.query(np.column_stack([np.ravel(x), np.ravel(y)]),
                                          distance_upper_bound=50.0)[0].reshape(np.shape(x))
    return _RD


def rail_bed():
    """railway.bed_fn, once per process (the workers too)."""
    global _BED
    if _BED is None:
        import railway
        _BED = railway.bed_fn()
    return _BED


def block(args):
    """New heights of the vertices of one block: (rows, cols, z) of the vertices that change.
    Q: the heights as they were (block with PAD vertices around); QA: heights to check instead of
    the aims (the second pass, on the blocks merged: a block only knows its own vertices)."""
    r0, c0, Q, QA = args
    sq, f = TER_SQUARE, TER_SQUARE / SUB
    H = Z0 + Q.astype(np.float64) / 65535.0 * MAXH
    nr, nc = H.shape
    xs = TER_X0 + (c0 - PAD + np.arange(nc)) * sq
    ys = TER_Y0 + (r0 - PAD + np.arange(nr)) * sq
    # face samples at the centres of SUB x SUB cells of every terrain square
    fx = xs[0] + (np.arange((nc - 1) * SUB) + 0.5) * f
    fy = ys[0] + (np.arange((nr - 1) * SUB) + 0.5) * f
    FX, FY = np.meshgrid(fx, fy)
    hu = UN.height(FX.ravel(), FY.ravel(), "low").reshape(FX.shape)
    U = np.isfinite(hu)
    if not U.any():
        return None
    ha = ALL.height(FX.ravel(), FY.ravel(), "low").reshape(FX.shape)
    # aim of every vertex: the nearest unpaved sample (distance from the vertex)
    d, (ir, ic) = ndi.distance_transform_edt(~U, sampling=f, return_indices=True)
    vr = np.clip(np.arange(nr) * SUB - 1, 0, U.shape[0] - 1)       # a sample next to every vertex
    vc = np.clip(np.arange(nc) * SUB - 1, 0, U.shape[1] - 1)
    D = d[np.ix_(vr, vc)] - 0.5 * f * np.sqrt(2)
    R = hu[ir[np.ix_(vr, vc)], ic[np.ix_(vr, vc)]] - EPS
    w = np.clip((BLEND - D) / (BLEND - SHOULDER), 0.0, 1.0)
    aim = np.where(w > 0, H + w * (R - H), H)
    aim = np.where((R - H < TALL) & (D < BLEND), np.maximum(aim, H), H)
    # v3.0: a road over the ground by TALL m or more stands on an embankment, not on a vertical face of asphalt:
    # the ground at its height for EMB_FLAT m, then down at EMB_SLOPE to the ground, filled by at most EMB_MAX m
    emb = R - np.maximum(D - EMB_FLAT, 0.0) * EMB_SLOPE
    rim = (D > 0) | ((D > -TER_SQUARE) & ndi.maximum_filter(D > 0, size=3))     # outside, or under the edge
    fill = (R - H >= TALL) & rim & (emb > H) & (emb - H <= EMB_MAX)
    aim = np.where(fill, np.maximum(aim, emb), aim)
    # v3.0: nothing raised into the cutting of a railway (railway.bed_fn): beside the Strada Cantonale over the line
    # along the lake the embankment of the road filled the bed of the tracks up to 3 m, in teeth along the rails
    X, Y = np.meshgrid(xs, ys)
    bed = rail_bed()(X, Y)
    aim = np.where(np.isfinite(bed), np.minimum(aim, np.maximum(H, bed)), aim)
    # and no ground raised within RAIL_CLEAR m of a track at all: a pavement of the Strada Cantonale sloping down
    # over the line (the survey's surface fitted on the slope) raised the ground beside the rails by 2-3 m
    aim = np.where(rail_dist()(X, Y) < RAIL_CLEAR, H, aim)
    if QA is not None:
        aim = Z0 + QA.astype(np.float64) / 65535.0 * MAXH
    moved = aim > H + 1e-4
    if not moved.any():
        return None
    # the constraints: samples on a face, in a square with a corner that moved
    sqm = moved[:-1, :-1] | moved[:-1, 1:] | moved[1:, :-1] | moved[1:, 1:]
    near = np.repeat(np.repeat(sqm, SUB, 0), SUB, 1) & np.isfinite(ha)
    sr, sc = np.nonzero(near)
    lim = ha[sr, sc] - np.where(U[sr, sc] & (np.abs(hu[sr, sc] - ha[sr, sc]) < 1e-6), EPS, EPS_PAVED)
    qr, qc = sr // SUB, sc // SUB                                 # square of every sample
    u = (sc % SUB + 0.5) / SUB
    v = (sr % SUB + 0.5) / SUB
    flat = lambda k: (qr + k // 2) * nc + (qc + k % 2)
    corners = [flat(k) for k in range(4)]
    splits = []
    for idx, wt in corner_weights(u, v):
        vid = np.stack([np.choose(idx[:, j], corners) for j in range(3)], 1)
        splits.append((vid, wt))
    Z = aim.ravel().copy()
    Hf = H.ravel()

    def excess():
        for vid, wt in splits:
            ex = (Z[vid] * wt).sum(1) - lim
            bad = (ex > 0) & (Z[vid] > Hf[vid] + 1e-6).any(1)     # a square as it was is safe
            yield vid[bad], wt[bad], ex[bad]
    for p in range(PASSES):                     # least change of the corners for every sample
        drop = np.zeros_like(Z)
        for vid, wt, ex in excess():
            corr = ex[:, None] * wt / (wt ** 2).sum(1)[:, None]
            np.maximum.at(drop, vid.ravel(), corr.ravel())
        if not drop.any():
            break
        Z = np.maximum(Z - drop, Hf)            # never under the ground as it was (it was safe)
    for p in range(50):                         # safe: the whole excess on every corner
        drop = np.zeros_like(Z)
        for vid, wt, ex in excess():
            np.maximum.at(drop, vid.ravel(), np.repeat(ex, 3))
        if not drop.any():
            break
        Z = np.maximum(Z - drop, Hf)
    else:                                       # still crossing: those corners back where they were
        for vid, wt, ex in excess():
            Z[vid.ravel()] = Hf[vid.ravel()]
    Z = Z.reshape(nr, nc)
    core = (slice(PAD, nr - PAD), slice(PAD, nc - PAD))
    ch = Z[core] > H[core] + 1e-4
    if not ch.any():
        return None
    rr, cc = np.nonzero(ch)
    return r0 + rr, c0 + cc, Z[core][rr, cc], D[core][rr, cc], R[core][rr, cc]


def terrain_top(Zt, x, y):
    """Height of the terrain at points, the higher of the two ways a square may be split."""
    c = (x - TER_X0) / TER_SQUARE
    r = (y - TER_Y0) / TER_SQUARE
    ci, ri = np.floor(c).astype(np.int64), np.floor(r).astype(np.int64)
    u, v = c - ci, r - ri
    z00, z01, z10, z11 = (Zt[ri + i, ci + j].astype(np.float64) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    a = np.where(u >= v, z00 + u * (z01 - z00) + v * (z11 - z01), z00 + v * (z10 - z00) + u * (z11 - z10))
    b = np.where(u + v <= 1, z00 + u * (z01 - z00) + v * (z10 - z00), z11 + (1 - u) * (z10 - z11) + (1 - v) * (z01 - z11))
    return Z0 + np.maximum(a, b) / 65535.0 * MAXH


QT = None             # the patched terrain (uint16): main() and terrain() in the workers


def init(files, z0, maxh):
    global UN, ALL, Z0, MAXH
    if UN is None:                    # spawned: not a fork of main()
        UN, ALL = (road_mesh.TriSurface(np.load(files[k])) for k in ("un", "al"))
    Z0, MAXH = z0, maxh


def pool(tmp, un, al):
    """Workers with the faces of the whole level: through .npy files in `tmp`, too big for the pipe
    of a spawned worker."""
    files = {k: os.path.join(tmp, k + ".npy") for k in ("un", "al")}
    np.save(files["un"], un)
    np.save(files["al"], al)
    return ProcessPoolExecutor(max_workers=WORKERS, initializer=init, initargs=(files, Z0, MAXH))


def terrain(path):
    """The patched terrain in a worker (written by main() after the workers started)."""
    global QT
    if QT is None:
        QT = np.load(path, mmap_mode="r")
    return QT


def taper(args):
    """The outer edges of the unpaved faces of one shape down onto the terrain: new DAE text, or None.
    An edge vertex goes down to EDGE_UP over the terrain where that is at most EDGE_DROP m down, with
    the top of the edge face under it; the vertices of the faces of another surface, and those on
    an edge with a road face beyond it (a seam between two tiles or two polygons), stay.
    v2.8: args may add the materials (default UNPAVED) and the deepest drop (default EDGE_DROP)."""
    name, text, pos, qpath = args[:4]
    mats = args[4] if len(args) > 4 else UNPAVED
    edge_drop = args[5] if len(args) > 5 else EDGE_DROP
    qt = terrain(qpath)
    V, _, _, _, parts, _ = optimize_level.parse(text)
    W = V + pos
    key = np.round(W * 1000).astype(np.int64)
    _, pid = np.unique(key, axis=0, return_inverse=True)
    pid = pid.ravel()
    tops, kinds, steep = [], [], []
    for mat, idx in parts:
        T = idx[:, 0].reshape(-1, 3)
        t = W[T]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        up = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12) > 0.5
        tops.append(T[up])
        kinds.append(np.full(up.sum(), mat in mats))
        if mat in mats:
            steep.append(T[~up].ravel())
    T = np.concatenate(tops)
    un = np.concatenate(kinds)
    if not un.any():
        return None
    P = pid[T]
    blocked = np.zeros(pid.max() + 1, bool)
    blocked[P[~un].ravel()] = True
    E = np.concatenate([P[:, [0, 1]], P[:, [1, 2]], P[:, [2, 0]]])
    F = np.tile(np.arange(len(T)), 3)
    Es = np.sort(E, 1)
    _, inv, cnt = np.unique(Es, axis=0, return_inverse=True, return_counts=True)
    b = (cnt[inv.ravel()] == 1) & un[F]
    Eb, Fb = E[b], F[b]
    rep = np.zeros(pid.max() + 1, np.int64)               # one vertex of every position
    rep[pid] = np.arange(len(pid))
    A, B = W[rep[Eb[:, 0]]], W[rep[Eb[:, 1]]]
    mid = 0.5 * (A + B)
    d = B[:, :2] - A[:, :2]
    nrm = np.column_stack([d[:, 1], -d[:, 0]]) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
    cen = W[T[Fb]].mean(1)
    nrm *= np.where(((mid[:, :2] - cen[:, :2]) * nrm).sum(1) < 0, -1.0, 1.0)[:, None]
    q = mid[:, :2] + OUT * nrm
    outer = np.isnan(ALL.height(q[:, 0], q[:, 1], "low"))
    on_edge = np.zeros(len(blocked), bool)
    inner_edge = np.zeros(len(blocked), bool)
    on_edge[Eb.ravel()] = True
    inner_edge[Eb[~outer].ravel()] = True
    move = np.flatnonzero(on_edge & ~inner_edge & ~blocked)
    if not len(move):
        return None
    X = W[rep[move]]
    drop = X[:, 2] - (terrain_top(qt, X[:, 0], X[:, 1]) + EDGE_UP)
    ok = (drop > 0.002) & (drop <= edge_drop)
    move, drop, X = move[ok], drop[ok], X[ok]
    if not len(move):
        return None
    dz = np.zeros(len(blocked))
    dz[move] = drop
    V2 = V.copy()
    V2[:, 2] -= dz[pid]
    # the tops of the edge faces (the strips of optimize_level moved them by up to 4 mm)
    sv = np.unique(np.concatenate(steep)) if steep else np.zeros(0, np.int64)
    sv = sv[dz[pid[sv]] == 0]
    if len(sv):
        from scipy.spatial import cKDTree
        dist, j = cKDTree(X).query(W[sv], distance_upper_bound=0.01)
        hit = np.isfinite(dist)
        V2[sv[hit], 2] -= drop[j[hit]]
    fmt = " ".join(("%.3f" % v).rstrip("0").rstrip(".") for v in V2.ravel())
    out = re.sub(r'(<float_array id="g-pa" count="\d+">)[^<]*(</float_array>)',
                 lambda m: m.group(1) + fmt + m.group(2), text, count=1)
    return name, out, len(move)


def cell_ids(x, y):
    """Ids of the 1 m cells (over the terrain block) of points."""
    from config import TER_SIZE
    w = int(np.ceil(TER_SIZE * TER_SQUARE))
    return np.floor(np.asarray(y) - TER_Y0).astype(np.int64) * w + np.floor(np.asarray(x) - TER_X0).astype(np.int64)


def in_cells(cells, x, y):
    """Points in the sorted cell ids `cells`."""
    c = cell_ids(x, y)
    i = np.clip(np.searchsorted(cells, c), 0, max(len(cells) - 1, 0))
    return (cells[i] == c) if len(cells) else np.zeros(len(c), bool)


def pits(q0, road_z, rr, cc):
    """v3.1 review: the vertices (rr, cc) that lie in a trench: ground higher than the road face they rise to (road_z,
    m) by PIT_UP m within PIT_N terrain steps, not under a road face (the road carve dug the trench between the road
    and the slope it climbs into, review point 0576: 1.7 m deep beside a wall). Filled up to the road, the ground
    there stays under the ground around it: no wall or house beside it is buried deeper than the slope buries it.
    Beyond a wall that holds a road the ground within reach is the road's own height or lower: no trench."""
    n = q0.shape[0]
    offs = [(a, b) for a in range(-PIT_N, PIT_N + 1) for b in range(-PIT_N, PIT_N + 1) if a or b]
    R = np.clip(rr[:, None] + np.array([a for a, _ in offs]), 0, n - 1)
    C = np.clip(cc[:, None] + np.array([b for _, b in offs]), 0, n - 1)
    lin, inv = np.unique(R * n + C, return_inverse=True)
    road = np.isfinite(ALL.height(TER_X0 + (lin % n) * TER_SQUARE, TER_Y0 + (lin // n) * TER_SQUARE, "low"))
    h = np.where(road, -np.inf, Z0 + q0.ravel()[lin] / 65535.0 * MAXH)[inv.reshape(R.shape)]
    return h.max(1) >= road_z[rr, cc] + PIT_UP


def unpaved_step(root, mats=None, edge_drop=EDGE_DROP, keep_out=None, keep_out_lift=0.0, hold=None):
    """mats: the materials whose surroundings are raised and whose outer edges go down (default UNPAVED);
    edge_drop: the deepest an edge goes down (0: the faces stay as they are, only the terrain is raised);
    keep_out: sorted 1 m cell ids (cell_ids) whose faces raise no terrain and whose terrain vertices stay as they
    were (v2.8, network_mesh.paved_edges_step); keep_out_lift > 0 (v3.0 fix round): there the faces raise the
    vertices within SHOULDER m of them by up to this much (m), the others stay as they were. hold(x, y, d) (v3.1
    review): True for the keep-out vertices (x, y, d m from the faces) that rise as anywhere else (nothing marks the
    edge between them and the faces); with it, the keep-out vertices in a trench (pits) rise too."""
    global UN, ALL, Z0, MAXH, QT
    mats = UNPAVED if mats is None else tuple(mats)
    zi = bng.LevelFiles(root)
    lv = f"levels/{LEVEL_NAME}"
    objs = read_items(zi, f"{lv}/main/MissionGroup/roads/surfaces/items.level.json")
    un, al = top_faces(zi, objs, mats)
    if keep_out is not None and keep_out_lift <= 0:     # no terrain raised to the faces there
        c = un.mean(1)
        un = un[~in_cells(keep_out, c[:, 0], c[:, 1])]
    print("top faces: %d of %s, %d in all" % (len(un), "the unpaved" if mats == UNPAVED else "these", len(al)))
    UN, ALL = road_mesh.TriSurface(un), road_mesh.TriSurface(al)
    blk = next(o for o in read_items(zi, f"{lv}/main/MissionGroup/level_objects/terrain/items.level.json")
               if o.get("class") == "TerrainBlock")
    Z0, MAXH = float(blk["position"][2]), float(blk["maxHeight"])
    ter_name = f"{lv}/theTerrain.ter"
    data = zi.read(ter_name)
    n, q = read_ter(data)
    q = q.copy()
    # blocks with an unpaved face
    cr = np.floor((un[:, :, 1].mean(1) - TER_Y0) / TER_SQUARE / BLOCK).astype(int)
    cc = np.floor((un[:, :, 0].mean(1) - TER_X0) / TER_SQUARE / BLOCK).astype(int)
    keys = sorted(set(zip(cr.tolist(), cc.tolist())))
    q0 = q.copy()
    dist = np.full(q.shape, np.inf, np.float32) if keep_out is not None else None    # m from the faces, raised vertices
    road_z = np.full(q.shape, np.nan, np.float32) if keep_out is not None else None  # the height of those faces
    tmp = tempfile.mkdtemp(prefix="unpaved_")

    ex = pool(tmp, un, al)

    def run(qa):
        jobs = []
        for br, bc in keys:
            r0, c0 = br * BLOCK, bc * BLOCK
            ra, rb, ca, cb = r0 - PAD, r0 + BLOCK + PAD + 1, c0 - PAD, c0 + BLOCK + PAD + 1
            if ra < 0 or ca < 0 or rb > n or cb > n:
                continue                                # the edge of the terrain: no roads there
            jobs.append((r0, c0, q0[ra:rb, ca:cb], None if qa is None else qa[ra:rb, ca:cb]))
        out = q0.copy()
        for res in ex.map(block, jobs, chunksize=2):
            if res is not None:
                rr, cc, z, dd, rz = res
                qn = np.floor((z - Z0) / MAXH * 65535.0).astype(np.int64)    # down: never over a face
                up = qn > q0[rr, cc]
                out[rr[up], cc[up]] = qn[up].astype(np.uint16)
                if keep_out is not None:
                    near = dd[up] < dist[rr[up], cc[up]]
                    dist[rr[up][near], cc[up][near]] = dd[up][near]
                    road_z[rr[up][near], cc[up][near]] = rz[up][near]
        return out
    print("%d blocks with unpaved faces" % len(keys))
    q = run(None)
    q = run(q)          # the vertices along the edges of the blocks, checked with both sides known
    if keep_out is not None:                    # the ground there as it was, also where a face farther off raised it
        rr, cc = np.nonzero(q != q0)
        back = in_cells(keep_out, TER_X0 + cc * TER_SQUARE, TER_Y0 + rr * TER_SQUARE)
        if keep_out_lift > 0:
            # v3.0 fix round: but the trench of the carve beside the road (0.1 m under the lowest face within one
            # terrain step) filled there too, where the edge stands at most keep_out_lift m over the ground within
            # SHOULDER m of it: a real step (a wall, a kerb, a plinth, the bank behind a guard rail) is higher, and
            # a vertex that would rise more stays as it was, not raised part of the way
            lift = (q[rr, cc].astype(np.float64) - q0[rr, cc]) / 65535.0 * MAXH
            ok = (dist[rr, cc] <= SHOULDER) & (lift <= keep_out_lift)
            print("terrain vertices in the keep-out raised to the edge of the road beside them: %d" % (back & ok).sum())
            back &= ~ok
        if hold is not None and back.any():
            free = hold(TER_X0 + cc[back] * TER_SQUARE, TER_Y0 + rr[back] * TER_SQUARE, dist[rr[back], cc[back]])
            print("terrain vertices in the keep-out with nothing between them and the road: %d" % free.sum())
            back[np.flatnonzero(back)[free]] = False
        if hold is not None and back.any():
            free = pits(q0, road_z, rr[back], cc[back])
            print("terrain vertices in the keep-out in a trench between the road and higher ground: %d" % free.sum())
            back[np.flatnonzero(back)[free]] = False
        q[rr[back], cc[back]] = q0[rr[back], cc[back]]
        print("terrain vertices in the keep-out left as they were: %d" % back.sum())
    ch = q != q0
    lift = (q[ch].astype(np.float64) - q0[ch]) / 65535.0 * MAXH
    print("terrain vertices raised: %d, by median %.2f m, 95th percentile %.2f m, at most %.2f m"
          % (ch.sum(), np.median(lift), np.percentile(lift, 95), lift.max()))
    QT = q
    qpath = os.path.join(tmp, "qt.npy")
    np.save(qpath, q)
    jobs = []
    for o in objs if edge_drop > 0 else ():             # edge_drop 0: the faces stay as they are
        sn = o.get("shapeName", "").lstrip("/")
        if o.get("class") == "TSStatic" and sn in zi.NameToInfo:
            text = zi.read(sn).decode("utf-8")
            if any(f'material="{m}-mat"' in text for m in mats):
                jobs.append((sn, text, np.asarray(o.get("position", [0, 0, 0]), np.float64), qpath, mats, edge_drop))
    shapes, nmove = {}, 0
    with ex:
        for res in ex.map(taper, jobs, chunksize=1):
            if res is not None:
                shapes[res[0]] = res[1].encode("utf-8")
                nmove += res[2]
    print("edge vertices lowered onto the terrain: %d, in %d shapes" % (nmove, len(shapes)))
    new_ter = data[:5] + q.astype("<u2").tobytes() + data[5 + 2 * n * n:]
    with zi.writer() as zo:
        now = time.localtime()[:6]
        for i in zi.infolist():
            if i.filename == ter_name or i.filename in shapes:
                # a new date: the game converts the shapes again instead of taking its cached ones
                ni = zipfile.ZipInfo(i.filename, now)
                ni.compress_type, ni.external_attr = i.compress_type, i.external_attr
                zo.writestr(ni, new_ter if i.filename == ter_name else shapes[i.filename])
            else:
                zo.writestr(i, zi.read(i), compress_type=i.compress_type)
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------------------
# Paved roads, yards and pavements flush with the ground where nothing marks their edge (v2.8), in a built
# level.
#
# On v2.7 the outer edges of the paved surfaces (asphalt, setts, cobbles, pavements) stood over the terrain like a
# slab with a trench beside it: the build carves the terrain 0.1 m under the lowest road face within one terrain
# step (network_mesh.carve_tile), so that it stays under the road whatever the grade. The dirt and gravel tracks
# were laid flush in v2.7 (network_mesh.unpaved_step: the terrain raised to them and their edges lowered onto it); here the
# terrain is raised to the paved surfaces with network_mesh.main, and the surfaces stay as they are:
# - the terrain within 1.5 m of a paved top face raised to network_mesh.EPS m under its height (fading back to the
#   ground at 4.5 m, only where the ground is lower; v3.0: where it lies 1-3 m under the road, an embankment
#   0.5 m flat then 1:1.5 down to the ground instead of a vertical face of asphalt), never over a road face;
# - the faces are not lowered (EDGE_DROP 0): lowering the outer edges tilts the last strip of a road, where the
#   wheels of a car run, and drive_test.py counted 40-60 % more hard knocks on the roads (tried in the first
#   version of this patch);
# - the paved paths are left out: the drive test runs its car along them with the wheels on the ground beside
#   them, which the raised terrain would change;
# - none of it within KEEP_OUT m of a guard rail, a fence, a wall (and the backfill behind it), a building, the
#   railway or a bridge parapet: there the step is the real one (an embankment behind a guard rail, a kerb
#   against a wall, a plinth). The faces there raise no terrain, and the terrain vertices there stay as they were
#   also where a face farther off would raise them (the first version left them to the fade of the faces up to
#   4.5 m away: the ground rose against walls, fences and houses, up to 1 m).
# The terrain gets the date of the build step; everything else is copied as it is. The report measures the edges
# before and after.
#
# A finishing step of build_level.py (FINISH), on the built level; alone: python build_level.py --finish paved_edges
# --------------------------------------------------------------------------------------------------

PAVED = tuple(m for g in osm_surface.MATS.values() for c, m in g.items()
              if c in ("hard", "sett", "cobble") and not m.startswith("mp_path_")) + ("mp_road_asphalt_fresh", "mp_sidewalk")
PAVED_EDGE_DROP = 0.0       # m: the faces stay as they are
KEEP_OUT = 2.0              # m around guard rails, fences, walls, buildings, railway, parapets
# v3.0 fix round: in the keep-out the ground within SHOULDER m of a paved edge rises to it where that is at most this
# much (m). Up to v3.0 it stayed as the carve left it: 530 of 973 km of edges, most of those in the villages (houses,
# walls and fences on both sides of the narrow roads of Vezio, Arosio, Mugena ...), kept the v2.7 slab with a trench
# beside it, median 0.21 m, and a wheel leaving the asphalt dropped off and hit the edge coming back.
KEEP_OUT_LIFT = 0.35
# v3.1 review (the visual check of every road, group 6): the keep-out is a 2 m band of 1 m cells around anything that
# could mark an edge, and 124 of the 128 km of paved edges standing more than 0.5 m over the ground lay in it: yards
# and road ends beside a house, a fence or a wall that does not hold them stood as slabs with vertical sides of
# asphalt (review points 0373, 0479, 0576 among them). In the keep-out a vertex rises as elsewhere (the embankment of a yard
# 1-3 m over the ground) where every such thing is at least HOLD m farther from it than the road faces are: no wall
# stands between it and the road, nor along the edge. The things are the outline samples of the keep-out groups,
# every HOLD_GRID m.
HOLD = 1.0
HOLD_GRID = 0.25
# ... and (network_mesh.pits) where it lies in a trench the carve dug between the road and higher ground: most of those
# edges were such trenches, up to 1.7 m deep beside a road climbing into a slope (point 0576). On the v3.1 level the two
# rules raise 227,040 vertices more (hold alone 43,446) and the edges over 0.5 m go from 127.8 to 49.9 km.
PIT_N = 2
PIT_UP = 0.3
SAMPLE = 0.5                # m between the samples along the edges of their faces
KEEP_GROUPS = ("roads/guardrails", "roads/fences", "walls", "buildings", "railway", "level_objects/Water")
PARAPET = ("mp_bridge_parapet",)


def outline_samples(t):
    """Points every SAMPLE m along the edges of triangles (k, 3, 2+), and their centres."""
    t = t[:, :, :2]
    pts = [t.mean(1)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        L = np.linalg.norm(t[:, b] - t[:, a], axis=1)
        k = np.maximum(np.ceil(L / SAMPLE).astype(np.int64), 1)
        rep = np.repeat(np.arange(len(t)), k + 1)
        start = np.repeat(np.cumsum(k + 1) - (k + 1), k + 1)
        f = (np.arange(len(rep)) - start) / k[rep]
        pts.append(t[rep, a] + f[:, None] * (t[rep, b] - t[rep, a]))
    return np.concatenate(pts)


def keep_out_cells(zi, lv, points=None):
    """Sorted 1 m cell ids (network_mesh.cell_ids) within KEEP_OUT m of what marks a road's edge. points: a dict
    that gets the outline samples of those things, every HOLD_GRID m (xy, "buildings" and "other")."""
    ids = []

    def keep_points(g, P):
        if points is not None:
            k = "buildings" if g == "buildings" else "other"
            points.setdefault(k, []).append(np.unique(np.round(P[:, :2] / HOLD_GRID).astype(np.int64), axis=0))
    for g in KEEP_GROUPS:
        f = f"{lv}/main/MissionGroup/{g}/items.level.json"
        if f not in zi.NameToInfo:
            continue
        for o in pl.read_items(zi, f):
            sn = o.get("shapeName", "").lstrip("/")
            if o.get("class") != "TSStatic" or sn not in zi.NameToInfo:
                continue
            V, _, _, _, parts, _ = optimize_level.parse(zi.read(sn).decode("utf-8"))
            W = V + np.asarray(o.get("position", [0, 0, 0]), np.float64)
            t = W[np.concatenate([idx[:, 0] for _, idx in parts])].reshape(-1, 3, 3)
            P = outline_samples(t)
            ids.append(np.unique(pu.cell_ids(P[:, 0], P[:, 1])))
            keep_points(g, P)
    import guardrail_mesh
    gr = guardrail_mesh.module_faces(zi)                # v2.8: the guard rails are forest items
    if len(gr):
        P = outline_samples(gr)
        ids.append(np.unique(pu.cell_ids(P[:, 0], P[:, 1])))
        keep_points("guardrails", P)
    for m, t in pl.faces(zi, lv, ["roads/surfaces"]).items():
        if m in PARAPET:
            P = outline_samples(t)
            ids.append(np.unique(pu.cell_ids(P[:, 0], P[:, 1])))
            keep_points("parapets", P)
    base = np.unique(np.concatenate(ids))
    if points is not None:
        for k in list(points):
            points[k] = np.unique(np.concatenate(points[k]), axis=0) * HOLD_GRID
    from config import TER_SIZE, TER_SQUARE
    w = int(np.ceil(TER_SIZE * TER_SQUARE))
    r = int(np.ceil(KEEP_OUT))
    offs = [dy * w + dx for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= KEEP_OUT ** 2]
    return np.unique(np.concatenate([base + o for o in offs]))


def measure(zi, keep_out):
    """The outer edges of the paved faces of a zip, out of the keep-out cells: their height over the terrain
    0.3 m beyond them (median, 75th and 90th percentiles, shares over 0.15 and 0.3 m), their length, and the
    terrain over the faces 0.15 and 0.5 m inside them (it should be nowhere)."""
    import unpaved_tour
    lv = f"levels/{LEVEL_NAME}"
    blk = next(o for o in pl.read_items(zi, f"{lv}/main/MissionGroup/level_objects/terrain/items.level.json")
               if o.get("class") == "TerrainBlock")
    z0, maxh = pu.Z0, pu.MAXH
    pu.Z0, pu.MAXH = float(blk["position"][2]), float(blk["maxHeight"])
    _, q, _, _ = pw.read_ter(zi.read(f"{lv}/theTerrain.ter"))
    road = pl.faces(zi, lv, ["roads/surfaces"])
    un = np.concatenate([road[m] for m in PAVED if m in road])
    ALL = road_mesh.TriSurface(np.concatenate(list(road.values())))
    mid, t, n, ln = unpaved_tour.edges(un)
    q_out = mid[:, :2] + 0.10 * n
    m = np.isnan(ALL.height(q_out[:, 0], q_out[:, 1], "low")) & (ln > 0.3) & ~pu.in_cells(keep_out, mid[:, 0], mid[:, 1])
    p = mid[m, :2] + 0.3 * n[m]
    gap = mid[m, 2] - pu.terrain_top(q, p[:, 0], p[:, 1])
    inside = []
    for k in (0.15, 0.5):
        p = mid[m, :2] - k * n[m]
        h = ALL.height(p[:, 0], p[:, 1], "near", mid[m, 2])
        ok = np.isfinite(h)
        inside.append(pu.terrain_top(q, p[ok, 0], p[ok, 1]) - h[ok])
    inside = np.concatenate(inside)
    pu.Z0, pu.MAXH = z0, maxh
    return {"edges_km": round(float(ln[m].sum()) / 1000, 1),
            "over_terrain_m": {k: round(float(np.percentile(gap, v)), 3) for k, v in (("p10", 10), ("median", 50),
                                                                                      ("p75", 75), ("p90", 90))},
            "share_over_0.15m": round(float((gap > 0.15).mean()), 3), "share_over_0.3m": round(float((gap > 0.3).mean()), 3),
            "terrain_over_the_faces_inside": {"share": round(float((inside > 0).mean()), 5),
                                              "max_m": round(float(inside.max()), 3)}}


def raised_vertices(q0, zi, keep_out):
    """The terrain vertices of the level (zi) raised over the terrain q0: in all, by how much, and in the
    keep-out (none)."""
    lv = f"levels/{LEVEL_NAME}"
    q1 = pw.read_ter(zi.read(f"{lv}/theTerrain.ter"))[1]
    blk = next(o for o in pl.read_items(zi, f"{lv}/main/MissionGroup/level_objects/terrain/items.level.json")
               if o.get("class") == "TerrainBlock")
    rr, cc = np.nonzero(q1 != q0)
    dz = (q1[rr, cc].astype(np.float64) - q0[rr, cc]) / 65535.0 * float(blk["maxHeight"])
    keep = pu.in_cells(keep_out, pu.TER_X0 + cc * pu.TER_SQUARE, pu.TER_Y0 + rr * pu.TER_SQUARE)
    return {"vertices": len(rr), "median_m": round(float(np.median(dz)), 3) if len(dz) else 0.0,
            "p95_m": round(float(np.percentile(dz, 95)), 3) if len(dz) else 0.0,
            "max_m": round(float(dz.max()), 3) if len(dz) else 0.0, "in_keep_out": int(keep.sum())}


def paved_edges_step(root, report=None):
    zi = bng.LevelFiles(root)
    lv = f"levels/{LEVEL_NAME}"
    pts = {}
    keep = keep_out_cells(zi, lv, pts)
    print("keep-out: %.1f km2 around guard rails, fences, walls, buildings, railway, parapets" % (len(keep) / 1e6),
          flush=True)
    from scipy.spatial import cKDTree
    things = cKDTree(np.concatenate(list(pts.values())))

    def hold(x, y, d):
        s = things.query(np.column_stack([x, y]), distance_upper_bound=KEEP_OUT + BLEND + HOLD)[0]
        return s > np.maximum(d, 0.0) + HOLD
    before = measure(zi, keep)
    print("before:", before, flush=True)
    q0 = pw.read_ter(zi.read(f"{lv}/theTerrain.ter"))[1].copy()
    pu.unpaved_step(root, mats=PAVED, edge_drop=PAVED_EDGE_DROP, keep_out=keep, keep_out_lift=KEEP_OUT_LIFT, hold=hold)
    zi = bng.LevelFiles(root)
    after = measure(zi, keep)
    print("after:", after, flush=True)
    raised = raised_vertices(q0, zi, keep)
    print("terrain vertices raised:", raised, flush=True)
    if report:
        os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
        json.dump({"materials": list(PAVED), "edge_drop_m": PAVED_EDGE_DROP,
                   "terrain_under_faces_m": pu.EPS, "keep_out_m": KEEP_OUT, "keep_out_lift_m": KEEP_OUT_LIFT, "keep_out_km2": round(len(keep) / 1e6, 2),
                   "paved_outer_edges_before": before, "paved_outer_edges_after": after,
                   "terrain_vertices_raised": raised}, open(report, "w"), indent=1)
    return 0
