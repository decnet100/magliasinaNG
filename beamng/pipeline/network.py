"""The road and path network of the playable area (v2.0), from swissTLM3D.

Every TLM_STRASSE line in the area (roads of every class, paths, mule tracks, stairs, fords and
bridges; not tunnels and underpasses) becomes a segment between two nodes; lines meet at shared
end points (swissTLM3D is noded at junctions; lines that only cross, like a road on a bridge over
another, stay apart). Segments are sampled into stations every STEP m.
Station width: the surveyed carriageway (MU 'strada_sentiero' polygons) measured with rays from the
line where it runs inside one (the line is moved to the middle of it when it is off by less than
MAX_SHIFT), otherwise the nominal width of the swissTLM3D class.
v3.1: on a bridge the survey draws the bridge (SOSF ponte_passerella), not the carriageway: a bridge line had the
nominal width of its class (4.5 m for a 4m Strasse) between approaches measured at 6-9 m, and the deck was narrower
than the road on most bridges. A bridge station outside the carriageways is measured in the bridge polygons, less
BRIDGE_RIM m on each side (the parapets); a bridge with neither is as wide as the narrower of the roads at its ends
(widen_bridges), and one measured in a bridge polygon no wider than its narrower approach + POLY_OVER m. A bridge
built on the survey's outline (ponti.json "deck": "survey") keeps its widths; ponti.json "width" sets one by hand.
v3.0.40 patch: an underpass under a railway (KUNSTBAUTE Unterfuehrung, under no road) is kept, and the track and the
paths over it become bridges there (over_underpass): skipped, Via Golf (Caslano - Magliaso) ran down into its cutting on
both sides of the FLP and ended against the embankment of the tracks.
v3.0.40: a dead end that runs down into the lake (a slipway: the whole Caslano shore came into the area) stops where
the ground (dtm05) reaches the lake level + WET m, so no road is drawn under the water.
Output: work/network.npz (stations and segments) and work/network.json (summary).
"""
import json, os, pickle
import numpy as np
import shapely
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from config import DATA, WORK, lv95_to_local
import area

STEP = 2.0              # m between stations
SNAP = 0.6              # m, line ends closer than this are one node
CLIP = 30.0             # m, the network continues this far beyond the area
MAX_SHIFT = 1.5         # m, largest move of a line onto the middle of the surveyed carriageway
# class: (nominal width m, kind, smoothing wavelength of the profile m)
CLASSES = {
    "Autobahn": (11.0, "road", 60.0), "Autostrasse": (9.0, "road", 55.0), "Ausfahrt": (6.0, "road", 40.0),
    "Einfahrt": (6.0, "road", 40.0), "Raststaette": (6.0, "road", 30.0), "Verbindung": (5.0, "road", 30.0),
    "10m Strasse": (10.0, "road", 50.0), "8m Strasse": (8.0, "road", 48.0), "6m Strasse": (6.0, "road", 45.0),
    "4m Strasse": (4.5, "road", 38.0), "3m Strasse": (3.2, "road", 30.0), "Zufahrt": (4.0, "road", 25.0),
    "Dienstzufahrt": (4.0, "road", 25.0), "Platz": (6.0, "road", 25.0), "Markierte Spur": (2.0, "path", 16.0),
    "2m Weg": (2.0, "path", 16.0), "2m Wegfragment": (2.0, "path", 16.0),
    "1m Weg": (1.3, "path", 12.0), "1m Wegfragment": (1.3, "path", 12.0),
}
WET = 0.15              # m over the lake level: a dead end stops where the ground comes down to this
LOW = 2.0               # m under the lake level: lower ground is dry land (the Tresa below Ponte Tresa), not lake
BRIDGE_RIM = 0.3       # m of a survey bridge polygon on each side outside the carriageway (the parapet)
APPROACH = (2.0, 12.0)  # m from a bridge's end: the stations of the road there that give its width (widen_bridges)
POLY_OVER = 2.0         # m a bridge measured in a survey bridge polygon may be wider than its narrower approach
SKIP = {"Tunnel", "Unterfuehrung", "Unterfuehrung mit Treppe", "Galerie"}
BRIDGE = {"Bruecke", "Bruecke mit Treppe", "Steg", "Bruecke mit Galerie"}
UNDER = "Unterfuehrung"     # kept where it passes under a railway and under no road (underpasses())
UNDER_CLEAR = 1.5           # m beyond the half width of a kept underpass: a line over it is a bridge there
UNDER_Z = 2.5               # m over the underpass: a line this high crosses over it (lower: it joins it)
RAIL_KINDS = ("Schmalspur", "Normalspur")


def load_lines():
    """swissTLM3D lines in the area: [(props, (n, 3) local x, y, z)], each cut to the area + CLIP."""
    d = json.load(open(os.path.join(DATA, "tlm", "TLM_STRASSE.json")))
    kept = {p.get("UUID") for p, _ in underpasses()[0]}
    keep = area.polygon().buffer(CLIP)
    shapely.prepare(keep)
    out = []
    for f in d["features"]:
        p = f["props"]
        if p.get("OBJEKTART") not in CLASSES or p.get("KUNSTBAUTE") in SKIP and p.get("UUID") not in kept:
            continue
        for part in f["parts"]:
            P = np.array(part, np.float64)
            if len(P) < 2:
                continue
            x, y = lv95_to_local(P[:, 0], P[:, 1])
            Q = np.column_stack([x, y, P[:, 2]])
            # cut to the area: runs of densified points inside it
            Qd = densify(Q, 1.0)
            ins = shapely.contains_xy(keep, Qd[:, 0], Qd[:, 1])
            if not ins.any():
                continue
            if ins.all():
                out += over_underpass(p, Q)
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, ins.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= 3:
                    out += over_underpass(p, Qd[a:b])
    return out


_UNDER = None


def _local(part):
    P = np.array(part, np.float64)
    x, y = lv95_to_local(P[:, 0], P[:, 1])
    return np.column_stack([x, y, P[:, 2]])


def underpasses():
    """v3.0.40 patch: the underpasses of swissTLM3D kept in the network: KUNSTBAUTE Unterfuehrung lines that cross a
    railway (TLM_EISENBAHN, not in a tunnel) and cross no road (a line of kind "road"; a path beside the tracks may
    pass over them). The underpass of Caslano station also runs under the Via Cantonale and stays out (a road over it
    would be cut). Returns ([(props, (n, 3) local x, y, z)], the area over them where a line higher by UNDER_Z m is a
    bridge (shapely, None without any), cKDTree and heights of their points every metre). Once per process."""
    global _UNDER
    if _UNDER is not None:
        return _UNDER
    _UNDER = ([], None, None, None)
    fr, fs = os.path.join(DATA, "tlm", "TLM_EISENBAHN.json"), os.path.join(DATA, "tlm", "TLM_STRASSE.json")
    if not (os.path.exists(fr) and os.path.exists(fs)):
        return _UNDER
    rails = [shapely.LineString(_local(part)[:, :2]) for f in json.load(open(fr))["features"]
             if f["props"].get("OBJEKTART") in RAIL_KINDS and f["props"].get("KUNSTBAUTE") != "Tunnel"
             for part in f["parts"] if len(part) > 1]
    feats = json.load(open(fs))["features"]
    roads = [shapely.LineString(_local(part)[:, :2]) for f in feats
             if CLASSES.get(f["props"].get("OBJEKTART"), (0, ""))[1] == "road" and f["props"].get("KUNSTBAUTE") not in SKIP
             for part in f["parts"] if len(part) > 1]
    zone = area.polygon().buffer(CLIP)
    rt, ot = shapely.STRtree(rails), shapely.STRtree(roads)
    kept = []
    for f in feats:
        p = f["props"]
        if p.get("KUNSTBAUTE") != UNDER or p.get("OBJEKTART") not in CLASSES:
            continue
        for part in f["parts"]:
            if len(part) < 2:
                continue
            Q = _local(part)
            g = shapely.LineString(Q[:, :2])
            if not g.intersects(zone):
                continue
            under_rail = any(g.crosses(rails[i]) for i in rt.query(g))
            under_road = any(g.crosses(roads[i]) for i in ot.query(g))
            if under_rail and not under_road:
                kept.append((p, Q))
    if not kept:
        return _UNDER
    over = shapely.union_all([shapely.LineString(Q[:, :2]).buffer(CLASSES[p["OBJEKTART"]][0] / 2 + UNDER_CLEAR)
                              for p, Q in kept])
    pts = np.concatenate([densify(Q, 1.0) for _, Q in kept])
    shapely.prepare(over)
    _UNDER = (kept, over, cKDTree(pts[:, :2]), pts[:, 2])
    print("underpasses under the railway kept: %s" % ", ".join("%s %s at %.0f, %.0f" % (
        p.get("OBJEKTART"), p.get("STRNAME") or "", Q[:, 0].mean(), Q[:, 1].mean()) for p, Q in kept), flush=True)
    return _UNDER


def over_underpass(props, Q):
    """[(props, Q)]: the line Q (n, 3) cut where it passes over a kept underpass (UNDER_Z m or more over it, in the area
    of underpasses()): those pieces are bridges (KUNSTBAUTE Bruecke), each piece shares its end points with the next.
    The track and the path beside it over Via Golf. Every other line comes back as it is."""
    kept, over, tree, zu = underpasses()
    if over is None or (props.get("KUNSTBAUTE") or "Keine") != "Keine":
        return [(props, Q)]
    x0, y0, x1, y1 = over.bounds
    if Q[:, 0].max() < x0 or Q[:, 0].min() > x1 or Q[:, 1].max() < y0 or Q[:, 1].min() > y1:
        return [(props, Q)]
    Qd = densify(Q, 0.5)
    ins = shapely.contains_xy(over, Qd[:, 0], Qd[:, 1])
    if ins.any():
        _, j = tree.query(Qd[ins, :2])
        ins[np.flatnonzero(ins)] = Qd[ins, 2] - zu[j] >= UNDER_Z
    e = np.flatnonzero(np.diff(np.r_[0, ins.astype(int), 0]))
    for a, b in zip(e[::2], e[1::2]):
        if (b - a) * 0.5 < 2.0:                  # a corner of the area, not a crossing
            ins[a:b] = False
    if not ins.any():
        return [(props, Q)]
    out = []
    e = np.flatnonzero(np.diff(ins.astype(int))) + 1
    cuts = np.r_[0, e, len(Qd) - 1]
    for a, b in zip(cuts[:-1], cuts[1:]):
        piece = Qd[a:b + 1]
        if len(piece) < 2:
            continue
        pr = dict(props, KUNSTBAUTE="Bruecke") if ins[a] else props
        out.append((pr, piece))
    return out


def densify(P, step):
    """Points along the polyline P (n, k) at most `step` apart (original vertices kept)."""
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / step)), 1)
        t = np.arange(1, n + 1)[:, None] / n
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def resample(P, step):
    """Stations every ~step m along P (n, 3): (m, 3) points and their arc length."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    n = max(int(np.round(d[-1] / step)), 1)
    s = np.linspace(0, d[-1], n + 1)
    Q = np.column_stack([np.interp(s, d, P[:, k]) for k in range(P.shape[1])])
    return Q, s


def nodes_of(lines):
    """Node id of both ends of every line (ends closer than SNAP are one node) and node positions."""
    ends = np.array([[L[0, :2], L[-1, :2]] for _, L in lines]).reshape(-1, 2)
    tree = cKDTree(ends)
    parent = np.arange(len(ends))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in tree.query_pairs(SNAP):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(len(ends))])
    _, nid = np.unique(roots, return_inverse=True)
    pos = np.zeros((nid.max() + 1, 2))
    cnt = np.zeros(nid.max() + 1)
    np.add.at(pos, nid, ends)
    np.add.at(cnt, nid, 1)
    return nid.reshape(-1, 2), pos / cnt[:, None]


def av_carriageway():
    """Union of the surveyed carriageways (MU strada_sentiero) and a tree of its parts."""
    av = pickle.load(open(os.path.join(WORK, "av_local.pkl"), "rb"))
    polys = [g for g, _ in av["LCSF"].get("strada_sentiero", [])]
    return polys


def av_bridges():
    """The surveyed bridges (SOSF ponte_passerella, the outline of the structure), v3.1."""
    av = pickle.load(open(os.path.join(WORK, "av_local.pkl"), "rb"))
    return [g for g, _ in av.get("SOSF", {}).get("ponte_passerella", []) if g.area > 1.0]


def bridge_overrides():
    """The corrections of dati/ponti.json by swissTLM3D id: "deck": "survey" (built on the survey's outline, its
    widths kept) and, v3.1, "width" (m, the carriageway on a bridge, set by hand)."""
    from config import PONTI as f
    if not os.path.exists(f):
        return {}
    return {b["tlm"]: b for b in json.load(open(f, encoding="utf-8"))["ponti"]}


def _approach(segs, at, sx, sy, sw, sgood, k, nd, e, bridges=True):
    """Width of the road at node nd (end e of segment k): the widest of the other roads there, each the median of
    its measured stations APPROACH m from the node; None if none is measured."""
    p = np.array([sx[k][e], sy[k][e]])
    ws = []
    for j in at.get(nd, []):
        if j == k or segs[j]["kind"] != "road" or (segs[j]["bridge"] and not bridges):
            continue
        d = np.hypot(sx[j] - p[0], sy[j] - p[1])
        m = (d >= APPROACH[0]) & (d <= APPROACH[1]) & sgood[j]
        if m.any():
            ws.append(float(np.median(sw[j][m])))
    return max(ws) if ws else None


def cap_bridge_polygons(segs, sx, sy, sw, spoly):
    """v3.1: a bridge measured in a survey bridge polygon (spoly: on half its stations or more) no wider than the
    narrower measured road at its ends + POLY_OVER m (sidewalks): a polygon can take in a bridgehead or a square
    beside the bridge. Returns the bridges narrowed."""
    at = {}
    for s in segs:
        for nd in s["nodes"]:
            at.setdefault(nd, []).append(s["id"])
    good = [np.ones(len(w), bool) for w in sw]
    out = 0
    for s in segs:
        k = s["id"]
        if not spoly[k]:
            continue
        ends = [w for nd, e in zip(s["nodes"], (0, -1))
                if (w := _approach(segs, at, sx, sy, sw, good, k, nd, e, bridges=False)) is not None]
        if ends:
            cap = max(min(ends) + POLY_OVER, CLASSES[s["class"]][0])
            if float(np.median(sw[k])) > cap + 0.05:
                sw[k] = np.minimum(sw[k], cap)
                out += 1
    return out


def widen_bridges(segs, sx, sy, sw, sgood, keep=()):
    """v3.1: a road bridge measured on less than half of its stations (no carriageway, no bridge polygon of the
    survey) as wide as the narrower of the roads at its two ends (one end if only one is measured): the median of
    their measured stations APPROACH m from the end. A bridge in pieces (lines that meet on it, none of them
    measured) is one: the roads at the ends of the chain, one width for all its pieces. Never narrower than it was.
    sw: widths per segment, changed in place; sgood: their measured masks. Returns the pieces widened."""
    at = {}
    for s in segs:
        for nd in s["nodes"]:
            at.setdefault(nd, []).append(s["id"])
    loose = {s["id"] for s in segs if s["bridge"] and s["kind"] == "road" and s["tlm"] not in keep
             and sgood[s["id"]].mean() < 0.5}
    out, seen = 0, set()
    for k0 in sorted(loose):
        if k0 in seen:
            continue
        chain, todo = {k0}, [k0]                        # the pieces of one bridge
        while todo:
            k = todo.pop()
            for nd in segs[k]["nodes"]:
                for j in at.get(nd, []):
                    if j in loose and j not in chain:
                        chain.add(j)
                        todo.append(j)
        seen |= chain
        ends = []
        for k in chain:
            for nd, e in zip(segs[k]["nodes"], (0, -1)):
                if any(j in chain for j in at.get(nd, []) if j != k):
                    continue                            # a node inside the chain
                w = _approach(segs, at, sx, sy, sw, sgood, k, nd, e)
                if w is not None:
                    ends.append(w)
        if not ends:
            continue
        w = min(ends)
        for k in chain:
            if w > float(np.median(sw[k])) + 0.05:
                sw[k] = np.maximum(sw[k], w)
                out += 1
    return out


def measure_width(P, N, polys, nominal):
    """Carriageway edges along the stations P (n, 2) with normals N from rays in the surveyed
    polygons: (width, offset of the middle along N) per station, NaN where the station is not
    inside a polygon or the edges are implausible."""
    tree = shapely.STRtree(polys)
    n = len(P)
    width, off = np.full(n, np.nan), np.full(n, np.nan)
    pts = shapely.points(P[:, 0], P[:, 1])
    hit = tree.query(pts, predicate="within")
    inside = np.zeros(n, bool)
    inside[hit[0]] = True
    if not inside.any():
        return width, off
    maxw = max(2.0 * nominal, 8.0)
    tt = np.arange(0.0, maxw, 0.1)
    union = shapely.union_all([polys[j] for j in np.unique(hit[1])])
    shapely.prepare(union)
    for sg in (1, -1):
        Q = P[inside, None, :] + sg * N[inside, None, :] * tt[None, :, None]
        ins = shapely.contains_xy(union, Q[..., 0].ravel(), Q[..., 1].ravel()).reshape(Q.shape[:2])
        first_out = np.where(ins.all(1), len(tt), np.argmin(ins, axis=1))
        dist = tt[np.clip(first_out, 0, len(tt) - 1)]
        dist[first_out >= len(tt)] = np.nan
        if sg == 1:
            dl = dist
        else:
            dr = dist
    w = dl + dr
    o = (dl - dr) / 2
    ok = (w > 0.5 * nominal) & (w < max(2.2 * nominal, 6.0)) & (np.abs(o) < MAX_SHIFT)
    width[np.flatnonzero(inside)[ok]] = w[ok]
    off[np.flatnonzero(inside)[ok]] = o[ok]
    return width, off


class LakeCut:
    """v3.0.40: g -> g without the lake: the dtm05 cells between LOW m under and WET m over the lake level (the lake is
    flat in swissALTI3D), for the surveyed paved polygons that reach into it (a slipway at Caslano). Only the patches of
    such cells that touch the surveyed lake (MU specchio_acqua) are cut: the floor of the Tresa valley below Ponte Tresa
    lies in the same band and its yards and roads were cut away (v3.1 test build). Identity when the lake level, dtm05
    or the survey's lake is not there."""

    def __init__(self, av=None):
        import water
        from geo import Grid
        self.cut = 0
        try:
            self.level, self.dtm = water.lake_level(), Grid.load(os.path.join(WORK, "dtm05.npz"))
            if av is None:
                av = pickle.load(open(os.path.join(WORK, "av_local.pkl"), "rb"))
        except (FileNotFoundError, OSError):
            self.dtm = None
            return
        lake = [g for g, _ in av["LCSF"].get("specchio_acqua", [])]
        self.lake = shapely.STRtree(lake) if lake else None
        if self.lake is None:
            self.dtm = None

    def wet(self, x, y):
        h = np.nan_to_num(self.dtm.sample(x, y), nan=1e4)
        return (h < self.level + WET) & (h >= self.level - LOW)

    def __call__(self, g):
        if self.dtm is None or g.is_empty:
            return g
        B = shapely.get_coordinates(shapely.segmentize(g.boundary, 2.0))
        if not self.wet(B[:, 0], B[:, 1]).any():
            return g
        from rasterio import features
        from rasterio.transform import Affine
        win, _, _ = self.dtm.window(*g.bounds, pad=2)
        a = np.asarray(win.a if win.a.ndim == 2 else win.a[0], np.float32)
        a = np.nan_to_num(a, nan=1e4)
        m = ((a < self.level + WET) & (a >= self.level - LOW)).astype(np.uint8)
        tr = Affine(win.res, 0, win.x_min, 0, -win.res, win.y_max)
        wet = [shapely.geometry.shape(q) for q, v in features.shapes(m, mask=m.astype(bool), transform=tr) if v]
        # the patches that touch the lake (within a cell and a half of it), not dry ground as low as the lake
        wet = [q for q in wet if len(self.lake.query(q, predicate="dwithin", distance=1.5 * win.res))]
        if not wet:
            return g
        self.cut += 1
        return g.difference(shapely.union_all(wet))


def dry_ends(lines, ends, node_pos):
    """v3.0.40: cut every dead end that runs down into the lake where the ground (dtm05) reaches the lake level + WET;
    its node moves to the cut. A line wholly in the water is dropped. Returns the lines and their ends."""
    import water
    from geo import Grid
    try:
        level, dtm = water.lake_level(), Grid.load(os.path.join(WORK, "dtm05.npz"))
    except (FileNotFoundError, OSError):
        return lines, ends
    deg = np.bincount(ends.ravel(), minlength=len(node_pos))
    out_l, out_e, cut, dropped = [], [], 0, 0
    for (props, L), (na, nb) in zip(lines, ends):
        # a pier or a footbridge over the water (KUNSTBAUTE Steg, Bruecke) stays as it is
        if (deg[na] == 1 or deg[nb] == 1) and (props.get("KUNSTBAUTE") or "Keine") not in BRIDGE:
            Q = densify(L, 1.0)
            h = np.nan_to_num(dtm.sample(Q[:, 0], Q[:, 1]), nan=1e4)
            dry = (h >= level + WET) | (h < level - LOW)          # not the valley of the Tresa below the lake
            if not dry.any():
                dropped += 1
                continue
            i0 = int(np.argmax(dry)) if deg[na] == 1 else 0
            i1 = len(Q) - int(np.argmax(dry[::-1])) if deg[nb] == 1 else len(Q)
            if i1 - i0 < 2:
                dropped += 1
                continue
            if i0 > 0 or i1 < len(Q):
                L = Q[i0:i1]
                if i0 > 0:
                    node_pos[na] = L[0, :2]
                if i1 < len(Q):
                    node_pos[nb] = L[-1, :2]
                cut += 1
        out_l.append((props, L))
        out_e.append((na, nb))
    print("dead ends into the lake (level %.2f m): %d cut at the shore, %d dropped" % (level, cut, dropped), flush=True)
    return out_l, np.array(out_e).reshape(-1, 2)


def build():
    lines = load_lines()
    ends, node_pos = nodes_of(lines)
    lines, ends = dry_ends(lines, ends, node_pos)
    polys = av_carriageway()
    bpolys = av_bridges()
    over = bridge_overrides()
    survey = {t for t, b in over.items() if b.get("deck") == "survey"}
    spoly = []
    segs = []
    sx, sy, sz, sseg, sidx, snode, sw, sss, sgood = [], [], [], [], [], [], [], [], []
    count = 0
    for k, ((props, L), (na, nb)) in enumerate(zip(lines, ends)):
        cls = props["OBJEKTART"]
        nominal, kind, lc = CLASSES[cls]
        L = L.copy()
        L[0, :2], L[-1, :2] = node_pos[na], node_pos[nb]
        Q, s = resample(L, STEP)
        if len(Q) < 2:
            continue
        T = np.gradient(Q[:, :2], axis=0)
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        N = np.column_stack([-T[:, 1], T[:, 0]])
        structure = props.get("KUNSTBAUTE") or "Keine"
        width, off = measure_width(Q[:, :2], N, polys, nominal) if kind == "road" else (np.full(len(Q), np.nan),) * 2
        on_bridge = kind == "road" and structure in BRIDGE and props.get("UUID") not in survey
        if on_bridge and bpolys and np.isnan(width).any():  # v3.1: the bridge polygons where there is no carriageway
            wb, ob = measure_width(Q[:, :2], N, bpolys, nominal)
            fill = np.isnan(width) & np.isfinite(wb) & (wb - 2 * BRIDGE_RIM > 0.5 * nominal)
            width[fill], off[fill] = wb[fill] - 2 * BRIDGE_RIM, ob[fill]
        else:
            fill = np.zeros(len(Q), bool)
        # smooth the measurements along the line; ends stay on the nodes
        good = ~np.isnan(width)
        if good.sum() >= 3:
            wf = np.where(good, width, np.nanmedian(width))
            of = np.where(good, off, 0.0)
            wf = median_filter(wf, 7, mode="nearest")
            of = median_filter(of, 7, mode="nearest")
            of[:3], of[-3:] = 0.0, 0.0
            Q[:, :2] += N * of[:, None]
            # v3.1: a bridge is one width along its stations (the nominal one where unmeasured narrowed the deck), never
            # under the nominal one (a bridge polygon narrower than its line's class made decks of 2.9 m)
            width = np.maximum(wf, nominal) if on_bridge else np.where(good, wf, nominal)
        else:
            width = np.full(len(Q), nominal)
        segs.append({"id": len(segs), "tlm": props.get("UUID"), "class": cls, "kind": kind, "lc": lc,
                     "surface": "hard" if props.get("BELAGSART") == "Hart" else "natural",
                     "structure": structure, "bridge": structure in BRIDGE, "stairs": "Treppe" in structure,
                     "level": int(props.get("STUFE") or 0), "name": props.get("STRNAME") or props.get("NAME") or "",
                     # roundabout, general traffic ban or not passable by car, owner (Kanton, Bund),
                     # through / connecting road, carriageways drawn apart
                     "roundabout": props.get("KREISEL") == "Wahr",
                     "ban": props.get("VERKEHRSBE") in ("Allgemeine Verkehrsbeschraenkung", "Gesperrt")
                            or props.get("BEFAHRBARK") == "Falsch",
                     "owner": props.get("EIGENTUEME") if props.get("EIGENTUEME") not in (None, "k_W") else "",
                     "importance": props.get("VERKEHRSBD") if props.get("VERKEHRSBD") not in (None, "k_W") else "",
                     "separated": props.get("RICHTUNGSG") == "Wahr",
                     "nodes": [int(na), int(nb)], "length": float(s[-1]), "n": len(Q), "first": count})
        count += len(Q)
        sx.append(Q[:, 0]); sy.append(Q[:, 1]); sz.append(Q[:, 2]); sss.append(s)
        sseg.append(np.full(len(Q), len(segs) - 1)); sidx.append(np.arange(len(Q)))
        nd = np.full(len(Q), -1)
        nd[0], nd[-1] = na, nb
        snode.append(nd); sw.append(width); sgood.append(good if good.sum() >= 3 else np.zeros(len(Q), bool))
        spoly.append(bool(fill.mean() >= 0.5) and good.sum() >= 3)
    capped = cap_bridge_polygons(segs, sx, sy, sw, spoly)
    wide = widen_bridges(segs, sx, sy, sw, sgood, keep=survey)
    hand = 0
    for s_ in segs:                                     # v3.1: widths set by hand (ponti.json "width")
        w = over.get(s_["tlm"], {}).get("width") if s_["bridge"] else None
        if w:
            sw[s_["id"]] = np.full(len(sw[s_["id"]]), float(w))
            hand += 1
    print("bridges as wide as their approaches: %d, narrowed to their approaches + %.0f m: %d, set by hand: %d"
          % (wide, POLY_OVER, capped, hand), flush=True)
    st = {k: np.concatenate(v) for k, v in (("x", sx), ("y", sy), ("z_tlm", sz), ("seg", sseg), ("idx", sidx),
                                              ("node", snode), ("width", sw), ("s", sss))}
    deg = np.bincount(np.array([s_["nodes"] for s_ in segs]).ravel(), minlength=len(node_pos))
    np.savez_compressed(os.path.join(WORK, "network.npz"), node_pos=node_pos, **st)
    json.dump(segs, open(os.path.join(WORK, "network.json"), "w"))
    km = {}
    for s_ in segs:
        km[s_["kind"]] = km.get(s_["kind"], 0) + s_["length"] / 1000
    print("network: %d segments, %d nodes (%d junctions, %d dead ends), %d stations; km %s; bridges %d" %
          (len(segs), len(node_pos), int((deg >= 3).sum()), int((deg == 1).sum()), len(st["x"]),
           {k: round(v, 1) for k, v in km.items()}, sum(s_["bridge"] for s_ in segs)))
    return segs, st, node_pos


def load():
    d = np.load(os.path.join(WORK, "network.npz"))
    segs = json.load(open(os.path.join(WORK, "network.json")))
    return segs, {k: d[k] for k in d.files if k != "node_pos"}, d["node_pos"]


if __name__ == "__main__":
    build()
