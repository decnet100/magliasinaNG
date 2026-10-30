"""Bridges of the road and path network (v2.0): decks over the gaps, reviewed one by one.

The DTM is the ground without bridges, so a road that follows it drops into every valley it
crosses. Every swissTLM3D line marked as a bridge (network.BRIDGE) outside the Strada Cantonale
corridor (which keeps its v1.1 surface) gets:
- a deck: the stations of the line at the heights of network_surface.py (no DTM data on a bridge:
  the smooth curve between the two approaches), flat across, as wide as the carriageway;
- a slab SLAB m thick under it, parapets on both sides (road bridges PARAPET m, footbridges RAIL
  m), piers down to the ground every PIER m where the deck is more than PIER_H m above it;
- where the deck is less than LOW m above the ground along the whole span (a culvert, a small
  stream under a road) the sides reach down to the ground instead.
The ground under a bridge is not filled: only where it rises above the underside of the slab is it
lowered to it.
v3.0: two road bridges side by side (the two carriageways of a road, a slip road beside the main one: swissTLM3D draws
a line each) with less than TWIN_GAP m between their decks are one bridge, as they are built: the gap between them is
closed by a deck of the same slab under a median MEDIAN_KERB m high, and no parapet stands along the side where the
other deck is. Up to the v3 test build they were two bridges with the river showing between them (the Via Cantonale
over the Magliasina at the Magliaso roundabout).
v3.1: survey decks. A deck drawn as a strip along the swissTLM3D line is only as wide as the line's carriageway: the
Via Cantonale over the Magliasina at the Magliaso roundabout is one deck about 25 m wide (the main road, the curve
from the west that joins it on the bridge, the sidewalk), and the strip left the river showing through it, parapets
in the middle of the road and the rest of the bridge at the height of the river bed. A bridge marked "deck": "survey"
in ponti.json is built on the outline of the cadastral survey's bridge polygons (SOSF ponte_passerella, drawn from
the aerial photos) that its line runs over; the lines over the same polygons are one deck. Its top is the plane of
the swissSURFACE3D surface model (the LiDAR returns of the deck) over the outline, eased at the edges where a road
joins it to the height of the ground there (swissALTI3D), with the side, the slab and the parapets along the edges
over a drop; the roads of the network and of the corridor stay out of the outline.
beamng/dati/ponti.json lists every bridge (swissTLM3D id, place, length, height above the ground,
the check flags) and holds the manual corrections, read back at every build:
  "z0" / "z1": deck height at the first / last end (m), "profile": "straight" (a straight deck
  between the ends) or "tlm" (the 3D line of swissTLM3D, met to the network at the ends: a footbridge
  over a road with its stairs), "type": "open" | "culvert", "skip": true (no deck: the road follows
  the ground), "deck": "survey" (v3.1: the deck on the survey's outline, above). bridge_report.py draws the
  profile and a view of every bridge for the review.
No pier stands on a road or path of the network (the one under the bridge).
"""
import json, os
import numpy as np
import shapely
from config import WORK
import network

HERE = os.path.dirname(os.path.abspath(__file__))
from config import PONTI  # noqa: E402  (dati/ponti.json, or MAGLIASO_PONTI on an external site)
SLAB = 0.6
PARAPET = 1.0
RAIL = 1.0
PARAPET_W = 0.3
PIER = 25.0
PIER_H = 4.0
PIER_W = 1.2
LOW = 1.5
EXTRA_W = 0.3          # m of deck beyond the carriageway on each side
PARAPET_DROP = 0.6     # m of drop beside the deck edge that needs a parapet
PARAPET_RUN = 2.0      # m the parapet goes on beyond it
TWIN_GAP = 10.0        # m between two road decks side by side that are one bridge
TWIN_ANGLE = 25.0      # degrees between their directions, at most
MEDIAN_KERB = 0.12     # m of the median between them over the decks
SURVEY_ON = 3.0        # m of a line over a survey bridge polygon for the polygon to be its deck
SURVEY_IN = 1.0        # m inside the outline: the surface model cells of the deck plane
SURVEY_RMS = 0.35      # m: a plane further from the cells than this is not a deck (the lines' heights then)
SURVEY_OFF = 1.5       # m: nor a plane further than this from the heights of the lines on it
SURVEY_JOIN = 0.5      # m: an edge with the ground within this of the deck 1-3 m out is a road joining the deck
SURVEY_FADE = 6.0      # m: how far into the deck the joins pull its height
SURVEY_CELL = 1.0      # m, the triangles of the top
SURVEY_HULL = 1.12     # the hull of a deck's outline if it is at most this much larger
SURVEY_ROAD = 0.5      # m: a road surface this close to the deck just outside its edge is a road joining the deck (one
                       # further off is a path under or beside it: the edge there is no join, also with the ground)
SURVEY_ROAD_OUT = (0.25, 0.5, 1.0)   # m outside the edge where the joining road surface is read (the first found)
_survey = {}


def load_overrides():
    if not os.path.exists(PONTI):
        return {}
    return {b["tlm"]: b for b in json.load(open(PONTI, encoding="utf-8"))["ponti"]}


def half_width(st, s):
    a, n = s["first"], s["n"]
    return 0.5 * float(np.median(st["width"][a:a + n])) + EXTRA_W


def survey_decks(segs=None, x=None, y=None, w=None):
    """v3.1: the decks on the survey's outline, [(footprint, [line ids])], of the bridges marked "deck": "survey" in
    ponti.json: the survey bridge polygons each line runs over for SURVEY_ON m or more and the polygons beside them,
    the lines whose polygons touch one deck; with the strip of each line where it goes on past the polygons (onto
    the abutment, where the terrain model slopes down under the bridge's end). Computed once per build (the
    network and the corridor cut them out before the bridges are built)."""
    if "decks" in _survey:
        return _survey["decks"]
    if segs is None:
        segs, st, _ = network.load()
        x, y, w = st["x"], st["y"], st["width"]
    over = load_overrides()
    want = [s for s in segs if s["bridge"] and over.get(s["tlm"], {}).get("deck") == "survey"]
    groups = []
    if want:
        import pickle
        av = pickle.load(open(os.path.join(WORK, "av_local.pkl"), "rb"))
        polys = [g for g, _ in av["SOSF"].get("ponte_passerella", []) if g.area > 1.0]
        tree = shapely.STRtree(polys)
        for s in want:
            a, n = s["first"], s["n"]
            L = shapely.LineString(np.column_stack([x[a:a + n], y[a:a + n]]))
            took = {int(i) for i in tree.query(L) if polys[i].intersection(L).length >= SURVEY_ON}
            while took:                                # and the polygons beside them (a bridge drawn in parts)
                more = {int(i) for i in tree.query(shapely.union_all([polys[k] for k in took]).buffer(0.2),
                                                   predicate="intersects")} - took
                if not more:
                    break
                took |= more
            took = [polys[i] for i in sorted(took)]
            if not took:
                print("survey deck: no survey bridge under %s, a strip deck" % s["tlm"], flush=True)
                continue
            strip = L.buffer(half_width({"width": w}, s), cap_style="flat")
            groups.append([shapely.union_all(took + [strip]), {s["id"]}])
        merged = True
        while merged:                                  # the lines over one bridge: one deck
            merged = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i][0].buffer(0.2).intersects(groups[j][0]):
                        groups[i] = [shapely.union_all([groups[i][0], groups[j][0]]), groups[i][1] | groups[j][1]]
                        del groups[j]
                        merged = True
                        break
                if merged:
                    break
    from shapely.geometry.polygon import orient
    out = []
    for g, ids in groups:
        # the survey polygons of one bridge side by side, the strips past them: no sliver or notch between them (the
        # hull where it adds little: the corners between a strip and the polygon's edge are road)
        g = g.buffer(0.05, join_style="mitre").buffer(-0.05, join_style="mitre")
        g = max(shapely.get_parts(g), key=lambda p: p.area)
        if g.convex_hull.area <= SURVEY_HULL * g.area:
            g = g.convex_hull
        out.append((orient(g), sorted(ids)))
    _survey["decks"] = out
    return out


def survey_height(foot, ground, P_line, z_line, dsm=None, surface=None):
    """Height (x, y) -> z of the top of a survey deck: the plane of the surface model (swissSURFACE3D, the deck's
    LiDAR returns) over the outline SURVEY_IN m inside it, fitted without the cells off it (vehicles, a lamp post);
    the heights of the lines on it where there is no such plane. Where a road joins the deck the height goes to the
    road's over SURVEY_FADE m: v3.1, the road surface just outside the edge (surface(x, y, z near): the meshes of the
    corridor and of the network, NaN where there is none) within SURVEY_ROAD m of the deck; elsewhere the ground 1-3 m
    out within SURVEY_JOIN m of it. Returns (fn, info)."""
    from scipy.spatial import cKDTree
    info = {}
    base = None
    if dsm is None and any(os.path.exists(os.path.join(WORK, f)) for f in ("dsm05.npz", "dsm05.npy")):
        from geo import Grid                           # a big grid is stored as dsm05.npy + .json (Grid.save)
        dsm = Grid.load(os.path.join(WORK, "dsm05.npz"))
    if dsm is not None:
        x0, y0, x1, y1 = foot.bounds
        win, _, _ = dsm.window(x0, y0, x1, y1, pad=2)
        h, w = win.a.shape
        cx = win.x_min + (np.arange(w) + 0.5) * win.res
        cy = win.y_max - (np.arange(h) + 0.5) * win.res
        X, Y = np.meshgrid(cx, cy)
        inner = foot.buffer(-SURVEY_IN)
        m = shapely.contains_xy(inner, X, Y) & np.isfinite(win.a)
        if m.sum() >= 12:
            c = np.array(foot.centroid.coords[0])
            A = np.column_stack([X[m] - c[0], Y[m] - c[1], np.ones(m.sum())])
            Z = np.asarray(win.a, np.float64)[m]
            keep = np.ones(len(Z), bool)
            for _ in range(4):
                co = np.linalg.lstsq(A[keep], Z[keep], rcond=None)[0]
                r = Z - A @ co
                mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
                keep = np.abs(r) <= max(3.0 * mad, 0.15)
            rms = float(np.sqrt(np.mean(r[keep] ** 2)))
            zl = float(np.median(z_line - (co[2] + (P_line - c) @ co[:2])))
            info.update(plane_rms=round(rms, 3), plane_vs_lines=round(zl, 2), cells=int(keep.sum()),
                        cross_grade=round(float(np.hypot(*co[:2])), 3))
            if rms <= SURVEY_RMS and abs(zl) <= SURVEY_OFF:
                base = lambda x, y, co=co, c=c: co[2] + (np.asarray(x) - c[0]) * co[0] + (np.asarray(y) - c[1]) * co[1]
    if base is None:                                   # the lines' heights, as the deck between twin decks
        kt = cKDTree(P_line)

        def base(x, y):
            q = np.column_stack([np.ravel(x), np.ravel(y)])
            d, k = kt.query(q, k=min(4, len(P_line)))
            d, k = d.reshape(len(q), -1), k.reshape(len(q), -1)
            wt = 1.0 / np.maximum(d, 0.1)
            return ((z_line[k] * wt).sum(1) / wt.sum(1)).reshape(np.shape(x))
        info["plane"] = "none: the heights of the lines"
    # the joins along the outline
    ring = foot.exterior
    s = np.arange(0.0, ring.length, 1.0)
    E = np.array([ring.interpolate(v).coords[0] for v in s])
    E2 = np.array([ring.interpolate(v + 0.05).coords[0] for v in s])
    T = E2 - E
    T /= np.maximum(np.linalg.norm(T, axis=1)[:, None], 1e-9)
    Nout = np.column_stack([T[:, 1], -T[:, 0]])          # exterior counter-clockwise: right of the way is out
    # the ground 1 to 3 m out, the highest (the terrain model slopes down for a metre or two at a bridge's end)
    g = np.max([np.asarray(ground((E + Nout * k)[:, 0], (E + Nout * k)[:, 1]), np.float64) for k in (1.0, 2.0, 3.0)], 0)
    zb = base(E[:, 0], E[:, 1])
    d = g - zb
    join = np.abs(d) <= SURVEY_JOIN
    # v3.1: where a road surface (the corridor's, the network's) meets the edge, the deck goes to its height there:
    # pulled to the bare ground the deck stood up to 0.45 m over or under the road that joins it (a step at the
    # Magliaso bridge over the Magliasina, at its corners)
    on_road = np.zeros(len(E), bool)
    if surface is not None:
        zr = np.full(len(E), np.nan)
        other = np.zeros(len(E), bool)
        for k in SURVEY_ROAD_OUT:
            Q = E + Nout * k
            h = np.asarray(surface(Q[:, 0], Q[:, 1], zb), np.float64)
            new = ~np.isfinite(zr) & np.isfinite(h) & (np.abs(h - zb) <= SURVEY_ROAD)
            zr[new] = h[new]
            other |= np.isfinite(h)
        on_road = np.isfinite(zr)
        d = np.where(on_road, zr - zb, d)
        join = on_road | (join & ~other)                # a road surface off the deck's height: a drop, no join
        info["road_joins_m"] = int(on_road.sum())
    info["joins_m"] = int(join.sum())
    info["join_step_max"] = round(float(np.abs(d[join]).max()), 2) if join.any() else 0.0
    Ej, dj = E[join], d[join]
    tj = cKDTree(Ej) if len(Ej) else None

    def fn(x, y):
        z = base(x, y)
        if tj is None:
            return z
        q = np.column_stack([np.ravel(x), np.ravel(y)])
        dd, k = tj.query(q, k=min(8, len(Ej)))
        dd, k = dd.reshape(len(q), -1), k.reshape(len(q), -1)
        # v3.1: the edge itself on the joins' heights (a floor of 0.5 m left it at a mean of the joins 1-2 m around)
        wt = 1.0 / np.maximum(dd, 0.05) ** 2
        corr = (dj[k] * wt).sum(1) / (wt.sum(1) + 1.0 / SURVEY_FADE ** 2)
        return z + corr.reshape(np.shape(z))
    fn.edge = (E, Nout, join)
    return fn, info


def survey_geometry(foot, zfn, ground, kind):
    """Triangles (k, 3, 3) of a survey deck: (top, sides and underside, parapets), facing outwards. The parapets
    stand on the edges over a drop of PARAPET_DROP m (not where a road joins), PARAPET_W m wide inside the edge."""
    import road_mesh
    V, T = road_mesh.mesh_polygon(foot, zfn, cell=SURVEY_CELL)
    top = V[T]
    n = np.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0])
    top[n[:, 2] < 0] = top[n[:, 2] < 0][:, ::-1]
    under = top[:, ::-1].copy()
    under[:, :, 2] -= SLAB
    sides, par = [under], []
    up = np.array([0, 0, 1.0])
    ph = PARAPET if kind == "road" else RAIL
    rise = np.array([0, 0, ph])
    for ring in [foot.exterior] + list(foot.interiors):
        # points every <= 1 m along the ring, the outward normal of every piece (orient(): the outline runs
        # counter-clockwise, the holes clockwise, so the outside of the deck is on the right of both)
        P = np.array(shapely.segmentize(shapely.LineString(np.array(ring.coords)), 1.0).coords)
        z = zfn(P[:, 0], P[:, 1])
        D = np.diff(P, axis=0)
        Lg = np.maximum(np.linalg.norm(D, axis=1), 1e-9)
        Tn = D / Lg[:, None]
        Nn = np.column_stack([Tn[:, 1], -Tn[:, 0]])
        for i in range(len(D)):
            sides.append(_quad(np.r_[P[i], z[i]], np.r_[P[i + 1], z[i + 1]], np.r_[P[i + 1], z[i + 1] - SLAB],
                               np.r_[P[i], z[i] - SLAB], np.r_[Nn[i], 0.0]))
        M = 0.5 * (P[:-1] + P[1:])
        g = np.max([np.asarray(ground((M + Nn * k)[:, 0], (M + Nn * k)[:, 1]), np.float64) for k in (1.0, 2.0, 3.0)], 0)
        drop = (0.5 * (z[:-1] + z[1:]) - g) > PARAPET_DROP
        drop &= Lg > 1e-6
        # runs of pieces over a drop (a ring is closed: a run may go on past its start)
        idx = np.flatnonzero(np.diff(np.r_[0, drop.astype(int), 0]))
        runs = list(zip(idx[0::2], idx[1::2]))
        if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == len(drop):
            runs = [(runs[-1][0], runs[0][1] + len(drop))] + runs[1:-1]
        for a, b in runs:
            seq = [i % len(drop) for i in range(a, b)]
            if sum(Lg[i] for i in seq) < 1.5:
                continue
            for i in seq:
                o0, o1 = np.r_[P[i], z[i]], np.r_[P[i + 1], z[i + 1]]
                i0 = np.r_[P[i] - Nn[i] * PARAPET_W, z[i]]
                i1 = np.r_[P[i + 1] - Nn[i] * PARAPET_W, z[i + 1]]
                par.append(_quad(o0, o1, o1 + rise, o0 + rise, np.r_[Nn[i], 0.0]))
                par.append(_quad(i0, i1, i1 + rise, i0 + rise, np.r_[-Nn[i], 0.0]))
                par.append(_quad(o0 + rise, o1 + rise, i1 + rise, i0 + rise, up))
            for i, k, s_ in ((seq[0], seq[0], -1.0), (seq[-1], seq[-1] + 1, 1.0)):
                o = np.r_[P[k], z[k]]
                ii = np.r_[P[k] - Nn[i] * PARAPET_W, z[k]]
                par.append(_quad(o, ii, ii + rise, o + rise, np.r_[Tn[i] * s_, 0.0]))
    cat = lambda L_: np.concatenate(L_) if L_ else np.zeros((0, 3, 3))
    return top, cat(sides), cat(par)


def _quad(a, b, c, d, out):
    """Two triangles of the quad a-b-c-d facing `out`."""
    t = np.array([a, b, c, a, c, d], np.float64).reshape(2, 3, 3)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (nrm @ np.asarray(out, np.float64)) < 0
    t[flip] = t[flip][:, ::-1]
    return t


def deck_geometry(P, N, z, hw, ground, kind, typ, avoid=None, no_parapet=None):
    """Triangles (k, 3, 3) of a deck along stations P (n, 2) with normals N and deck heights z:
    (top, sides and underside, parapets, piers), each facing outwards. avoid: the drivable surfaces
    under the bridge (no pier on them). no_parapet: (left, right) masks of the stations where that side
    of the deck joins a twin deck (no parapet there)."""
    n = len(P)
    L = P + N * hw
    R = P - N * hw
    up, down = np.array([0, 0, 1.0]), np.array([0, 0, -1.0])
    top = [_quad(np.r_[L[i], z[i]], np.r_[R[i], z[i]], np.r_[R[i + 1], z[i + 1]], np.r_[L[i + 1], z[i + 1]], up)
           for i in range(n - 1)]
    gL, gR = ground(L[:, 0], L[:, 1]), ground(R[:, 0], R[:, 1])
    if typ == "culvert":                               # sides down to the ground
        botL, botR = np.minimum(gL, z - SLAB), np.minimum(gR, z - SLAB)
    else:
        botL, botR = z - SLAB, z - SLAB
    sides = []
    for E, bot, sign in ((L, botL, 1.0), (R, botR, -1.0)):
        for i in range(n - 1):
            out = np.r_[N[i] * sign, 0.0]
            sides.append(_quad(np.r_[E[i], z[i]], np.r_[E[i + 1], z[i + 1]], np.r_[E[i + 1], bot[i + 1]],
                               np.r_[E[i], bot[i]], out))
    for i in range(n - 1):
        sides.append(_quad(np.r_[L[i], botL[i]], np.r_[R[i], botR[i]], np.r_[R[i + 1], botR[i + 1]],
                           np.r_[L[i + 1], botL[i + 1]], down))
    ph = PARAPET if kind == "road" else RAIL
    rise = np.array([0, 0, ph])
    par = []
    T = np.column_stack([N[:, 1], -N[:, 0]])
    sst = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    for E, sign, gE in ((L, 1.0, gL), (R, -1.0, gR)):
        inner = E - N * sign * PARAPET_W
        # v3.0: the parapet where there is a drop beside the deck (PARAPET_DROP m), carried PARAPET_RUN m on: where
        # the deck lies on the ground at its ends, the approach road goes on wider than the deck, and a parapet
        # there stood as a block on the road
        need = (z - gE) > PARAPET_DROP
        if need.any():
            idx = np.flatnonzero(need)
            near = np.abs(sst[:, None] - sst[idx][None, :]).min(1) <= PARAPET_RUN
            need = near
        if no_parapet is not None:                     # v3.0: the side joined to a twin deck
            need = need & ~no_parapet[0 if sign > 0 else 1]
        for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, need.astype(int), 0]))[k::2] for k in (0, 1)]):
            if b - a < 2:
                continue
            for i in range(a, b - 1):
                o0, o1 = np.r_[E[i], z[i]], np.r_[E[i + 1], z[i + 1]]
                i0, i1 = np.r_[inner[i], z[i]], np.r_[inner[i + 1], z[i + 1]]
                par.append(_quad(o0, o1, o1 + rise, o0 + rise, np.r_[N[i] * sign, 0.0]))
                par.append(_quad(i0, i1, i1 + rise, i0 + rise, np.r_[-N[i] * sign, 0.0]))
                par.append(_quad(o0 + rise, o1 + rise, i1 + rise, i0 + rise, up))
            for i, s_ in ((a, -1.0), (b - 1, 1.0)):
                o, ii = np.r_[E[i], z[i]], np.r_[inner[i], z[i]]
                par.append(_quad(o, ii, ii + rise, o + rise, np.r_[T[i] * s_, 0.0]))
    piers = []
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if typ != "culvert" and s[-1] > PIER:
        k = int(s[-1] // PIER)
        for j in range(1, k + 1):
            sj = s[-1] * j / (k + 1)
            i = int(np.clip(np.searchsorted(s, sj), 1, n - 1))
            f = (sj - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
            c = P[i - 1] + (P[i] - P[i - 1]) * f
            zc = z[i - 1] + (z[i] - z[i - 1]) * f - SLAB
            g = float(ground([c[0]], [c[1]])[0])
            if zc - g < PIER_H:
                continue
            Nn, Tt = N[i], T[i]
            hw_p, ht = hw * 0.8, PIER_W / 2
            if avoid is not None and avoid.distance(shapely.Point(c)) < hw_p + ht + 0.3:
                continue
            cs = [c + Nn * hw_p + Tt * ht, c - Nn * hw_p + Tt * ht, c - Nn * hw_p - Tt * ht, c + Nn * hw_p - Tt * ht]
            for q in range(4):
                a_, b_ = cs[q], cs[(q + 1) % 4]
                mid = 0.5 * (a_ + b_) - c
                piers.append(_quad(np.r_[a_, g - 0.5], np.r_[b_, g - 0.5], np.r_[b_, zc], np.r_[a_, zc], np.r_[mid, 0.0]))
    cat = lambda L_: np.concatenate(L_) if L_ else np.zeros((0, 3, 3))
    return cat(top), cat(sides), cat(par), cat(piers)


def ends_to_ground(P, N, z, s, degree, ground, max_ext=60.0):
    """A bridge end where the network ends (the area cuts the line; a footbridge on its own):
    a deck end in the air carries on straight until the ground comes within 1 m of it, at most
    max_ext m, easing down onto it; a deck end below the ground rises to it (the deck is tilted
    towards that end). Returns P, N, z and the notes of what was done."""
    notes = []
    for end in (0, 1):
        if degree.get(s["nodes"][end], 0) != 1 or len(P) < 2:
            continue
        i, j = (0, 1) if end == 0 else (len(P) - 1, len(P) - 2)
        g = float(ground([P[i][0]], [P[i][1]])[0])
        if z[i] - g > 1.5:
            u = (P[i] - P[j]) / max(np.linalg.norm(P[i] - P[j]), 1e-9)
            add, reached, gq = [], False, g
            for k in np.arange(2.0, max_ext + 0.1, 2.0):
                q = P[i] + u * k
                add.append(q)
                gq = float(ground([q[0]], [q[1]])[0])
                if gq >= z[i] - 1.0:
                    reached = True
                    break
            if not reached:
                notes.append("suspended")
                continue
            A = np.array(add)
            ze = np.linspace(z[i], min(z[i], gq + 0.05), len(A) + 1)[1:]    # easing onto the ground
            if end == 0:
                P = np.vstack([A[::-1], P]); N = np.vstack([np.repeat(N[:1], len(A), 0), N])
                z = np.r_[ze[::-1], z]
            else:
                P = np.vstack([P, A]); N = np.vstack([N, np.repeat(N[-1:], len(A), 0)])
                z = np.r_[z, ze]
            notes.append("extended %d m" % round(2.0 * len(A)))
        elif z[i] - g < -0.5:
            ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            w = ss / max(ss[-1], 1e-9) if end == 1 else 1.0 - ss / max(ss[-1], 1e-9)
            z = z + (g - z[i]) * w
            notes.append("joined to terrain")
    return P, N, z, notes


def build(net, ground, on_mesh, on_carve_min, xs, ys, exclude=None, surface=None):
    """Deck meshes of every bridge (on_mesh(x, y, material, uv tile, soup, kind)), the ground
    under them lowered where it rises above the slab (on_carve_min(rows, cols, z)), the record of
    every bridge in ponti.json. `exclude`: polygon of the cantonale corridor (its bridges stay).
    `surface(x, y, z)`: the height of the road meshes already built nearest to z (NaN: none), v3.1."""
    import shapely
    from rasterio import features
    from rasterio.transform import Affine
    segs = net.segs
    st_x, st_y = net.x, net.y
    over = load_overrides()
    records = []
    sq = xs[1] - xs[0]
    degree = {}
    for q in segs:
        for nd in q["nodes"]:
            degree[nd] = degree.get(nd, 0) + 1
    sdecks = survey_decks(segs, st_x, st_y, net.w)   # v3.1: decks on the survey's outline
    on_survey = {i: k for k, (_, ids) in enumerate(sdecks) for i in ids}

    def carve(foot, zf):
        """The ground under a deck no higher than the underside of its slab (zf: its top)."""
        x0, y0, x1, y1 = foot.bounds
        c0, c1 = max(int(np.searchsorted(xs, x0)) - 1, 0), min(int(np.searchsorted(xs, x1)) + 1, len(xs))
        r0, r1 = max(int(np.searchsorted(ys, y0)) - 1, 0), min(int(np.searchsorted(ys, y1)) + 1, len(ys))
        if c1 > c0 and r1 > r0:
            tr = Affine(sq, 0, xs[c0] - 0.5 * sq, 0, sq, ys[r0] - 0.5 * sq)
            m = features.rasterize([(foot, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, dtype=np.uint8,
                                   all_touched=True).astype(bool)
            rr, cc = np.nonzero(m)
            if len(rr):
                on_carve_min(r0 + rr, c0 + cc, zf(xs[c0 + cc], ys[r0 + rr]) - SLAB - 0.15)
    todo = []
    for s in segs:
        if not s["bridge"]:
            continue
        a, n = s["first"], s["n"]
        P = np.column_stack([st_x[a:a + n], st_y[a:a + n]])
        if exclude is not None and s["id"] not in on_survey and shapely.LineString(P).within(exclude.buffer(1.0)):
            continue
        N = np.column_stack([net.nx[a:a + n], net.ny[a:a + n]])
        z = net.z[a:a + n].copy()
        o = over.get(s["tlm"], {})
        if o.get("skip"):
            continue
        if o.get("profile") == "straight" or "z0" in o or "z1" in o:
            z0 = float(o.get("z0", z[0])); z1 = float(o.get("z1", z[-1]))
            ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            z = z0 + (z1 - z0) * ss / max(ss[-1], 1e-9)
        hw = half_width({"width": net.w}, s)
        z_line = z.copy()                              # on the stations (checks)
        notes = []
        if not ("z0" in o or "z1" in o or o.get("profile")):
            P, N, z, notes = ends_to_ground(P, N, z, s, degree, ground)
        todo.append(dict(s=s, a=a, n=n, P=P, N=N, z=z, o=o, hw=hw, z_line=z_line, notes=notes,
                         foot=shapely.LineString(P).buffer(hw, cap_style="flat"), survey=on_survey.get(s["id"])))
    plain = [i for i, b in enumerate(todo) if b["survey"] is None]      # the survey decks have no twins
    twins = [(plain[i], plain[j]) for i, j in twin_pairs([todo[i] for i in plain])]
    for b in todo:
        s, a, n, P, N, z, o, hw, z_line, notes = (b[k] for k in ("s", "a", "n", "P", "N", "z", "o", "hw", "z_line", "notes"))
        g = ground(P[:, 0], P[:, 1])
        clear = z - SLAB - g
        typ = o.get("type") or ("culvert" if clear.max() < LOW else "open")
        avoid = None
        if getattr(net, "ptree", None) is not None:
            under = net.ptree.query(shapely.LineString(P).buffer(hw + 3.0), predicate="intersects")
            if len(under):
                avoid = shapely.union_all([net.polys[i]["geom"] for i in under])
        mat_top = {("road", "hard"): "mp_road_asphalt", ("road", "natural"): "mp_road_gravel",
                   ("path", "hard"): "mp_path_paved", ("path", "natural"): "mp_path_dirt"}[(s["kind"], s["surface"])]
        b["mat_top"] = mat_top
        cxy = P.mean(0)
        if twins:
            notes = notes + [f"twin of {t}" for t in b.get("twin_of", [])]
        if b["survey"] is not None:                    # built below, one deck for all its lines
            notes = notes + ["survey deck"]
            top = None
        else:
            top, sides, par, piers = deck_geometry(P, N, z, hw, ground, s["kind"], typ, avoid, b.get("no_parapet"))
        if top is not None:
            on_mesh(cxy[0], cxy[1], mat_top, 6.0 if mat_top in ("mp_road_asphalt", "mp_path_paved", "mp_sidewalk", "mp_hard_asphalt") else 1.25, top, "top")   # v3.0: the asphalt over 6 m
            on_mesh(cxy[0], cxy[1], "mp_road_wall", 1.6, sides, "side")
            if len(par):
                on_mesh(cxy[0], cxy[1], "mp_bridge_parapet", 1.6, par, "side")
            if len(piers):
                on_mesh(cxy[0], cxy[1], "mp_road_wall", 1.6, piers, "side")
            net.deck_tops.append(top)
            foot = shapely.LineString(P).buffer(hw, cap_style="flat")
            if hasattr(net, "deck_feet"):
                net.deck_feet.append((s["kind"], foot))
            from scipy.spatial import cKDTree
            kt = cKDTree(P)
            carve(foot, lambda x, y, kt=kt, z=z: z[kt.query(np.column_stack([x, y]))[1]])
        # checks for the review: deck against the 3D line of swissTLM3D (on a bridge its height is the
        # deck's), steepest grade, grade change where the deck meets the approaches
        ztlm = net.z_tlm[a:a + n] if getattr(net, "z_tlm", None) is not None else np.full(n, np.nan)
        Pl = np.column_stack([st_x[a:a + n], st_y[a:a + n]])
        ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pl, axis=0), axis=1))]
        z = z_line if len(z_line) == n else z[:n]
        grade = np.gradient(z, ss) if n > 1 and ss[-1] > 0 else np.zeros(n)
        g_in = net.g[a - 1] if a > 0 and net.seg[a - 1] != s["id"] else np.nan
        flags = []
        dtlm = float(np.nanmax(np.abs(z - ztlm))) if np.isfinite(ztlm).any() else np.nan
        if np.isfinite(dtlm) and dtlm > 1.0:
            flags.append("tlm")
        if np.abs(grade).max() > (0.15 if s["kind"] == "road" else 0.3):
            flags.append("steep")
        if clear.min() < -0.5 and typ == "open":
            flags.append("terrain")
        flags += notes
        records.append({"tlm": s["tlm"], "seg": s["id"], "class": s["class"], "name": s["name"],
                        "dz_tlm_m": round(dtlm, 2) if np.isfinite(dtlm) else None,
                        "max_grade": round(float(np.abs(grade).max()), 3), "flags": flags,
                        "x": round(float(cxy[0]), 1), "y": round(float(cxy[1]), 1),
                        "length_m": round(s["length"], 1), "width_m": round(2 * hw, 1),
                        "z_ends": [round(float(z[0]), 2), round(float(z[-1]), 2)],
                        "max_height_m": round(float(clear.max() + SLAB), 1), "auto_type": typ,
                        "override": {k: v for k, v in o.items() if k in ("z0", "z1", "profile", "type", "skip", "deck", "width")}})
    for i, j in twins:                                 # v3.0: the deck between two twin decks
        med = median_geometry(todo[i], todo[j])
        if med is None:
            continue
        top, sides, foot, zf = med
        cxy = np.array(foot.centroid.coords[0])
        on_mesh(cxy[0], cxy[1], "mp_sidewalk", 6.0, top, "top")
        on_mesh(cxy[0], cxy[1], "mp_road_wall", 1.6, sides, "side")
        net.deck_tops.append(top)
        if hasattr(net, "deck_feet"):
            net.deck_feet.append(("road", foot))
        carve(foot, zf)
    for k, (foot, ids) in enumerate(sdecks):           # v3.1: the decks on the survey's outline
        mine = [b for b in todo if b["survey"] == k]
        if not mine:
            continue
        P_l = np.vstack([b["P"] for b in mine])
        z_l = np.concatenate([b["z"] for b in mine])
        zf, info = survey_height(foot, ground, P_l, z_l, surface=surface)
        kind = "road" if any(b["s"]["kind"] == "road" for b in mine) else "path"
        top, sides, par = survey_geometry(foot, zf, ground, kind)
        mat_top = next((b["mat_top"] for b in mine if b["s"]["kind"] == kind), mine[0]["mat_top"])
        cxy = np.array(foot.centroid.coords[0])
        on_mesh(cxy[0], cxy[1], mat_top, 6.0 if mat_top in ("mp_road_asphalt", "mp_path_paved") else 1.25, top, "top")
        on_mesh(cxy[0], cxy[1], "mp_road_wall", 1.6, sides, "side")
        if len(par):
            on_mesh(cxy[0], cxy[1], "mp_bridge_parapet", 1.6, par, "side")
        net.deck_tops.append(top)
        if hasattr(net, "deck_feet"):
            net.deck_feet.append((kind, foot))
        carve(foot, zf)
        info.update(area_m2=round(foot.area, 1), parapet_tris=len(par))
        print("survey deck", [b["s"]["tlm"] for b in mine], info, flush=True)
        for r in records:
            if r["seg"] in ids:
                r["survey"] = info
    return records


def twin_pairs(todo):
    """Pairs (i, j) of road decks side by side (less than TWIN_GAP m apart, directions within TWIN_ANGLE degrees,
    alongside each other over most of the shorter one); marks on every deck the stations of the side that faces
    its twin (no_parapet) and the twin's name (twin_of)."""
    pairs = []
    for i, bi in enumerate(todo):
        for j in range(i + 1, len(todo)):
            bj = todo[j]
            if bi["s"]["kind"] != "road" or bj["s"]["kind"] != "road":
                continue
            if bi["foot"].distance(bj["foot"]) > TWIN_GAP:
                continue
            di, dj = bi["P"][-1] - bi["P"][0], bj["P"][-1] - bj["P"][0]
            c = abs(np.dot(di, dj)) / max(np.linalg.norm(di) * np.linalg.norm(dj), 1e-9)
            if c < np.cos(np.radians(TWIN_ANGLE)):
                continue
            short = bi if len(bi["P"]) and np.linalg.norm(di) <= np.linalg.norm(dj) else bj
            other = bj if short is bi else bi
            band = other["foot"].buffer(TWIN_GAP + short["hw"] + 0.5)
            if shapely.LineString(short["P"]).intersection(band).length < 0.6 * shapely.LineString(short["P"]).length:
                continue
            pairs.append((i, j))
    for i, j in pairs:
        for me, tw in ((todo[i], todo[j]), (todo[j], todo[i])):
            P, N, hw = me["P"], me["N"], me["hw"]
            masks = []
            for sign in (1.0, -1.0):
                E = P + N * hw * sign
                probe = E + N * sign * (TWIN_GAP / 2)
                d = shapely.distance(tw["foot"], shapely.points(probe))
                masks.append(d <= TWIN_GAP / 2 + 0.5)
            prev = me.get("no_parapet")
            me["no_parapet"] = tuple(m | p for m, p in zip(masks, prev)) if prev is not None else tuple(masks)
            me.setdefault("twin_of", []).append(tw["s"]["tlm"])
    return pairs


def median_geometry(bi, bj):
    """(top triangles, side and underside triangles, footprint, height function) of the deck that closes the gap
    between the twin decks bi and bj: the hull of their edges that face each other less the decks, MEDIAN_KERB m
    over them."""
    E = []                                             # the edges of the two decks that face each other
    for b in (bi, bj):
        for side, sign in ((0, 1.0), (1, -1.0)):
            m = b["no_parapet"][side]
            if m.any():
                E.append(b["P"][m] + b["N"][m] * b["hw"] * sign)
    if len(E) < 2:
        return None
    hull = shapely.MultiPoint(np.vstack(E)).convex_hull
    gap = hull.difference(bi["foot"]).difference(bj["foot"])
    gap = gap.buffer(-0.05, join_style="mitre").buffer(0.05, join_style="mitre")     # no slivers along the decks
    if gap.is_empty:
        return None
    parts = [g for g in getattr(gap, "geoms", [gap]) if g.geom_type == "Polygon" and g.area > 1.0]
    if not parts:
        return None
    foot = max(parts, key=lambda g: g.area)
    PZ = np.vstack([np.column_stack([b["P"], b["z"]]) for b in (bi, bj)])
    from scipy.spatial import cKDTree
    kt = cKDTree(PZ[:, :2])

    def zf(x, y):
        d, k = kt.query(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]), k=min(4, len(PZ)))
        w = 1.0 / np.maximum(d, 0.1)
        return (PZ[k, 2] * w).sum(1) / w.sum(1)
    tris = shapely.constrained_delaunay_triangles(foot)
    top, sides = [], []
    for t in getattr(tris, "geoms", []):
        q = np.array(t.exterior.coords[:3])
        zz = zf(q[:, 0], q[:, 1])
        T = np.column_stack([q, zz + MEDIAN_KERB])[None]
        n = np.cross(T[0, 1] - T[0, 0], T[0, 2] - T[0, 0])
        if n[2] < 0:
            T = T[:, ::-1]
        top.append(T)
        B = np.column_stack([q, zz - SLAB])[None]
        if np.cross(B[0, 1] - B[0, 0], B[0, 2] - B[0, 0])[2] > 0:
            B = B[:, ::-1]
        sides.append(B)
    ring = np.array(foot.exterior.coords)
    zr = zf(ring[:, 0], ring[:, 1])
    c = np.array(foot.centroid.coords[0])
    for k in range(len(ring) - 1):
        a, b_ = ring[k], ring[k + 1]
        out = np.r_[(a + b_) / 2 - c, 0.0]
        sides.append(_quad(np.r_[a, zr[k] + MEDIAN_KERB], np.r_[b_, zr[k + 1] + MEDIAN_KERB], np.r_[b_, zr[k + 1] - SLAB],
                           np.r_[a, zr[k] - SLAB], out))
    cat = lambda L_: np.concatenate(L_) if L_ else np.zeros((0, 3, 3))
    return cat(top), cat(sides), foot, zf


def save_records(records, reviewed=None):
    """ponti.json: the bridge list with the manual corrections kept."""
    old = load_overrides()
    out = []
    for r in sorted(records, key=lambda r: -r["length_m"]):
        o = old.get(r["tlm"], {})
        e = dict(r)
        e.pop("override", None)
        for k in ("z0", "z1", "profile", "type", "skip", "deck", "width", "note", "checked"):
            if k in o:
                e[k] = o[k]
        out.append(e)
    json.dump({"note": "Bridges of the v2.0 network (bridges.py). Manual corrections: z0/z1 (deck height at "
                       "the ends, m), profile 'straight', type 'open'|'culvert', skip true, deck 'survey', width (m, the "
                       "carriageway, v3.1); note, checked.",
               "ponti": out}, open(PONTI, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
