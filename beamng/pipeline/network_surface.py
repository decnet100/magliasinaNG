"""Smooth idealised heights of the road and path network (v2.0), for every station of network.py.

Longitudinal profile: one sparse least-squares problem over all the stations of the network,
    minimise  sum w (z - d)^2 + sum a/h^4 (second difference of z along a line)^2,
    a = (LC / 2 pi)^4 with the smoothing wavelength LC of the line's class (network.CLASSES),
so the profile keeps the grades and curves of the terrain and loses the lidar noise, bumps and
hollows shorter than about LC. d is the DTM under the line (median across the middle of the
carriageway). The ends of lines that meet at a junction share one height; where two lines carry
on straight through a junction the second difference runs across it too. The fit is iteratively
reweighted with Tukey weights (2 m -> 0.3 m), so parked cars, smeared walls and the edge of a gap
lose their weight. Bridges have no DTM data (the DTM is the ground under them): their deck is the
smooth curve between the approaches, adjusted by beamng/dati/ponti.json (bridges.py); a bridge
where the DTM shows no dip deeper than DIP under it (a culvert) keeps its data.
Along the Strada Cantonale the idealised surface of v1.1 (roadheight.py, corridor fit) is kept:
stations on it are held to it, so the side roads join it without a step.
Cross slope: the plane of the DTM across the carriageway, smoothed along the line, at most
MAX_CROSS; none on bridges, tapered to none at their ends.
Output: work/network_surface.npz: z (height), g (grade), c (cross slope) per station.
"""
import collections, json, os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from config import WORK
from geo import Grid
import network

TUKEY = (2.0, 1.0, 0.6, 0.4, 0.3, 0.3)
MAX_CROSS = {"road": 0.08, "path": 0.15}
CROSS_SMOOTH = {"road": 8.0, "path": 4.0}      # m, Gaussian smoothing of the cross slope along a line
TAPER = 10.0                                    # m, cross slope fades to 0 at the ends of a bridge
W_HOLD = 100.0                                  # weight of the Strada Cantonale surface
W_DECK = 1e-3                                   # weight of the swissTLM3D deck height on a bridge
STRAIGHT = 145.0                                # deg, two lines leaving a node this far apart carry on
DIP = 1.5                                       # m, a bridge over a dip shallower than this rests on the ground
ON_GROUND = 1.0                                 # m, ... if its swissTLM3D line is no higher above the DTM
UNDER_KERB = 0.15                               # m, a footway of an underpass over its road (v3.1)
UNDER_NEAR = 8.0                                # m, ... where the road runs this near


def tangents(st, segs):
    T = np.zeros((len(st["x"]), 2))
    for s in segs:
        a, b = s["first"], s["first"] + s["n"]
        P = np.column_stack([st["x"][a:b], st["y"][a:b]])
        t = np.gradient(P, axis=0)
        T[a:b] = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return T, np.column_stack([-T[:, 1], T[:, 0]])


def dtm_across(dtm, x, y, N, w, fr):
    """DTM at fractions `fr` of the width across the line: (n, len(fr))."""
    out = np.zeros((len(x), len(fr)))
    for k, f in enumerate(fr):
        out[:, k] = dtm.sample(x + N[:, 0] * f * w, y + N[:, 1] * f * w)
    return out


def under_walkways(st, segs, d):
    """v3.1: the paths of a kept underpass (network.underpasses: the two 1 m footways beside Via Golf under the FLP) take
    the height of its road + UNDER_KERB, not the DTM: beside the road the DTM is the railway embankment smeared under
    the deck, and in the 3.1 test build they ran 2 m over the road, 1.2 m under the rail deck (9 'binario' obstacles
    on the paths, check_level). d (DTM data of every station) is changed in place."""
    sel = lambda kind: np.concatenate([np.arange(s["first"], s["first"] + s["n"]) for s in segs
                                       if s["structure"] == network.UNDER and s["kind"] == kind] or [np.zeros(0, int)])
    road, path = sel("road"), sel("path")
    if not len(road) or not len(path):
        return
    dist, j = cKDTree(np.column_stack([st["x"][road], st["y"][road]])).query(np.column_stack([st["x"][path], st["y"][path]]))
    near = dist < UNDER_NEAR
    d[path[near]] = d[road[j[near]]] + UNDER_KERB
    print("  underpass footways on the height of their road: %d stations" % int(near.sum()), flush=True)


def unknowns(st, segs, n_nodes):
    """Index of the unknown of every station: junction ends share their node's."""
    u = np.where(st["node"] >= 0, st["node"], -1).astype(np.int64)
    inner = u < 0
    u[inner] = n_nodes + np.arange(inner.sum())
    return u, n_nodes + int(inner.sum())


def smooth_rows(st, segs, u, node_pos):
    """(rows, cols, vals, weights) of the second differences along lines and through straight junctions."""
    R, C, V, Wt = [], [], [], []
    r = 0
    alpha = lambda lc: (lc / (2 * np.pi)) ** 4
    for s in segs:
        a, n = s["first"], s["n"]
        if n < 3:
            continue
        ss = st["s"][a:a + n]
        h1, h2 = np.diff(ss)[:-1], np.diff(ss)[1:]
        k = np.arange(1, n - 1)
        c0 = 2.0 / (h1 * (h1 + h2)); c2 = 2.0 / (h2 * (h1 + h2)); c1 = -(c0 + c2)
        rows = r + np.arange(n - 2)
        for cc, off in ((c0, -1), (c1, 0), (c2, 1)):
            R.append(rows); C.append(u[a + k + off]); V.append(cc)
        Wt.append(np.full(n - 2, alpha(s["lc"]) * np.mean(np.diff(ss))))
        r += n - 2
    # through junctions: pairs of lines leaving a node in nearly opposite directions
    ends = {}
    for s in segs:
        a, n = s["first"], s["n"]
        if n < 2:
            continue
        for end, (i0, i1) in ((0, (a, a + 1)), (1, (a + n - 1, a + n - 2))):
            node = s["nodes"][end]
            d = np.array([st["x"][i1] - st["x"][i0], st["y"][i1] - st["y"][i0]])
            ends.setdefault(node, []).append((d / max(np.linalg.norm(d), 1e-9), i0, i1, s["lc"]))
    for node, lst in ends.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                di, a0, a1, lca = lst[i]
                dj, b0, b1, lcb = lst[j]
                if np.degrees(np.arccos(np.clip(di @ dj, -1, 1))) < STRAIGHT:
                    continue
                ha = np.hypot(st["x"][a1] - st["x"][a0], st["y"][a1] - st["y"][a0])
                hb = np.hypot(st["x"][b1] - st["x"][b0], st["y"][b1] - st["y"][b0])
                c0 = 2.0 / (ha * (ha + hb)); c2 = 2.0 / (hb * (ha + hb))
                R.append(np.array([r, r, r])); C.append(np.array([u[a1], u[a0], u[b1]]))
                V.append(np.array([c0, -(c0 + c2), c2]))
                Wt.append(np.array([alpha(min(lca, lcb)) * 0.5 * (ha + hb)]))
                r += 1
    return np.concatenate(R), np.concatenate(C), np.concatenate(V), np.concatenate(Wt), r


def cantonale_hold(st):
    """Stations on the Strada Cantonale corridor surface (roadheight.py): (mask, height). None on an external site."""
    from config import EXTERNAL
    if EXTERNAL:
        return np.zeros(len(st["x"]), bool), np.full(len(st["x"]), np.nan)
    import roadheight
    S = roadheight.load()
    x, y = st["x"], st["y"]
    near = S.distance(x, y) < 0.3
    z = np.full(len(x), np.nan)
    if near.any():
        z[near] = S.height(x[near], y[near])
    return near & np.isfinite(z), z


def manual_bridges():
    """The corrections of dati/ponti.json by swissTLM3D id (bridges.py keeps them there)."""
    from config import PONTI as f
    if not os.path.exists(f):
        return {}
    return {b["tlm"]: b for b in json.load(open(f, encoding="utf-8"))["ponti"]}


def solve(verbose=True):
    segs, st, node_pos = network.load()
    dtm = Grid.load(os.path.join(WORK, "dtm05.npz"))
    T, N = tangents(st, segs)
    x, y, w = st["x"], st["y"], st["width"]
    seg = st["seg"]
    kind = np.array([segs[k]["kind"] for k in seg])
    bridge = np.array([segs[k]["bridge"] for k in seg])
    D = dtm_across(dtm, x, y, N, w, (-1 / 3, -1 / 6, 0.0, 1 / 6, 1 / 3))
    d = np.median(D, axis=1)
    under_walkways(st, segs, d)
    u, n_unk = unknowns(st, segs, len(node_pos))
    hold, zhold = cantonale_hold(st)
    # not on a bridge (its deck is not the surface beside it) nor where the corridor surface is
    # far from the DTM (the edge of a lower street beside the line)
    hold &= ~bridge & (np.abs(np.nan_to_num(zhold, nan=1e9) - d) < 3.0)
    w0 = np.where(bridge, 0.0, 1.0)
    # a bridge whose ground the DTM shows unbroken under it (a culvert, a stream under the road,
    # a gap the DTM fills in) and whose swissTLM3D line (on a bridge: the deck) runs on that
    # ground keeps the road on the ground: its stations keep their data. Not over water: the DTM
    # is flat there, the deck is above it.
    zt = st["z_tlm"]
    for s in segs:
        a, n = s["first"], s["n"]
        if s["bridge"] and n >= 2:
            ss = st["s"][a:a + n]
            line = d[a] + (d[a + n - 1] - d[a]) * ss / max(ss[-1], 1e-9)
            above = np.nanmax(zt[a:a + n] - d[a:a + n]) if np.isfinite(zt[a:a + n]).any() else 0.0
            if (d[a:a + n] - line).min() > -DIP and above < ON_GROUND:
                w0[a:a + n] = 1.0
    # the type set by hand in dati/ponti.json decides: a culvert keeps the road on the ground (its
    # data), an open bridge has none
    manual = manual_bridges()
    for s in segs:
        t = manual.get(s["tlm"], {}).get("type") if s["bridge"] else None
        if t in ("culvert", "open"):
            w0[s["first"]:s["first"] + s["n"]] = 1.0 if t == "culvert" else 0.0
    w0[hold] = 0.0
    Rs, Cs, Vs, Ws, n_s = smooth_rows(st, segs, u, node_pos)
    S_mat = sp.csr_matrix((Vs, (Rs, Cs)), shape=(n_s, n_unk))
    Q = (S_mat.T @ sp.diags(Ws) @ S_mat).tocsr()
    rows = np.arange(len(x))
    Pm = sp.csr_matrix((np.ones(len(x)), (rows, u)), shape=(len(x), n_unk))
    zdata = np.where(hold, zhold, d)
    # pieces of the network without any data (a footbridge not joined to anything in the area):
    # the heights of the swissTLM3D line, kept whole
    S_abs = abs(S_mat)
    ncomp, lab = connected_components((S_abs.T @ S_abs) + sp.eye(n_unk), directed=False)
    has = np.bincount(lab[u], weights=w0 + hold, minlength=ncomp) > 0
    fixed = ~has[lab[u]]
    zt = st["z_tlm"]
    zdata[fixed] = np.where(np.isfinite(zt[fixed]) & (np.abs(zt[fixed] - d[fixed]) < 30), zt[fixed], d[fixed])
    w0[fixed] = 1.0
    # a bridge with an end that joins nothing (a dead end, or the network cut at the edge of the area)
    # has only the smoothness to hold it, which leaves its grade free (the deck would run on along
    # any slope): a faint pull to the swissTLM3D line, the deck, holds it there. A bridge held at both
    # ends keeps the curve between them.
    degree = collections.Counter(nd for q in segs for nd in q["nodes"])
    loose = np.array([segs[k]["bridge"] and min(degree[nd] for nd in segs[k]["nodes"]) == 1 for k in range(len(segs))])
    deck = loose[seg] & (w0 == 0) & ~hold & ~fixed & np.isfinite(zt) & (np.abs(np.nan_to_num(zt) - d) < 30)
    zdata[deck] = zt[deck]
    w_deck = np.where(deck, W_DECK, 0.0)
    wt = w0.copy()
    for it, c in enumerate(TUKEY + (None,)):
        ww = wt + np.where(hold, W_HOLD, 0.0) + w_deck
        A = Q + (Pm.T @ sp.diags(ww) @ Pm)
        b = Pm.T @ (ww * zdata)
        A = A + sp.eye(n_unk) * 1e-9
        z_u = spsolve(A.tocsc(), b)
        z = z_u[u]
        if c is None:
            break
        r = z - zdata
        t = np.clip(np.abs(r) / c, 0, 1)
        wt = w0 * (1 - t ** 2) ** 2
        wt[fixed] = 1.0
        # a piece whose data were all rejected keeps them all (no weights left to hold it up)
        lost = ~(np.bincount(lab[u], weights=wt + hold, minlength=ncomp) > 0)[lab[u]]
        wt[lost] = w0[lost]
        if verbose:
            print("  IRLS %d (%.1f m): data kept %.1f %%, |z - DTM| p50 %.3f p99 %.2f" %
                  (it, c, 100 * (wt[w0 > 0] > 0).mean(), np.median(np.abs(r[w0 > 0])), np.percentile(np.abs(r[w0 > 0]), 99)), flush=True)
    if verbose:
        print("  pieces without data (heights of swissTLM3D):", int(len(set(lab[u][fixed].tolist()))),
              "stations", int(fixed.sum()), flush=True)
    # deck heights set by hand (dati/ponti.json: z0, z1, profile 'straight' or 'tlm'; bridges.py builds
    # the deck on them): the line follows them too, from the heights at its ends where only the shape
    # is set. 'tlm': the 3D line of swissTLM3D (a footbridge over a road with its stairs), moved to
    # meet the network at both ends
    for s in segs:
        m = manual.get(s["tlm"], {}) if s["bridge"] else {}
        a, n = s["first"], s["n"]
        ss = st["s"][a:a + n]
        f = (ss - ss[0]) / max(ss[-1] - ss[0], 1e-9)
        if "z0" in m or "z1" in m or m.get("profile") == "straight":
            z0, z1 = float(m.get("z0", z[a])), float(m.get("z1", z[a + n - 1]))
            z[a:a + n] = z0 + (z1 - z0) * f
        elif m.get("profile") == "tlm" and np.isfinite(zt[a:a + n]).sum() >= 2:
            zl = zt[a:a + n].copy()
            ok = np.isfinite(zl)
            zl = np.interp(ss, ss[ok], zl[ok])
            z[a:a + n] = zl + (z[a] - zl[0]) * (1 - f) + (z[a + n - 1] - zl[-1]) * f
    # grade and cross slope
    g = np.zeros(len(x))
    cross = np.zeros(len(x))
    fr = np.linspace(-0.4, 0.4, 7)
    Dc = dtm_across(dtm, x, y, N, w, fr)
    tt = fr[None, :] * w[:, None]
    tm = tt - tt.mean(1, keepdims=True)
    cr = ((Dc - Dc.mean(1, keepdims=True)) * tm).sum(1) / np.maximum((tm ** 2).sum(1), 1e-9)
    for s in segs:
        a, n = s["first"], s["n"]
        sl = slice(a, a + n)
        ss = st["s"][sl]
        g[sl] = np.nan_to_num(np.gradient(z[sl], ss)) if n >= 2 and ss[-1] > 0 else 0.0
        cc = cr[sl].copy()
        sig = CROSS_SMOOTH[s["kind"]] / network.STEP
        cc = gaussian_filter1d(cc, sig, mode="nearest") if n > 2 else cc
        cc = np.clip(cc, -MAX_CROSS[s["kind"]], MAX_CROSS[s["kind"]])
        if s["bridge"]:
            cc[:] = 0.0
        cross[sl] = cc
    # taper the cross slope to 0 approaching a bridge end (both lines at the bridge's nodes)
    bridge_nodes = {n for s in segs if s["bridge"] for n in s["nodes"]}
    for s in segs:
        if s["bridge"]:
            continue
        a, n = s["first"], s["n"]
        ss = st["s"][a:a + n]
        f = np.ones(n)
        if s["nodes"][0] in bridge_nodes:
            f = np.minimum(f, np.clip(ss / TAPER, 0, 1))
        if s["nodes"][1] in bridge_nodes:
            f = np.minimum(f, np.clip((ss[-1] - ss) / TAPER, 0, 1))
        cross[a:a + n] *= f
    np.savez_compressed(os.path.join(WORK, "network_surface.npz"), z=z, g=g, c=cross, d=d, w=wt, hold=hold,
                        nx=N[:, 0], ny=N[:, 1])
    info = {"stations": int(len(x)), "unknowns": int(n_unk), "held_to_cantonale": int(hold.sum()),
            "bridge_stations": int(bridge.sum()),
            "abs_z_minus_dtm_p50": float(np.median(np.abs(z - d)[w0 > 0])),
            "abs_z_minus_dtm_p99": float(np.percentile(np.abs(z - d)[w0 > 0], 99)),
            "data_rejected_pct": float(100 * (wt[w0 > 0] == 0).mean())}
    json.dump(info, open(os.path.join(WORK, "network_surface.json"), "w"), indent=1)
    print("network surface:", info, flush=True)
    return z, g, cross


def load():
    return dict(np.load(os.path.join(WORK, "network_surface.npz")))


if __name__ == "__main__":
    solve()
