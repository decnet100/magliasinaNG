"""The playable area of the map (v2.0): the boundary drawn by the user, widened by AREA_MARGIN,
joined with the Street View route (ROUTE_MARGIN around it, so the whole original road stays in) and
with the cantonal road from Magliaso along the lake to Agno and up the Vedeggio valley to Gravesano
(dati/cantonale_gravesano.json, strade_extra.py): EXTRA_ROAD_MARGIN around it and the strip between
it and the side P1-P2 of the boundary, so no bare slope is left between the map and the road.
v2.2: joined with a corridor of EXTRA_CORRIDOR m around the roads of dati/strade_extra_v22.json
(strade_extra.py): the pass above Gravesano to Arosio (Stradón da Rós, the "Penudria"), the cantonal
road from Ponte Tresa through Caslano to Magliaso and the road from Caslano along the lake to the
Torrazza (Via Torrazza). Up to v2.1 they left the map: only the terrain, no road. v2.8: also the cantonal
road from Arosio through Mugena, Vezio, Fescoggia and Breno to Miglieglia and Novaggio. v3.0.40: joined with
the whole villages of dati/paesi_interi.json (paesi_interi.py): Caslano with its lakeside and the Monte Caslano,
and Arosio, which the corridors only cut through.

Everything that is generated only where it can be seen or driven (roads, buildings, trees, raster
data at 0.5 m) uses this polygon; the terrain block (config.TER_*) is a square around it and the
distant backdrop starts at the edge of that square.
"""
import json, os
import numpy as np
import shapely
from config import (AREA_FILE, AREA_MARGIN, BOUNDARY, EXTERNAL, EXTRA_CORRIDOR, EXTRA_ROAD_MARGIN, K, ROUTE_MARGIN,
                    TER_SIZE, TER_SQUARE, TER_X0, TER_X1, TER_Y0, TER_Y1, local_to_lv95, lv95_to_local, wgs_to_local)

DATI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dati")
_cache = {}


def boundary():
    """The quadrilateral drawn by the user (local coordinates)."""
    return shapely.Polygon([wgs_to_local(lat, lon) for lat, lon in BOUNDARY])


def route():
    """The Street View route (main run of the calibrated poses); none on an external site."""
    if EXTERNAL:
        return shapely.LineString()
    if "route" not in _cache:
        poses = json.load(open(os.path.join(DATI, "poses.json")))
        _cache["route"] = shapely.LineString([p["pos"][:2] for p in poses if p["main_run"]])
    return _cache["route"]


EXTRA_ROADS = ("cantonale_gravesano.json",)


def extra_roads():
    """The roads added beyond the boundary (dati/<file> of EXTRA_ROADS, strade_extra.py), as local
    lines."""
    if "extra" not in _cache:
        out = []
        for name in EXTRA_ROADS:
            f = os.path.join(DATI, name)
            if os.path.exists(f):
                E, N = np.array(json.load(open(f, encoding="utf-8"))["lv95"], np.float64).T
                out.append(shapely.LineString(np.column_stack(lv95_to_local(E, N))))
        _cache["extra"] = out
    return _cache["extra"]


CORRIDORS = "strade_extra_v22.json"


def extra_corridors():
    """v2.2: [(key, name, local MultiLineString)] of the roads of dati/strade_extra_v22.json (strade_extra.py)."""
    if "corridors" not in _cache:
        out = []
        f = os.path.join(DATI, CORRIDORS)
        if os.path.exists(f):
            for r in json.load(open(f, encoding="utf-8"))["roads"]:
                parts = []
                for P in r["parts"]:
                    E, N = np.array(P, np.float64).T
                    parts.append(shapely.LineString(np.column_stack(lv95_to_local(E, N))))
                out.append((r["key"], r["name"], shapely.MultiLineString(parts)))
        _cache["corridors"] = out
    return _cache["corridors"]


VILLAGES = "paesi_interi.json"


def villages():
    """v3.0.40: [(key, name, local Polygon)] of the whole villages of dati/paesi_interi.json (paesi_interi.py)."""
    if "villages" not in _cache:
        out = []
        f = os.path.join(DATI, VILLAGES)
        if os.path.exists(f):
            for v in json.load(open(f, encoding="utf-8"))["villages"]:
                rings = []
                for R in v["rings"]:
                    E, N = np.array(R, np.float64).T
                    rings.append(np.column_stack(lv95_to_local(E, N)))
                out.append((v["key"], v["name"], shapely.Polygon(rings[0], rings[1:])))
        _cache["villages"] = out
    return _cache["villages"]


def external_polygon():
    """An external site (config.EXTERNAL): the polygon of MAGLIASO_AREA (GeoJSON in the map CRS), local."""
    from shapely.geometry import shape
    feats = json.load(open(AREA_FILE, encoding="utf-8"))["features"]
    g = shapely.union_all([shape(f["geometry"]) for f in feats])
    return shapely.transform(g, lambda c: np.column_stack(lv95_to_local(c[:, 0], c[:, 1])))


def polygon():
    """The playable area (local coordinates)."""
    if "area" not in _cache:
        _cache["area"] = external_polygon() if EXTERNAL else _polygon(corridors=True)
    return _cache["area"]


def polygon_v21():
    """The area of v2.1, without the corridors of v2.2: the comparisons with the v2.1 level (drive_test.py)."""
    if "area_v21" not in _cache:
        _cache["area_v21"] = _polygon(corridors=False)
    return _cache["area_v21"]


def _polygon(corridors=True, whole=True):
    """The boundary with its margin, the route, the cantonal road to Gravesano, (corridors) the roads of
    v2.2 and (corridors and whole) the whole villages of v3.0.40."""
    parts = [boundary().buffer(AREA_MARGIN, join_style="mitre"), route().buffer(ROUTE_MARGIN)]
    p1, p2 = np.asarray(boundary().exterior.coords)[:2]
    for road in extra_roads():
        # the strip between the road and the side P1-P2 (the road runs from its south end, near
        # P1, to its north end, near P2), and the road with its margin
        parts.append(shapely.Polygon(list(road.coords) + [tuple(p2), tuple(p1)]).buffer(0))
        parts.append(road.buffer(EXTRA_ROAD_MARGIN))
    for _, _, road in (extra_corridors() if corridors else []):
        parts.append(road.buffer(EXTRA_CORRIDOR))
    for _, _, g in (villages() if corridors and whole else []):
        parts.append(g)
    A = shapely.union_all(parts)
    # v2.2: a pocket that a corridor closes against the rest of the area (between the cantonal road of
    # Caslano and the boundary) is part of it
    if A.geom_type == "Polygon" and len(A.interiors):
        A = shapely.Polygon(A.exterior)
    return A


def bounds(margin=0.0):
    """(x_min, y_min, x_max, y_max) of the area, widened by `margin` m."""
    x0, y0, x1, y1 = polygon().bounds
    return x0 - margin, y0 - margin, x1 + margin, y1 + margin


def raster_bounds(res=0.5, margin=60.0):
    """Bounds of the 0.5 m data grids (dtm05, dsm05, landcover05): the area + margin, snapped to res."""
    x0, y0, x1, y1 = bounds(margin)
    return (np.floor(x0 / res) * res, np.floor(y0 / res) * res, np.ceil(x1 / res) * res, np.ceil(y1 / res) * res)


def block_bounds(res=2.0, margin=100.0):
    """Bounds of the coarse grids covering the whole terrain block (dtm2, ortho), snapped to res."""
    x0, y0, x1, y1 = TER_X0 - margin, TER_Y0 - margin, TER_X1 + margin, TER_Y1 + margin
    return (np.floor(x0 / res) * res, np.floor(y0 / res) * res, np.ceil(x1 / res) * res, np.ceil(y1 / res) * res)


def terrain_bounds():
    """(x_min, y_min, x_max, y_max) of the terrain vertices."""
    return TER_X0, TER_Y0, TER_X1, TER_Y1


def lv95_polygon(margin=0.0):
    """The area in LV95 (E, N), widened by `margin` m."""
    g = polygon().buffer(margin) if margin else polygon()
    return shapely.transform(g, lambda c: np.column_stack(local_to_lv95(c[:, 0], c[:, 1])))


def tiles_1km(margin=200.0):
    """LV95 kilometre tiles ('EEEE-NNNN', south-west corner in km) that touch the area + margin."""
    g = lv95_polygon(margin)
    shapely.prepare(g)
    e0, n0, e1, n1 = g.bounds
    out = []
    for e in range(int(np.floor(e0 / 1000)), int(np.floor(e1 / 1000)) + 1):
        for n in range(int(np.floor(n0 / 1000)), int(np.floor(n1 / 1000)) + 1):
            if g.intersects(shapely.box(e * 1000, n * 1000, (e + 1) * 1000, (n + 1) * 1000)):
                out.append(f"{e}-{n}")
    return out


def lv95_rect_tiles(e0, n0, e1, n1):
    """LV95 kilometre tiles covering the rectangle [e0, e1] x [n0, n1]."""
    return [f"{e}-{n}" for e in range(int(np.floor(e0 / 1000)), int(np.floor(e1 / 1000)) + 1)
            for n in range(int(np.floor(n0 / 1000)), int(np.floor(n1 / 1000)) + 1)]


def check():
    """The terrain block covers the area with room to spare."""
    x0, y0, x1, y1 = bounds()
    tx0, ty0, tx1, ty1 = terrain_bounds()
    m = min(x0 - tx0, y0 - ty0, tx1 - x1, ty1 - y1)
    return {"area_km2": polygon().area / 1e6, "bounds": [round(v, 1) for v in (x0, y0, x1, y1)],
            "terrain": [tx0, ty0, tx1, ty1], "terrain_side_m": (TER_SIZE - 1) * TER_SQUARE,
            "min_margin_m": round(m, 1), "K": K}


if __name__ == "__main__":
    print(check())
    print(len(tiles_1km()), "km tiles")
