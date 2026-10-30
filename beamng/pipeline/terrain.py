"""BeamNG terrain: TER_SIZE x TER_SIZE vertices TER_SQUARE m apart (config: 8192 at 1.5 m, a
12.3 km square around the playable area). Heights: swissALTI3D 0.5 m (dtm05) over the playable area,
swissALTI3D 2 m (dtm2) around it, blended; Copernicus where neither exists (Italy). Material layers
from the cadastral land cover (MU) with slope-based rock; outside the survey a rough class from the
orthophoto colour. Optional carving (road surfaces etc.) as a height override raster.
The block is processed in row bands (67 M vertices).

build_terrain(level_dir, override=None) writes <level>/theTerrain.ter + .terrain.json
and art/terrains/main.materials.json; returns (z0, max_height).
"""
import json, os
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from config import CRS, WORK, TER_SIZE, TER_SQUARE, TER_X0, TER_Y0, LEVEL_NAME
from geo import Grid, ortho_sampler
from landcover import CODE
import bng

A = "/assets/materials/terrain"
# name: (groundmodel, detail set dir/prefix, macro set, detail size m, macro size m)
TERRAIN_MATS = {
    "Grass":        ("GRASS",   f"{A}/grass/t_grass_01/t_grass_01",               f"{A}/grass/t_macro_grass/t_macro_grass", 4, 40),
    "GardenGrass":  ("GRASS",   f"{A}/grass/t_dirt_vegetation/t_dirt_vegetation", f"{A}/grass/macro_grass/t_macro_grass", 4, 40),
    "ForestFloor":  ("DIRT",    f"{A}/forest/t_forest_ground/t_forest_ground",    f"{A}/forest/t_macro_forest/t_macro_forest", 4, 50),
    "ForestFloor2": ("DIRT",    f"{A}/forest/t_forest_ground_02/t_forest_ground_02", f"{A}/forest/t_macro_dirt_forest/t_macro_dirt_forest", 4, 50),
    "Asphalt":      ("ASPHALT", f"{A}/asphalt/t_asphalt_02/t_asphalt_02",         f"{A}/asphalt/macro_asphalt/t_macro_asphalt", 3, 60),
    "Concrete":     ("ASPHALT", f"{A}/concrete/concrete/t_concrete_damaged",      f"{A}/asphalt/macro_asphalt/t_macro_asphalt", 3, 60),
    "Gravel":       ("GRAVEL",  f"{A}/soil/t_gravel/t_gravel",                    f"{A}/rock/macro_rocky/t_macro_rocky", 3, 50),
    "Rock":         ("ROCK",    f"{A}/rock/t_dirt_rocky/t_dirt_rocky",            f"{A}/rock/macro_rocky/t_macro_rocky", 5, 60),
    "Mud":          ("MUD",     f"{A}/mud/mud/t_mud",                             f"{A}/forest/t_macro_dirt_forest/t_macro_dirt_forest", 4, 50),
    "Moss":         ("GRASS",   f"{A}/forest/t_moss/t_moss",                      f"{A}/forest/t_macro_forest/t_macro_forest", 4, 50),
}
# v3.0: steep banks. A terrain material is projected from above: on the banks along the roads (cut slopes,
# embankments, the ground between hairpins) its texture was stretched into vertical smears. The steep
# vertices of the natural layers (STEEP_BANK degrees and more) take this material, earth with stones and
# roots, drawn with the side projection (the texture laid on the slope, not dropped onto it); Rock too.
TERRAIN_MATS["Bank"] = ("DIRT", f"{A}/forest/t_forest_ground_02/t_forest_ground_02",
                        f"{A}/rock/macro_rocky/t_macro_rocky", 3, 40)
STEEP_BANK = 40.0
SIDE_PROJECTED = ("Rock", "Bank")
COLOR_OF = {"Bank": "ForestFloor2"}      # the measured colour a material without its own takes

# v2.4: the meadows and lawns within a terrain square of the roads and paths: the same look, another
# layer, so the grass of groundcover.py (on Grass and GardenGrass) does not stand through the road meshes
VERGE = {"Grass": "GrassVerge", "GardenGrass": "GardenGrassVerge"}
DECK_COVER = 1.5       # m under the slab of a bridge deck: no grass there either (build_terrain)
for _m, _v in VERGE.items():
    TERRAIN_MATS[_v] = TERRAIN_MATS[_m]
VERGE_OF = {v: m for m, v in VERGE.items()}
VERGE_OF.update(COLOR_OF)
MAT_ORDER = list(TERRAIN_MATS)
MAT_ID = {m: i for i, m in enumerate(MAT_ORDER)}

LC_TO_MAT = {
    "none": "Grass", "edificio": "Concrete", "altro_rivestimento_duro": "Asphalt", "giardino": "GardenGrass",
    "campo_prato_pascolo": "Grass", "strada_sentiero": "Asphalt", "bacino_idrico": "Concrete",
    "bosco_fitto": "ForestFloor", "corso_acqua": "Gravel", "vigna": "GardenGrass", "marciapiede": "Asphalt",
    "altro_bosco": "ForestFloor2", "altro_humus": "GardenGrass", "spartitraffico": "GardenGrass",
    "pietraia_sabbia": "Gravel", "specchio_acqua": "Mud", "canneti": "Moss", "ferrovia": "Gravel",
    "altra_coltura_intensiva": "GardenGrass", "cava_di_ghiaia_discarica": "Gravel",
    "altra_senza_vegetazione": "Gravel", "torbiera": "Moss",
    "pascolo_boscato_fitto": "ForestFloor2", "pascolo_boscato_aperto": "Grass", "roccia": "Rock",
    "ghiacciaio_nevaio": "Rock", "binario": "Gravel", "pista_aerea": "Asphalt", "altro": "Grass",
}


# v3.0: the edges between the natural classes of the land cover (meadow, forest, garden, vineyard, rock...)
# are warped by a smooth noise of WARP m (wavelength about WARP_L m): up to v2.8 they followed the 1.5 m
# terrain grid, straight lines with stair steps where the cadastral outline runs at an angle. The paved
# classes, buildings and water keep their outline (the meshes of roads and houses sit on it).
WARP = 2.2
WARP_L = 9.0
NATURAL = ("giardino", "campo_prato_pascolo", "bosco_fitto", "vigna", "altro_bosco", "altro_humus",
           "pietraia_sabbia", "canneti", "altra_coltura_intensiva", "altra_senza_vegetazione", "torbiera",
           "pascolo_boscato_fitto", "pascolo_boscato_aperto", "roccia", "altro")


def warp_band(r0, r1, seed=17):
    """(dx, dy) m of the warp for the terrain rows r0:r1 (every column), smooth and repeatable."""
    rng = np.random.default_rng(seed)
    cell = WARP_L / TER_SQUARE                       # terrain samples per noise cell
    nc = int(np.ceil(TER_SIZE / cell)) + 3
    if not hasattr(warp_band, "_g"):
        warp_band._g = [rng.normal(0, 1, (nc, nc)).astype(np.float32) for _ in range(2)]
    rows = (np.arange(r0, r1) / cell)[:, None]
    cols = (np.arange(TER_SIZE) / cell)[None, :]
    R, C = np.broadcast_arrays(rows, cols)
    out = []
    for g in warp_band._g:
        v = map_coordinates(g, [R.ravel(), C.ravel()], order=3, mode="wrap").reshape(R.shape)
        out.append(WARP * v / 1.3)
    return out[0], out[1]


# v3.0: terrain layers outside the survey from the classes of ESA WorldCover (download_worldcover.py)
WC_TO_MAT = {10: "ForestFloor", 20: "Grass", 30: "Grass", 40: "GardenGrass", 50: "GardenGrass", 60: "Gravel",
             80: "Mud", 90: "Moss", 100: "Grass"}


# v3.0: the detail textures (the blades, stones and leaves of the ground) fade out between 70 and 160 m
# (v2.8: 40-80 m, so a meadow 100 m away was already flat colour); the macro textures vary more
DETAIL_DIST = [0, 0, 70, 160]


# pixel size of the base textures (the TerrainMaterialTextureSet baseTexSize). 1024 since v2.5:
# 13 materials x 5 maps at 2048 took about 1 GB of the game's memory, for smooth colour noise
# that is only seen from afar (the detail textures cover it near the camera)
BASE_TEX = 1024


# v3.0: the base colour of every open-ground material is a map over the whole terrain block (one pixel every
# BASE_SIZE_M / BASE_TEX m, about 12 m), the colour of the orthophoto on the ground of that material
# (the crowns and their shadows left out) where there is some, the measured colour of the material elsewhere:
# the meadows have their real light and dark patches (mown, dry, wet, in the shade of a slope) seen from
# afar, where up to v2.8 the whole map had one colour per material with a faint noise tiled every 256 m.
# The forest floors (under the crowns, never seen in the orthophoto) keep the measured colour and noise.
BASE_SIZE_M = TER_SIZE * TER_SQUARE
BASE_FLIP = True               # image row 0 at the north edge of the block (terrain row TER_SIZE - 1)
# v3.1 review: not Concrete: on its cells the orthophoto shows the roofs and the water of the pools (grey-blue)
ORTHO_MATS = ("Grass", "GardenGrass", "Gravel", "Rock", "Moss", "Mud", "Asphalt")
ORTHO_SHARE = 0.75             # of the orthophoto colour against the measured one
ORTHO_MAX_DEV = 0.28           # largest relative departure of a pixel from the measured colour, each channel
CANOPY_V = 92                  # orthophoto pixels darker than this and green are crowns or shadows


def base_color_maps(colors, layers):
    """{material: (BASE_TEX, BASE_TEX, 3) uint8} base colour maps of the ORTHO_MATS from the orthophoto
    (WORK/ortho.tif) on the cells of their own layer (layers: the terrain layer grid, row 0 south)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from geo import grid_transform
    n = BASE_TEX
    f = TER_SIZE // n                                   # terrain samples per base pixel
    sums = {m: np.zeros((n, n, 3), np.float64) for m in ORTHO_MATS}
    cnts = {m: np.zeros((n, n), np.float64) for m in ORTHO_MATS}
    ids = {m: MAT_ID[m] for m in ORTHO_MATS}
    verge = {MAT_ID[v]: MAT_ID[m] for m, v in VERGE.items()}
    with rasterio.open(os.path.join(WORK, "ortho.tif")) as src:
        band_px = BAND // f * f
        for r0 in range(0, TER_SIZE, band_px):
            r1 = min(r0 + band_px, TER_SIZE)
            # the orthophoto on the terrain samples of rows r0:r1 (row 0 south: the band's north edge is r1 - 1)
            y_top = TER_Y0 + (r1 - 1) * TER_SQUARE + TER_SQUARE / 2
            tr = grid_transform(TER_X0 - TER_SQUARE / 2, y_top, TER_SQUARE)
            o = np.zeros((3, r1 - r0, TER_SIZE), np.uint8)
            reproject(source=rasterio.band(src, [1, 2, 3]), destination=o, src_transform=src.transform,
                      src_crs=src.crs, dst_transform=tr, dst_crs=CRS, resampling=Resampling.bilinear,
                      src_nodata=0, dst_nodata=0)
            o = o[:, ::-1].transpose(1, 2, 0).astype(np.float32)        # rows south -> north like the layers
            lay = layers[r0:r1].copy()
            for v, m in verge.items():
                lay[lay == v] = m
            R_, G_, B_ = o[..., 0], o[..., 1], o[..., 2]
            ok = (o > 0).all(2) & ~(((R_ + G_ + B_) / 3 < CANOPY_V) & (G_ > R_) & (G_ > B_))
            rb0, rb1 = r0 // f, r1 // f
            for m, i in ids.items():
                sel = ok & (lay == i)
                if not sel.any():
                    continue
                w = sel.astype(np.float64)
                cnts[m][rb0:rb1] += w.reshape(rb1 - rb0, f, n, f).sum((1, 3))
                sums[m][rb0:rb1] += (o * w[..., None]).reshape(rb1 - rb0, f, n, f, 3).sum((1, 3))
    out = {}
    from scipy.ndimage import gaussian_filter
    for m in ORTHO_MATS:
        c = np.array(colors[m], np.float64)
        cnt, sm = cnts[m], sums[m]
        # fill the pixels without that ground from the ones around (normalised convolution), then the measured colour
        wsum = gaussian_filter(cnt, 2.0)
        mean = np.stack([gaussian_filter(sm[..., k], 2.0) for k in range(3)], -1) / np.maximum(wsum, 1e-6)[..., None]
        have = cnt > 0.25 * f * f
        mean[have] = sm[have] / cnt[have][:, None]
        trust = np.clip(wsum / (0.25 * f * f), 0, 1)[..., None] * ORTHO_SHARE
        # the orthophoto's colour, brought to the material's measured brightness on average (it is pale with haze)
        ok = cnt > 0
        if ok.sum() > 100:
            gain = c / np.maximum(np.median(sm[ok] / cnt[ok][:, None], 0), 1)
            mean = mean * gain[None, None, :]
        mean = np.clip(mean, c * (1 - ORTHO_MAX_DEV), c * (1 + ORTHO_MAX_DEV))
        img = c[None, None, :] * (1 - trust) + mean * trust
        if BASE_FLIP:
            img = img[::-1]
        out[m] = np.clip(img, 0, 255).astype(np.uint8)
        print("base colour map %s: %.1f %% of its pixels from the orthophoto" % (m, 100 * have.mean()), flush=True)
    return out


def terrain_materials(level, base_tex):
    """TerrainMaterial dicts. base_tex: {material: path of its base colour texture (level art)}."""
    mats = {}
    for name, (gm, det, mac, dsize, msize) in TERRAIN_MATS.items():
        base = base_tex[name]
        m = {
            "internalName": name, "class": "TerrainMaterial", "persistentId": bng.pid(),
            "groundmodelName": gm,
            "baseColorBaseTex": base["b"], "baseColorBaseTexSize": base["size"],
            "normalBaseTex": f"/levels/{level}/art/terrains/t_flat_nm.png", "normalBaseTexSize": base["size"],
            "roughnessBaseTex": f"/levels/{level}/art/terrains/t_rough_r.png", "roughnessBaseTexSize": base["size"],
            "aoBaseTex": f"/levels/{level}/art/terrains/t_white_ao.png", "aoBaseTexSize": base["size"],
            "heightBaseTex": f"/levels/{level}/art/terrains/t_grey_h.png", "heightBaseTexSize": base["size"],
            "baseColorDetailTex": f"{det}_b.png", "normalDetailTex": f"{det}_nm.png",
            "roughnessDetailTex": f"{det}_r.png", "aoDetailTex": f"{det}_ao.png", "heightDetailTex": f"{det}_h.png",
            "baseColorDetailTexSize": dsize, "normalDetailTexSize": dsize, "roughnessDetailTexSize": dsize,
            "aoDetailTexSize": dsize, "heightDetailTexSize": dsize,
            "baseColorDetailStrength": [0.45, 0.2], "normalDetailStrength": [1.0, 0.4],
            "roughnessDetailStrength": [0.5, 0.3], "aoDetailStrength": [0.7, 0.2],
            "baseColorMacroTex": f"{mac}_b.png", "normalMacroTex": f"{mac}_nm.png",
            "roughnessMacroTex": f"{mac}_r.png", "aoMacroTex": f"{mac}_ao.png", "heightMacroTex": f"{mac}_h.png",
            "baseColorMacroTexSize": msize, "normalMacroTexSize": msize, "roughnessMacroTexSize": msize,
            "aoMacroTexSize": msize, "heightMacroTexSize": msize,
            "baseColorMacroStrength": [0.25, 0.15], "normalMacroStrength": [0.4, 0.3],
            "roughnessMacroStrength": [0.3, 0.3], "aoMacroStrength": [0.3, 0.2],
            "detailDistances": DETAIL_DIST, "macroDistances": [0, 10, 150, 3000],
        }
        if name in SIDE_PROJECTED:
            m["useSideProjection"] = True
        mats[f"{name}-{m['persistentId']}"] = m
    mats["TextureSet"] = {"name": f"{level}TerrainMaterialTextureSet", "class": "TerrainMaterialTextureSet",
                          "baseTexSize": [BASE_TEX, BASE_TEX], "detailTexSize": [1024, 1024], "macroTexSize": [1024, 1024]}
    return mats


def vertex_coords():
    xs = TER_X0 + np.arange(TER_SIZE) * TER_SQUARE
    ys = TER_Y0 + np.arange(TER_SIZE) * TER_SQUARE
    return xs, ys                 # x of columns, y of rows (row 0 = south)


BAND = 256                        # terrain rows per band
BLEND = 40.0                      # m, fine DTM -> coarse DTM transition at the edge of the fine grid


def fine_weight(dtm):
    """Weight of dtm05 against dtm2 on a 5 m grid: 1 on swissALTI3D 0.5 m data, 0 on cells filled
    from Copernicus and at the grid edge, smooth in between."""
    nod = np.load(os.path.join(WORK, "dtm05_nodata.npy"), mmap_mode="r")
    h, w = nod.shape
    f = 10                                               # 0.5 m cells per 5 m cell
    hc, wc = h // f, w // f
    valid = np.zeros((hc, wc), np.float32)
    for r0 in range(0, hc, 256):
        blk = np.asarray(nod[r0 * f:min(r0 + 256, hc) * f, :wc * f])
        valid[r0:r0 + blk.shape[0] // f] = 1.0 - blk.reshape(-1, f, wc, f).mean((1, 3))
    valid[:2], valid[-2:], valid[:, :2], valid[:, -2:] = 0, 0, 0, 0
    wgt = np.clip((gaussian_filter(valid, BLEND / 5.0 / 2.0) - 0.5) * 2.0, 0.0, 1.0)
    return Grid(wgt.astype(np.float32), dtm.x_min, dtm.y_max, dtm.res * f)


def heights_band(xs, ys, dtm, dtm2, wgrid):
    """Terrain heights (len(ys), len(xs)) for one band of rows."""
    X, Y = np.meshgrid(xs, ys)
    H = dtm2.sample(X.ravel(), Y.ravel()).reshape(X.shape).astype(np.float64)
    w = wgrid.sample(X.ravel(), Y.ravel()).reshape(X.shape)
    # outside the 0.5 m raster the samplers repeat its edge: no weight there
    bx0, by0, bx1, by1 = dtm.bounds()
    w[(X < bx0 + 1) | (X > bx1 - 1) | (Y < by0 + 1) | (Y > by1 - 1)] = 0.0
    m = w > 0
    if m.any():
        x0, x1 = X[m].min(), X[m].max()
        y0, y1 = Y[m].min(), Y[m].max()
        sub, _, _ = dtm.window(x0, y0, x1, y1, pad=6)
        sub.a = gaussian_filter(sub.a.astype(np.float32), 0.4 * TER_SQUARE / dtm.res)   # anti-alias
        H[m] = w[m] * sub.sample(X[m], Y[m]) + (1 - w[m]) * H[m]
    return H


def build_terrain(level_dir, override=None, override_mask=None, layer_override=None, post_fn=None, cap=None,
                  no_cover=None):
    dtm = Grid.load(os.path.join(WORK, "dtm05.npz"))
    dtm2 = Grid.load(os.path.join(WORK, "dtm2.npz"))
    wgrid = fine_weight(dtm)
    lc = np.load(os.path.join(WORK, "landcover05.npz"))
    lcg = Grid(lc["a"], float(lc["x_min"]), float(lc["y_max"]), float(lc["res"]))
    ortho = ortho_sampler()
    xs, ys = vertex_coords()
    from landcover import CLASSES
    lut = np.array([MAT_ID[LC_TO_MAT[cname]] for cname in CLASSES], np.uint8)
    natural = np.array([cname in NATURAL for cname in CLASSES])
    H = np.zeros((TER_SIZE, TER_SIZE), np.float64)
    layers = np.zeros((TER_SIZE, TER_SIZE), np.uint8)
    for r0 in range(0, TER_SIZE, BAND):
        r1 = min(r0 + BAND, TER_SIZE)
        H[r0:r1] = heights_band(xs, ys[r0:r1], dtm, dtm2, wgrid)
    if override is not None:
        H = np.where(override_mask, override, H)
    if cap is not None:                                  # under bridge decks: no higher than the slab
        H = np.minimum(H, cap)
    if post_fn is not None:
        H = post_fn(H, xs, ys)
        if override is not None:                         # walls may lower it, nothing may raise it over a road
            H = np.where(override_mask, np.minimum(H, override), H)
        if cap is not None:
            H = np.minimum(H, cap)
    gy, gx = np.gradient(H, TER_SQUARE)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    del gx, gy
    grassy = [MAT_ID["Grass"], MAT_ID["ForestFloor"], MAT_ID["ForestFloor2"], MAT_ID["GardenGrass"]]
    try:
        import forest_fill
        wc = forest_fill.worldcover_grid()
    except FileNotFoundError:
        wc = None
    for r0 in range(0, TER_SIZE, BAND):
        r1 = min(r0 + BAND, TER_SIZE)
        X, Y = np.meshgrid(xs, ys[r0:r1])
        L = lcg.sample(X.ravel(), Y.ravel(), order=0).astype(np.int64).reshape(X.shape)
        lx0, ly0, lx1, ly1 = lcg.bounds()
        L[(X < lx0) | (X >= lx1) | (Y <= ly0) | (Y > ly1)] = 0          # outside the survey raster
        # v3.0: the edges between meadow, forest, garden and rock follow a wavy line, not the 1.5 m grid
        if WARP > 0:
            dx, dy = warp_band(r0, r1)
            Lw = lcg.sample((X + dx).ravel(), (Y + dy).ravel(), order=0).astype(np.int64).reshape(X.shape)
            nat = natural[L] & natural[Lw]
            L = np.where(nat, Lw, L)
        lay = lut[L]
        rgb = ortho(X.ravel(), Y.ravel()).reshape(X.shape + (3,))
        R_, G_, B_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        green = (G_ > R_ + 4) & (G_ > B_ + 4)
        # outside the survey (L == 0): a rough class from the orthophoto colour
        out = L == 0
        dark = np.nan_to_num((R_ + G_ + B_) / 3.0, nan=100.0) < 85
        lay[out & green & dark] = MAT_ID["ForestFloor"]
        lay[out & green & ~dark] = MAT_ID["Grass"]
        lay[out & ~green & (slope[r0:r1] > 35)] = MAT_ID["Rock"]
        # v3.0: and from WorldCover where it has a class (both sides of the border; up to v2.8 the Italian
        # side, without an orthophoto, was all meadow): the forest of the far trees of forest_fill.py
        if wc is not None:
            W = wc.sample(X.ravel(), Y.ravel(), order=0).reshape(X.shape)
            for cls_, mat in WC_TO_MAT.items():
                lay[out & (W == cls_)] = MAT_ID[mat]
        # steep bare slopes -> rock, but only where the orthophoto is not green (vegetated / ivy slopes
        # along the road stay vegetated, as in the photos)
        lay[(slope[r0:r1] > 50) & ~green & np.isin(lay, grassy)] = MAT_ID["Rock"]
        lay[(slope[r0:r1] > STEEP_BANK) & np.isin(lay, grassy)] = MAT_ID["Bank"]
        layers[r0:r1] = lay
    del slope
    if layer_override is not None:
        m = layer_override >= 0
        layers[m] = layer_override[m]
    if cap is not None:
        # v3.1 review: and the ground under a bridge deck within DECK_COVER m of its slab (the end of a deck on its bank):
        # the Ponte Tresa deck, 11 m wide since v3.1, lies on the meadow of its Italian bank, and the grass of that
        # layer stood through the asphalt (the road carve that makes no_cover leaves the decks out)
        under = np.isfinite(cap) & (H > cap - DECK_COVER)
        if under.any():
            from scipy.ndimage import binary_dilation
            under = binary_dilation(under, iterations=1)
            no_cover = under if no_cover is None else (no_cover | under)
    if no_cover is not None:                             # v2.4: the verges of the roads (VERGE)
        for a, b in VERGE.items():
            layers[no_cover & (layers == MAT_ID[a])] = MAT_ID[b]
    z0 = float(np.floor(H.min() - 2.0))
    max_h = float(np.ceil(H.max() - z0 + 2.0))
    bng.write_ter(os.path.join(level_dir, "theTerrain.ter"), H, z0, max_h, layers, MAT_ORDER)
    bng.write_terrain_json(os.path.join(level_dir, "theTerrain.terrain.json"), LEVEL_NAME, "theTerrain",
                           TER_SIZE, MAT_ORDER)
    np.save(os.path.join(WORK, "terrain_heights.npy"), H.astype(np.float32))
    np.save(os.path.join(WORK, "terrain_layers.npy"), layers)
    return z0, max_h, H
