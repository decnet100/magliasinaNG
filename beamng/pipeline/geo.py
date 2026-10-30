"""Raster helpers: resample swisstopo LV95 tiles onto grids of the local map frame.

Local frame = LV95 shifted to (E0, N0) and divided by K, so a local grid of
resolution r metres is an LV95 grid of resolution r*K; rasterio does the rest.
Arrays are returned north-up (row 0 = largest y), like the source GeoTIFFs.
"""
import glob, os
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from config import CRS, DATA, E0, N0, K, WORK


def grid_transform(x_min, y_max, res):
    """Affine (map CRS) of a north-up local grid whose top-left corner is (x_min, y_max)."""
    return Affine(res * K, 0, E0 + x_min * K, 0, -res * K, N0 + y_max * K)


def local_grid(x_min, y_min, x_max, y_max, res):
    w = int(round((x_max - x_min) / res))
    h = int(round((y_max - y_min) / res))
    return grid_transform(x_min, y_max, res), w, h


def mosaic(pattern, x_min, y_min, x_max, y_max, res, resampling=Resampling.bilinear,
           dtype=np.float32, nodata=np.nan, src_crs=CRS):
    """Resample every tile matching `pattern` into one local grid (bands, h, w). Each tile is
    resampled into the window of the grid it covers only (large grids: v2.0 area)."""
    files = sorted(glob.glob(os.path.join(DATA, pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    tr, w, h = local_grid(x_min, y_min, x_max, y_max, res)
    with rasterio.open(files[0]) as s:
        bands = s.count
    out = np.full((bands, h, w), nodata, dtype)
    # LV95 bounds of the destination for a cheap tile filter
    e1, n2 = tr * (0, 0)
    e2, n1 = tr * (w, h)
    inv = ~tr
    for f in files:
        with rasterio.open(f) as s:
            b = s.bounds
            if src_crs == CRS:
                if b.right < e1 or b.left > e2 or b.top < n1 or b.bottom > n2:
                    continue
                corners = [(b.left, b.top), (b.right, b.bottom)]
            else:
                from rasterio.warp import transform as warp_pts
                xs, ys = warp_pts(s.crs, CRS, [b.left, b.right, b.left, b.right],
                                  [b.bottom, b.bottom, b.top, b.top])
                corners = list(zip(xs, ys))
            cr = [inv * c for c in corners]
            c0 = max(int(np.floor(min(c for c, _ in cr))) - 2, 0)
            c1 = min(int(np.ceil(max(c for c, _ in cr))) + 2, w)
            r0 = max(int(np.floor(min(r for _, r in cr))) - 2, 0)
            r1 = min(int(np.ceil(max(r for _, r in cr))) + 2, h)
            if c1 <= c0 or r1 <= r0:
                continue
            wtr = tr * Affine.translation(c0, r0)
            tmp = np.full((bands, r1 - r0, c1 - c0), nodata, dtype)
            reproject(source=rasterio.band(s, list(range(1, bands + 1))), destination=tmp,
                      src_transform=s.transform, src_crs=s.crs, src_nodata=s.nodata,
                      dst_transform=wtr, dst_crs=CRS, dst_nodata=nodata,
                      resampling=resampling)
            m = ~np.isnan(tmp) if np.issubdtype(dtype, np.floating) else tmp != nodata
            dst = out[:, r0:r1, c0:c1]
            dst[m] = tmp[m]
    return out


def smoothed_sampler(grid, sigma):
    """fn(x, y): heights of `grid` smoothed with a Gaussian of `sigma` cells, computed on the
    window of the query points only (the v2.0 grids are too large to smooth whole)."""
    from scipy.ndimage import gaussian_filter
    pad = int(np.ceil(4 * sigma)) + 2

    def fn(x, y):
        x = np.atleast_1d(np.asarray(x, np.float64))
        y = np.atleast_1d(np.asarray(y, np.float64))
        if len(x) == 0:
            return np.zeros(0)
        sub, _, _ = grid.window(np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y), pad=pad)
        sub.a = gaussian_filter(np.asarray(sub.a, np.float32), sigma)
        return sub.sample(x, y)
    return fn


def ortho_sampler(path=None):
    """Colour of the orthophoto (an LV95 GeoTIFF, WORK/ortho.tif) at local points:
    fn(x, y) -> (n, 3) float32 RGB 0-255 (NaN outside the image or on its nodata)."""
    path = path or os.path.join(WORK, "ortho.tif")
    with rasterio.open(path) as s:
        o = s.read()
        inv = ~s.transform

    def fn(x, y):
        x = np.atleast_1d(np.asarray(x, np.float64))
        y = np.atleast_1d(np.asarray(y, np.float64))
        c, r = inv * (E0 + x * K, N0 + y * K)
        c = np.floor(c).astype(np.int64)
        r = np.floor(r).astype(np.int64)
        ok = (c >= 0) & (c < o.shape[2]) & (r >= 0) & (r < o.shape[1])
        out = np.full((len(x), 3), np.nan, np.float32)
        v = o[:, r[ok], c[ok]].T.astype(np.float32)
        v[(v == 0).all(1)] = np.nan
        out[ok] = v
        return out
    return fn


def save_geotiff(path, arr, x_min, y_max, res, nodata=None):
    arr = arr if arr.ndim == 3 else arr[None]
    prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                dtype=arr.dtype, crs=CRS, transform=grid_transform(x_min, y_max, res),
                compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr)


class Grid:
    """A north-up local raster with bilinear sampling at arbitrary local (x, y).
    Large grids are stored as <name>.npy (+ .json header) and memory-mapped when loaded."""

    BIG = 50_000_000          # cells: store as .npy for memory mapping

    def __init__(self, arr, x_min, y_max, res):
        self.a, self.x_min, self.y_max, self.res = arr, x_min, y_max, res

    @staticmethod
    def _npy(name):
        stem = name[:-4] if name.endswith(".npz") else name
        return stem + ".npy", stem + ".json"

    @classmethod
    def load(cls, name, mmap=True):
        npy, js = cls._npy(name)
        if os.path.exists(npy) and os.path.exists(js):
            import json
            h = json.load(open(js))
            return cls(np.load(npy, mmap_mode="r" if mmap else None), h["x_min"], h["y_max"], h["res"])
        d = np.load(name, allow_pickle=False)
        return cls(d["a"], float(d["x_min"]), float(d["y_max"]), float(d["res"]))

    def save(self, name):
        npy, js = self._npy(name)
        if self.a.size >= self.BIG:
            import json
            np.save(npy, self.a)
            json.dump({"x_min": float(self.x_min), "y_max": float(self.y_max), "res": float(self.res),
                       "shape": list(self.a.shape)}, open(js, "w"))
            if os.path.exists(name):
                os.remove(name)
        else:
            np.savez(name, a=self.a, x_min=self.x_min, y_max=self.y_max, res=self.res)
            for f in (npy, js):
                if os.path.exists(f):
                    os.remove(f)

    @property
    def shape(self):
        return self.a.shape[-2:]

    def bounds(self):
        """(x_min, y_min, x_max, y_max) of the cell edges."""
        h, w = self.shape
        return self.x_min, self.y_max - h * self.res, self.x_min + w * self.res, self.y_max

    def rc(self, x, y):
        """Fractional (row, col) of local points (pixel centres at integer + 0.5 offsets)."""
        return (self.y_max - np.asarray(y)) / self.res - 0.5, (np.asarray(x) - self.x_min) / self.res - 0.5

    def sample(self, x, y, order=1):
        """Heights at local points; only the window of the grid they cover is read (memory-mapped
        grids stay on disk)."""
        from scipy.ndimage import map_coordinates
        x = np.atleast_1d(np.asarray(x, np.float64))
        y = np.atleast_1d(np.asarray(y, np.float64))
        r, c = self.rc(x, y)
        a = self.a if self.a.ndim == 2 else self.a[0]
        if isinstance(a, np.memmap) or a.base is not None and isinstance(a.base, np.memmap):
            h, w = a.shape
            r0 = int(np.clip(np.floor(np.nanmin(r)) - 2, 0, h - 1)) if len(r) else 0
            r1 = int(np.clip(np.ceil(np.nanmax(r)) + 3, 1, h)) if len(r) else 1
            c0 = int(np.clip(np.floor(np.nanmin(c)) - 2, 0, w - 1)) if len(c) else 0
            c1 = int(np.clip(np.ceil(np.nanmax(c)) + 3, 1, w)) if len(c) else 1
            sub = np.asarray(a[r0:r1, c0:c1])
            return map_coordinates(sub, [r - r0, c - c0], order=order, mode="nearest")
        return map_coordinates(a, [r, c], order=order, mode="nearest")

    def window(self, x0, y0, x1, y1, pad=0):
        """(sub-grid, row0, col0) covering the local rectangle, `pad` cells around it."""
        h, w = self.shape
        c0 = max(int(np.floor((x0 - self.x_min) / self.res)) - pad, 0)
        c1 = min(int(np.ceil((x1 - self.x_min) / self.res)) + pad, w)
        r0 = max(int(np.floor((self.y_max - y1) / self.res)) - pad, 0)
        r1 = min(int(np.ceil((self.y_max - y0) / self.res)) + pad, h)
        a = self.a if self.a.ndim == 2 else self.a[0]
        return Grid(np.asarray(a[r0:r1, c0:c1]), self.x_min + c0 * self.res, self.y_max - r0 * self.res,
                    self.res), r0, c0
