"""
Direct-S3 acquisition of Sentinel-2 L2A imagery for the SDB dashboard.

Why direct S3 instead of STAC: this VPS's network blocks every STAC API we
tested (Microsoft Planetary Computer, Element84 Earth Search, Copernicus Data
Space). The only reachable Sentinel-2 source is the public
sentinel-cogs.s3.us-west-2.amazonaws.com bucket. We list it anonymously
and read window-of-COG via rasterio's VSI HTTP driver.

Workflow:
   1. lat,lon → MGRS 100-km grid square (e.g. (35,"S","NB"))
   2. List S3 prefix sentinel-s2-l2a-cogs/{zone}/{band}/{grid}/{year}/{month}/
      across the requested date window.
   3. For each candidate scene, fetch tileinfo_metadata.json to read
      cloudyPixelPercentage.
   4. Return the scene with the lowest cloud cover.
   5. download_chip() reads B04/B03/B02 windows centered on (lat,lon)
      and returns a (3, H, W) float array in surface-reflectance units
      ([0, 1] after dividing the int reflectance scale by 10_000).
"""
from __future__ import annotations
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np

S3_BASE = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
S3_KEY_PREFIX = "sentinel-s2-l2a-cogs"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _http_get(url: str, timeout=20, retries=3) -> bytes:
    last_err = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sdb-dash/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"GET failed after {retries} tries: {url}  ({last_err})")


# ----------------------------------------------------------------------
# MGRS conversion
# ----------------------------------------------------------------------
def lat_lon_to_mgrs_grid(lat: float, lon: float) -> Tuple[int, str, str]:
    """Return (utm_zone, latitude_band, 100km_grid) e.g. (35, 'S', 'NB')."""
    import mgrs
    m = mgrs.MGRS()
    s = m.toMGRS(lat, lon, MGRSPrecision=0)         # e.g. "35SNB" + numerics
    # Robust parse: leading 1-2 digit zone, 1 letter band, 2 letter grid
    mt = re.match(r"^(\d{1,2})([A-HJ-NP-Z])([A-HJ-NP-Z]{2})", s)
    if not mt:
        raise ValueError(f"Unexpected MGRS string: {s}")
    return int(mt.group(1)), mt.group(2), mt.group(3)


# ----------------------------------------------------------------------
# Scene discovery
# ----------------------------------------------------------------------
@dataclass
class S2Scene:
    scene_id: str
    s3_prefix: str          # path inside the bucket, ends with /
    acq_date: str           # YYYY-MM-DD
    cloud_cover: float      # 0..100
    data_coverage: float    # 0..100
    utm_zone: int
    lat_band: str
    grid_id: str
    band_urls: dict         # {"B02": url, "B03": url, "B04": url}


def _months_in_range(start: date, end: date):
    """Yield (year, month_no_leading_zero) tuples covering the range."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        if m == 12: y, m = y + 1, 1
        else: m += 1


def _list_scene_prefixes_in_month(zone: int, band: str, grid: str,
                                  year: int, month: int) -> List[str]:
    """Return list of scene-folder S3 prefixes (each ending with /)."""
    prefix = f"{S3_KEY_PREFIX}/{zone}/{band}/{grid}/{year}/{month}/"
    out = []
    cont = None
    while True:
        params = {"list-type": "2", "prefix": prefix,
                  "delimiter": "/", "max-keys": "1000"}
        if cont: params["continuation-token"] = cont
        url = f"{S3_BASE}/?" + urllib.parse.urlencode(params)
        body = _http_get(url, timeout=20).decode("utf-8")
        root = ET.fromstring(body)
        for cp in root.findall("s3:CommonPrefixes", S3_NS):
            p = cp.findtext("s3:Prefix", default="", namespaces=S3_NS)
            if p:
                out.append(p)
        is_trunc = root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) == "true"
        cont = root.findtext("s3:NextContinuationToken", default="", namespaces=S3_NS)
        if not is_trunc or not cont:
            break
    return out


def _scene_metadata(prefix: str, zone: int, band: str, grid: str) -> Optional[S2Scene]:
    """Fetch tileinfo_metadata.json for a scene prefix; return S2Scene or None."""
    info_url = f"{S3_BASE}/{prefix}tileinfo_metadata.json"
    try:
        body = _http_get(info_url, timeout=15).decode("utf-8")
        info = json.loads(body)
    except Exception:
        return None
    scene_id = prefix.rstrip("/").rsplit("/", 1)[-1]   # e.g. S2A_35SNB_20240104_0_L2A
    # Date parse from scene_id (YYYYMMDD)
    md = re.search(r"_(\d{8})_", scene_id)
    acq = md.group(1) if md else ""
    acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}" if len(acq) == 8 else ""
    band_urls = {b: f"{S3_BASE}/{prefix}{b}.tif" for b in ("B02", "B03", "B04")}
    return S2Scene(
        scene_id=scene_id,
        s3_prefix=prefix,
        acq_date=acq_iso,
        cloud_cover=float(info.get("cloudyPixelPercentage", 100.0)),
        data_coverage=float(info.get("dataCoveragePercentage", 0.0)),
        utm_zone=zone, lat_band=band, grid_id=grid,
        band_urls=band_urls,
    )


def find_best_scene(lat: float, lon: float,
                    start: date, end: date,
                    max_cloud: float = 100.0,
                    min_data_coverage: float = 30.0,
                    progress_cb=None,
                    n_workers: int = 16,
                    early_exit_below: float = 1.0) -> Optional[S2Scene]:
    """Search the S3 bucket for the lowest-cloud Sentinel-2 L2A scene covering
    the MGRS 100-km grid square containing (lat,lon), restricted to [start, end]
    and cloud_cover <= max_cloud and data_coverage >= min_data_coverage.

    Optimized for the slow VPS network: tileinfo fetches are parallelized
    across `n_workers` threads, and the search exits early once any scene with
    cloud_cover ≤ early_exit_below is found (since you cannot beat ~0%).

    Returns the best S2Scene or None if nothing matched.
    """
    import concurrent.futures as cf

    zone, band, grid = lat_lon_to_mgrs_grid(lat, lon)

    # 1. enumerate candidate prefixes across all months (cheap)
    all_prefixes = []
    for y, m in _months_in_range(start, end):
        if progress_cb: progress_cb(f"listing {zone}{band}{grid}  {y}-{m:02d}")
        try:
            all_prefixes.extend(_list_scene_prefixes_in_month(zone, band, grid, y, m))
        except Exception as e:
            if progress_cb: progress_cb(f"  list error {y}-{m:02d}: {e}")
    if progress_cb:
        progress_cb(f"found {len(all_prefixes)} candidate scenes; fetching metadata in parallel "
                    f"(workers={n_workers})")
    if not all_prefixes:
        return None

    # 2. fetch metadata in parallel; track best, support early exit
    best: Optional[S2Scene] = None
    matched = 0
    seen = 0

    def _fetch(p):
        return _scene_metadata(p, zone, band, grid)

    with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_fetch, p): p for p in all_prefixes}
        for fut in cf.as_completed(futures):
            seen += 1
            meta = fut.result()
            if meta is None:
                if progress_cb and seen % 10 == 0:
                    progress_cb(f"  {seen}/{len(all_prefixes)} metas fetched")
                continue
            if meta.cloud_cover > max_cloud:        continue
            if meta.data_coverage < min_data_coverage: continue
            matched += 1
            if best is None or meta.cloud_cover < best.cloud_cover:
                best = meta
                if progress_cb:
                    progress_cb(f"  ↑ best so far: {best.scene_id}  "
                                f"cloud={best.cloud_cover:.1f}%  cov={best.data_coverage:.1f}%")
                if best.cloud_cover <= early_exit_below:
                    if progress_cb:
                        progress_cb(f"  early-exit: found cloud_cover≤{early_exit_below}%")
                    for f in futures: f.cancel()
                    break

    if progress_cb:
        progress_cb(f"examined {seen}/{len(all_prefixes)} scenes, {matched} matched filter")
    return best


# ----------------------------------------------------------------------
# COG window read
# ----------------------------------------------------------------------
def _enable_gdal_http():
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")
    os.environ.setdefault("VSI_CACHE", "TRUE")


def download_chip(scene: S2Scene,
                  lat: float, lon: float,
                  size_px: int = 256,
                  resolution_m: float = 10.0,
                  size_m: Optional[float] = None,
                  ) -> Tuple[np.ndarray, dict]:
    """Read a (3, size_px, size_px) window centered on (lat, lon) from B04/B03/B02
    of `scene`. Bands are returned in (R, G, B) order. Values are in surface
    reflectance ([0,1]) after dividing the int reflectance scale by 10 000.

    If `size_m` is given, the window's geographic extent is `size_m × size_m`
    (and resampled to size_px × size_px). Otherwise it defaults to
    `size_px * resolution_m` (i.e. native Sentinel-2 10 m per output pixel).

    Returns (chip_chw_float32, info_dict).
    """
    import rasterio
    from rasterio.windows import from_bounds
    import pyproj

    _enable_gdal_http()
    # UTM CRS for this tile
    utm_epsg = (32600 + scene.utm_zone) if lat >= 0 else (32700 + scene.utm_zone)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}",
                                         always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    extent_m = float(size_m) if size_m else (size_px * resolution_m)
    half = extent_m / 2.0
    bbox = (cx - half, cy - half, cx + half, cy + half)

    band_arrays = []
    src_info = {}
    for b_name in ("B04", "B03", "B02"):                # R, G, B
        url = f"/vsicurl/{scene.band_urls[b_name]}"
        with rasterio.open(url) as src:
            if not src_info:
                src_info = dict(crs=str(src.crs), transform=src.transform,
                                bounds=src.bounds, shape=src.shape)
            window = from_bounds(*bbox, transform=src.transform)
            arr = src.read(1, window=window,
                           out_shape=(size_px, size_px),
                           resampling=rasterio.enums.Resampling.bilinear,
                           boundless=True, fill_value=0)
            band_arrays.append(arr.astype(np.float32) / 10000.0)
    chip = np.stack(band_arrays, axis=0).clip(0.0, 1.0).astype(np.float32)

    info = dict(
        scene=asdict(scene),
        center_lat=lat, center_lon=lon,
        utm_zone=scene.utm_zone, utm_epsg=utm_epsg,
        bbox_utm=bbox, size_px=size_px,
        extent_m=extent_m,
        m_per_pixel=extent_m / float(size_px),
        bands_order="R(B04), G(B03), B(B02)",
        chip_min=float(chip.min()), chip_max=float(chip.max()),
        chip_mean=float(chip.mean()),
    )
    return chip, info


# ----------------------------------------------------------------------
# Pretty bridge for the dashboard
# ----------------------------------------------------------------------
def search_and_download(lat: float, lon: float,
                        days_back: int = 365,
                        max_cloud: float = 10.0,
                        size_px: int = 256,
                        size_m: Optional[float] = None,
                        progress_cb=None,
                        ) -> Tuple[Optional[np.ndarray], Optional[dict], List[str]]:
    """High-level helper used by the dashboard.

    Returns (chip, info, log_lines).
    """
    log = []
    def log_cb(msg):
        log.append(msg)
        if progress_cb: progress_cb(msg)

    log_cb(f"input lat={lat:.4f}, lon={lon:.4f}, days_back={days_back}, max_cloud={max_cloud}%")
    end = date.today()
    start = end - timedelta(days=int(days_back))
    log_cb(f"date window {start.isoformat()} → {end.isoformat()}")

    try:
        scene = find_best_scene(lat, lon, start, end,
                                max_cloud=max_cloud,
                                progress_cb=log_cb)
    except Exception as e:
        log_cb(f"search failed: {e}")
        return None, None, log
    if scene is None:
        log_cb("no scene matched the filter")
        return None, None, log
    log_cb(f"selected scene: {scene.scene_id}  "
           f"date={scene.acq_date}  cloud={scene.cloud_cover:.1f}%  cov={scene.data_coverage:.1f}%")
    log_cb("downloading B04/B03/B02 windows from S3 COGs …")
    try:
        chip, info = download_chip(scene, lat, lon, size_px=size_px, size_m=size_m)
    except Exception as e:
        log_cb(f"download failed: {e}")
        return None, asdict(scene), log
    log_cb(f"chip ready: shape={chip.shape}, mean={chip.mean():.3f}, "
           f"min={chip.min():.3f}, max={chip.max():.3f}")
    return chip, info, log


# ----------------------------------------------------------------------
# CLI for debugging
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--days-back", type=int, default=365)
    ap.add_argument("--max-cloud", type=float, default=10.0)
    args = ap.parse_args()
    chip, info, log = search_and_download(args.lat, args.lon,
                                          args.days_back, args.max_cloud,
                                          progress_cb=print)
    print("---")
    print(json.dumps({k: v for k, v in (info or {}).items() if k != "scene"},
                     indent=2, default=str))
