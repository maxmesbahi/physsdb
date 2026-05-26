"""
Fetch reference bathymetry from NOAA NGDC's DEM Global Mosaic for any bbox.

Why this source: among all bathymetry APIs tested, this is the only one
reachable from the VPS (EMODnet, GEBCO, Copernicus DataSpace, USGS,
opentopodata.org, etc. are all blocked).  The mosaic is a NOAA composite
that uses high-resolution multibeam/BAG surveys where available
(~3 m/px in many U.S. and well-surveyed coastlines) and falls back to
coarser sources (down to ETOPO ~1.85 km) elsewhere.

Returned arrays are **depth in meters, positive below sea level**, with
NaN for land (i.e. wherever the source elevation is positive).
"""
from __future__ import annotations
import urllib.parse, urllib.request, io
from typing import Tuple, Optional
import numpy as np

ENDPOINT = ("https://gis.ngdc.noaa.gov/arcgis/rest/services/"
            "DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage")


def fetch_truth_chip(lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float,
                     size_px: int = 256,
                     timeout: int = 25,
                     retries: int = 3) -> Tuple[Optional[np.ndarray], dict]:
    """Return (depth_m, info). depth_m is a (size_px, size_px) float32 array
    in meters (positive = below sea level). Land pixels become NaN.

    Returns (None, info_with_error) if the fetch fails.
    """
    if not (lat_min < lat_max and lon_min < lon_max):
        return None, {"error": "invalid bbox"}
    params = {
        # ArcGIS ImageServer uses bbox = xmin,ymin,xmax,ymax in bboxSR units
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "bboxSR": "4326",
        "size": f"{size_px},{size_px}",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)

    last_err = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sdb-dash/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
            break
        except Exception as e:
            last_err = e
    else:
        return None, {"error": f"fetch failed after {retries} retries: {last_err}"}

    # parse the GeoTIFF via rasterio (in-memory)
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(blob) as memfile, memfile.open() as src:
            elev = src.read(1).astype(np.float32)
            transform = src.transform
            crs = str(src.crs)
    except Exception as e:
        return None, {"error": f"tiff parse failed: {e}", "blob_bytes": len(blob)}

    # NoData → NaN
    elev = np.where(np.isfinite(elev) & (elev > -9000), elev, np.nan)
    # depth = -elevation where elev<0; land (elev>=0) becomes NaN
    depth = np.where(elev < 0, -elev, np.nan).astype(np.float32)

    finite = depth[np.isfinite(depth)]
    info = {
        "source": "NOAA NGDC DEM Global Mosaic (composite: multibeam/BAG → ETOPO)",
        "url_kind": "ArcGIS ImageServer exportImage",
        "bbox_latlon": (lat_min, lat_max, lon_min, lon_max),
        "size_px": size_px,
        "crs": crs,
        "water_pixels": int(np.isfinite(depth).sum()),
        "total_pixels": int(depth.size),
        "water_fraction": float(np.isfinite(depth).mean()),
        "depth_min_m": float(finite.min()) if finite.size else None,
        "depth_max_m": float(finite.max()) if finite.size else None,
        "depth_mean_m": float(finite.mean()) if finite.size else None,
        "land_fraction": float(np.mean(~np.isfinite(depth))),
        "blob_bytes": len(blob),
    }
    return depth, info


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True,
                    help="lat_min,lat_max,lon_min,lon_max  e.g. 34.715,34.725,33.305,33.315")
    args = ap.parse_args()
    parts = [float(x) for x in args.bbox.split(",")]
    d, info = fetch_truth_chip(*parts)
    print(json.dumps(info, indent=2, default=str))
    if d is not None:
        print(f"depth array shape={d.shape}  finite={int(np.isfinite(d).sum())}/{d.size}")
