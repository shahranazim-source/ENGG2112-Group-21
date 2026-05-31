import pandas as pd
import numpy as np
import cdsapi
import re
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping
from collections import Counter
from datetime import datetime, timedelta
import xarray as xr
import os

# --- Load flood data ---
floods = pd.read_csv("gfd_qcdatabase_2019_08_01.csv")

rain_floods = floods.loc[
    floods["MainCause"].str.contains("rain", case=False, na=False)
].copy()


# --- Extract first polygon from geometry ---
def first_polygon(s):
    if not isinstance(s, str):
        return None

    s = s.split("</coordinates>")[0]
    s = re.sub(r"<.*?>", "", s)

    points = []

    for p in s.split():
        if "," in p:
            try:
                lon, lat = map(float, p.split(",")[:2])
                points.append((lon, lat))
            except Exception:
                continue

    if len(points) < 3:
        return None

    try:
        poly = Polygon(points)

        # Fix invalid polygons
        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            return None

        return poly

    except Exception:
        return None


rain_floods["Polygons"] = rain_floods["geometry"].apply(first_polygon)


# --- Soil type lookup ---
def get_soil(geom, raster_path):

    if geom is None:
        return np.nan

    try:
        with rasterio.open(raster_path) as src:

            out_image, _ = mask(
                src,
                [mapping(geom)],
                crop=True
            )

            values = out_image[0].flatten()

            if src.nodata is not None:
                values = values[values != src.nodata]

            values = values[~np.isnan(values)]

            if len(values) == 0:
                return np.nan

            values = np.round(values).astype(int)

            return Counter(values).most_common(1)[0][0]

    except Exception as e:
        print(f"Soil error: {e}")
        return np.nan


rain_floods["Soil"] = rain_floods["Polygons"].apply(
    lambda g: get_soil(g, "zobler106.asc")
)


# --- Parse date ---
rain_floods["Began"] = pd.to_datetime(
    rain_floods["Began"],
    format="%m/%d/%Y",
    errors="coerce"
)

# --- CDS client ---
c = cdsapi.Client()


# --- ERA5 rainfall function ---
def get_rainfall(geom, date):

    if geom is None or pd.isna(date):
        return (np.nan, np.nan, np.nan, np.nan)

    try:

        # Fix geometry if invalid
        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty:
            return (np.nan, np.nan, np.nan, np.nan)

        # Bounds
        minx, miny, maxx, maxy = geom.bounds

        # Add padding so ERA5 has at least one grid cell
        pad = 0.25

        north = maxy + pad
        south = miny - pad
        west = minx - pad
        east = maxx + pad

        # ERA5 area format = [N, W, S, E]
        area = [north, west, south, east]

        # Safety check
        if north <= south or east <= west:
            print("Invalid area:", area)
            return (np.nan, np.nan, np.nan, np.nan)

        # Previous 3 days
        dates = [(date - timedelta(days=i)) for i in range(3)]

        years = sorted(set(d.strftime("%Y") for d in dates))
        months = sorted(set(d.strftime("%m") for d in dates))
        days = sorted(set(d.strftime("%d") for d in dates))

        request = {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "variable": [
                "total_precipitation",
                "runoff"
            ],
            "year": years,
            "month": months,
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
        }

        tmp_file = "temp_era5.nc"

        # Download ERA5
        try:

            c.retrieve(
                "reanalysis-era5-single-levels",
                request,
                tmp_file
            )

        except Exception as e:
            print(f"ERA5 download failed: {e}")
            return (np.nan, np.nan, np.nan, np.nan)

        # Open dataset
        ds = xr.open_dataset(tmp_file)

        print("Variables in dataset:", list(ds.data_vars))

        # Total precipitation
        tp = ds["tp"].values.flatten()

        # Find runoff variable safely
        runoff_candidates = [
            v for v in ds.data_vars
            if "runoff" in v.lower() or v.lower() == "ro"
        ]

        if len(runoff_candidates) == 0:
            print("No runoff variable found")
            ds.close()

            if os.path.exists(tmp_file):
                os.remove(tmp_file)

            return (
                float(np.nanmean(tp)),
                float(np.nanstd(tp)),
                np.nan,
                np.nan
            )

        ro_var = runoff_candidates[0]
        ro = ds[ro_var].values.flatten()

        ds.close()

        # Delete temp file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        return (
            float(np.nanmean(tp)),
            float(np.nanstd(tp)),
            float(np.nanmean(ro)),
            float(np.nanstd(ro)),
        )

    except Exception as e:

        print(f"Rainfall processing error: {e}")

        if os.path.exists("temp_era5.nc"):
            os.remove("temp_era5.nc")

        return (np.nan, np.nan, np.nan, np.nan)


# --- Apply rainfall extraction ---
rain_floods[
    [
        "hourly_precip_mean",
        "hourly_precip_std",
        "hourly_runoff_mean",
        "hourly_runoff_std",
    ]
] = rain_floods.apply(
    lambda x: get_rainfall(
        x.Polygons,
        x.Began
    ),
    axis=1,
    result_type="expand"
)


# --- Save output ---
rain_floods.to_csv("Negative_set.csv", index=False)

print("Done.")
