"""
Compute distance to the nearest road for each fire, using Statistics
Canada's National Road Network (NRN), fetched/processed one province
at a time.

Processing all 13 provinces' road shapefiles in one pass without
releasing memory exhausted RAM in an earlier run (BC/ON/QC in particular
have very large road networks). Explicit del + gc.collect() after each
province avoids that.
"""
import gc

import geopandas as gpd
import pandas as pd
from shapely.strtree import STRtree
from shapely.geometry import Point

PC_PROVINCE_OVERRIDE = {
    # Parks Canada (PC) fires are resolved to the province the park
    # physically sits in, so they can use that province's road network.
    # Filled in per-park as needed; extend as new PC parks appear in the data.
}


def resolve_pc_province(park_name: str) -> str | None:
    return PC_PROVINCE_OVERRIDE.get(park_name)


def fetch_road_distance(fires: pd.DataFrame, nrn_shapefile_paths: dict[str, str]) -> pd.DataFrame:
    """
    fires must have columns: UNIQUE_ID, SRC_AGENCY, LATITUDE, LONGITUDE,
    and optionally NAT_PARK (for PC province resolution).
    nrn_shapefile_paths maps province code -> path to that province's
    NRN road shapefile.
    """
    results = []

    for province, shapefile_path in nrn_shapefile_paths.items():
        province_fires = fires[fires.SRC_AGENCY == province].copy()
        if province == "PC":
            province_fires["resolved_province"] = province_fires.NAT_PARK.apply(resolve_pc_province)
            # PC fires are handled per resolved_province in a separate pass;
            # skipped here for brevity.
            continue
        if province_fires.empty:
            continue

        roads = gpd.read_file(shapefile_path)
        roads_proj = roads.to_crs(epsg=3347)  # Statistics Canada Lambert, metres

        # Build a spatial index of road geometries for fast nearest-neighbor lookup.
        tree = STRtree(list(roads_proj.geometry))

        pts = gpd.GeoSeries(
            [Point(lon, lat) for lon, lat in zip(province_fires.LONGITUDE, province_fires.LATITUDE)],
            crs="EPSG:4326",
        ).to_crs(epsg=3347)

        for uid, pt in zip(province_fires.UNIQUE_ID, pts):
            nearest_geom = tree.nearest(pt)
            dist_m = pt.distance(nearest_geom)
            results.append({"UNIQUE_ID": uid, "dist_to_road_m": dist_m})

        # Explicit cleanup — large provinces (BC, ON, QC) will exhaust RAM
        # across a 13-province loop without this.
        del roads, roads_proj, tree, pts
        gc.collect()
        print(f"  {province}: done ({len(province_fires):,} fires)")

    return pd.DataFrame(results)


if __name__ == "__main__":
    fires = pd.read_csv("fires_to_fetch.csv")
    nrn_paths = {
        "AB": "nrn_data/AB_roads.shp",
        "BC": "nrn_data/BC_roads.shp",
        # ... one entry per province, matching SRC_AGENCY codes
    }
    result = fetch_road_distance(fires, nrn_paths)
    result.to_csv("road_distance_output.csv", index=False)
    print(f"Fetched road distance for {len(result):,} fires")
