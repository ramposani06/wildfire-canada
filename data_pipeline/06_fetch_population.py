"""
Compute population within 10km and 25km of each fire, using the Kontur
Population Dataset (a global H3-hexagon population grid derived from
multiple census/settlement sources).
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

RADII_KM = [10, 25]


def fetch_population(fires: pd.DataFrame, kontur_gpkg_path: str) -> pd.DataFrame:
    """
    fires must have columns: UNIQUE_ID, LATITUDE, LONGITUDE.
    kontur_gpkg_path points to the Kontur population GeoPackage for the
    region covering these fires.
    """
    pop_grid = gpd.read_file(kontur_gpkg_path)
    pop_grid_proj = pop_grid.to_crs(epsg=3347)

    results = []
    for fire in fires.itertuples():
        pt = gpd.GeoSeries([Point(fire.LONGITUDE, fire.LATITUDE)], crs="EPSG:4326").to_crs(epsg=3347).iloc[0]
        row = {"UNIQUE_ID": fire.UNIQUE_ID}
        for r_km in RADII_KM:
            buffer = pt.buffer(r_km * 1000)
            nearby = pop_grid_proj[pop_grid_proj.intersects(buffer)]
            row[f"pop_within_{r_km}km"] = nearby["population"].sum() if len(nearby) else 0.0
        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    fires = pd.read_csv("fires_to_fetch.csv")
    result = fetch_population(fires, kontur_gpkg_path="kontur_population_canada.gpkg")
    result.to_csv("population_output.csv", index=False)
    print(f"Fetched population for {len(result):,} fires")
