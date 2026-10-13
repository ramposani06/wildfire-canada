"""Look up distance to the nearest road and population within 10 / 25 km for a point,
from a small pre-built grid (automation/data/road_pop_grid.npz). No big shapefiles needed.
Build the grid once with automation/colab_step2_lookup.py.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# the file may sit in automation/data/ or directly in automation/ (both work)
DEFAULT_PATH = next((p for p in (os.path.join(_HERE, "data", "road_pop_grid.npz"),
                                 os.path.join(_HERE, "road_pop_grid.npz")) if os.path.exists(p)),
                    os.path.join(_HERE, "data", "road_pop_grid.npz"))


class RoadPopLookup:
    def __init__(self, path: str = DEFAULT_PATH):
        z = np.load(path)
        self.dist = z["dist_10m"]            # uint16, units of 10 m
        self.pop10 = z["pop10"]              # float32, people within 10 km
        self.pop25 = z["pop25"]
        m = z["meta"]                        # lat0, lat1, lon0, lon1, res_dist, res_pop
        self.lat0, self.lat1, self.lon0, self.lon1, self.res_d, self.res_p = [float(v) for v in m]

    def _idx(self, lat, lon, res, shape):
        if not (self.lat0 <= lat < self.lat1 and self.lon0 <= lon < self.lon1):
            raise ValueError(f"({lat}, {lon}) is outside the lookup grid")
        i = min(int((lat - self.lat0) / res), shape[0] - 1)
        j = min(int((lon - self.lon0) / res), shape[1] - 1)
        return i, j

    def get(self, lat: float, lon: float) -> dict:
        i, j = self._idx(lat, lon, self.res_d, self.dist.shape)
        k, l = self._idx(lat, lon, self.res_p, self.pop10.shape)
        return {
            "dist_to_road_m": float(self.dist[i, j]) * 10.0,
            "pop_within_10km": float(self.pop10[k, l]),
            "pop_within_25km": float(self.pop25[k, l]),
        }

    def get_many(self, lats, lons) -> np.ndarray:
        """Vectorised; returns array (n, 3). Points outside the grid give NaN."""
        lats = np.asarray(lats, float); lons = np.asarray(lons, float)
        out = np.full((len(lats), 3), np.nan)
        ok = (lats >= self.lat0) & (lats < self.lat1) & (lons >= self.lon0) & (lons < self.lon1)
        i = np.clip(((lats[ok] - self.lat0) / self.res_d).astype(int), 0, self.dist.shape[0] - 1)
        j = np.clip(((lons[ok] - self.lon0) / self.res_d).astype(int), 0, self.dist.shape[1] - 1)
        k = np.clip(((lats[ok] - self.lat0) / self.res_p).astype(int), 0, self.pop10.shape[0] - 1)
        l = np.clip(((lons[ok] - self.lon0) / self.res_p).astype(int), 0, self.pop10.shape[1] - 1)
        out[ok, 0] = self.dist[i, j].astype(float) * 10.0
        out[ok, 1] = self.pop10[k, l]
        out[ok, 2] = self.pop25[k, l]
        return out
