import xarray as xr
import geopandas as gpd
import pandas as pd


nc = xr.open_dataset("synthetic_valley_truth.nc")
"""
for i in range(1, 6):
    da = nc[f"bottom_layer{i}"]
    da.rio.write_crs("EPSG:26911", inplace=True)
    da.rio.to_raster(f"bottom_{i}.tif")

for i in range(1, 6):
    da = nc[f"k1_layer{i}"]
    da.rio.write_crs("EPSG:26911", inplace=True)
    da.rio.to_raster(f"hk_{i}.tif")

da = nc["top_layer1"]
da.rio.write_crs("EPSG:26911", inplace=True)
da.rio.to_raster("mv_dem.tif")
"""
da = nc["izone5"]
da.rio.write_crs("EPSG:26911", inplace=True)
da.rio.to_raster("izone_x.tif")
print('break')

x = pd.read_parquet("temporal_data_annual.parquet")
print('break')