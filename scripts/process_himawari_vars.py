#!/home/users/annaju/miniforge3/envs/jasmin-env/bin/python
import argparse
import ast
import os
import pathlib
import tempfile

import fsspec
import goes2go
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from google.cloud import storage
from loguru import logger
from process_utils import (
    HIMAWARI_WAVELENGTHS,
    encode_and_clip,
    get_abi_lat_lon,
    get_satellite_viewing_angles,
    get_sza_and_azi,
)
from pyproj import Proj
from satpy import Scene
from scipy.interpolate import make_splrep
from tqdm import tqdm

fs = fsspec.filesystem("s3", anon=True)
fsspec_caching = {
    "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
    "block_size": 100
    * 1024
    * 1024,  # size in bytes per block, adjust depends on the file size but the recommended size is in the MB}
}


def reprocess_himawari(ds, time, chunksize, **encoding_kwargs):
    def _prepare_aux_var(var: xr.DataArray, dtype) -> xr.DataArray:
        """Normalize auxiliary variable dims and dtype before assigning to output dataset."""
        da = var.astype(dtype)

        # Avoid merge conflicts when auxiliary variables carry their own geolocation coords.
        drop_coord_names = [
            name
            for name in ("latitude", "longitude", "Latitude", "Longitude")
            if name in da.coords
        ]
        if drop_coord_names:
            da = da.reset_coords(names=drop_coord_names, drop=True)

        rename_map = {}
        for old_dim, new_dim in (
            ("Rows", "y"),
            ("Columns", "x"),
            ("rows", "y"),
            ("columns", "x"),
            ("row", "y"),
            ("col", "x"),
        ):
            if old_dim in da.dims and new_dim not in da.dims:
                rename_map[old_dim] = new_dim

        if rename_map:
            da = da.rename(rename_map)

        squeeze_dims = [
            dim
            for dim in da.dims
            if dim not in ("y", "x") and da.sizes.get(dim, 0) == 1
        ]
        if squeeze_dims:
            da = da.squeeze(dim=squeeze_dims, drop=True)

        return da

    # Load and stack data vars and apply quality flags
    data = xr.concat(
        (ds[var] for var in HIMAWARI_WAVELENGTHS.keys()), dim="channel"
    ).assign_coords(channel=("channel", list(HIMAWARI_WAVELENGTHS.keys())))

    new_ds = (
        data.to_dataset(name="data")
        .assign_attrs(ds.attrs)
        .drop_vars(["latitude", "longitude"])
    )
    new_ds = new_ds.assign_coords(t=time)
    new_ds["data"] = encode_and_clip(
        new_ds.data,
        0,
        400,
        np.uint16,
        {"x": chunksize, "y": chunksize},
        **encoding_kwargs,
    )

    new_ds["latitude"] = (("y", "x"), ds.latitude.values.astype(np.float32))
    new_ds["latitude"] = encode_and_clip(
        new_ds.latitude,
        -90,
        90,
        np.uint16,
        {"x": chunksize, "y": chunksize},
        **encoding_kwargs,
    )

    new_ds["longitude"] = (("y", "x"), ds.longitude.values.astype(np.float32))
    new_ds["longitude"] = encode_and_clip(
        new_ds.longitude,
        -180,
        180,
        np.uint16,
        {"x": chunksize, "y": chunksize},
        **encoding_kwargs,
    )

    zenith, azimuth = get_satellite_viewing_angles(
        lat=new_ds.latitude,
        lon=new_ds.longitude,
        sat_lat=ast.literal_eval(ds.B01.orbital_parameters)["projection_latitude"],
        sat_lon=ast.literal_eval(ds.B01.orbital_parameters)["projection_longitude"],
        sat_alt=ast.literal_eval(ds.B01.orbital_parameters)["projection_altitude"]
        / 1e3,  # convert to km
    )
    new_ds["sat_angle"] = (
        ("angle", "y", "x"),
        np.stack([zenith.astype(np.float32), azimuth.astype(np.float32)], axis=0),
    )
    new_ds["sat_angle"] = encode_and_clip(
        new_ds.sat_angle,
        0,
        360,
        np.uint16,
        {"x": chunksize, "y": chunksize},
        **encoding_kwargs,
    )

    time = pd.Timestamp(time).to_pydatetime()
    zenith, azimuth = get_sza_and_azi(
        date=time, lat=new_ds.latitude.values, lon=new_ds.longitude.values
    )
    new_ds["solar_angle"] = (
        ("angle", "y", "x"),
        np.stack([zenith.astype(np.float32), azimuth.astype(np.float32)], axis=0),
    )
    new_ds["solar_angle"] = encode_and_clip(
        new_ds.solar_angle,
        0,
        360,
        np.uint16,
        {"x": chunksize, "y": chunksize},
        **encoding_kwargs,
    )

    # Reprocess additional variables without encoding and clipping
    new_ds["height"] = _prepare_aux_var(ds.height, np.float32)
    new_ds["height_DQF"] = _prepare_aux_var(ds.height_DQF, np.int8)

    new_ds["optical_depth"] = _prepare_aux_var(ds.optical_depth, np.float32)

    new_ds["mask_binary"] = _prepare_aux_var(ds.mask_binary, np.int8)
    new_ds["mask_advanced"] = _prepare_aux_var(ds.mask_advanced, np.int8)
    new_ds["mask_DQF"] = _prepare_aux_var(ds.mask_DQF, np.int8)

    new_ds["temperature"] = _prepare_aux_var(ds.temperature, np.float32)

    new_ds["phase"] = _prepare_aux_var(ds.phase, np.float32)
    new_ds["phase_DQF"] = _prepare_aux_var(ds.phase_DQF, np.int8)

    new_ds["pressure"] = _prepare_aux_var(ds.pressure, np.float32)

    new_ds["type"] = _prepare_aux_var(ds.type, np.int8)

    # drop all coordinates except for ['x', 'y', 't'] and the channel coordinate

    coords_to_drop = [
        coord for coord in new_ds.coords if coord not in ["x", "y", "t", "channel"]
    ]
    new_ds = new_ds.drop_vars(coords_to_drop)

    return new_ds


def get_ahi_proj(sat_height: float, sat_lon_0: float, sat_lat_0: float) -> Proj:
    """
    Return a pyproj projection from the information contained within an AHI file
    """
    return Proj(
        proj="geos",
        h=sat_height,
        lon_0=sat_lon_0,
        lat_0=sat_lat_0,
    )


def get_ahi_x_y(
    lat: np.ndarray,
    lon: np.ndarray,
    sat_height: float = None,
    sat_lon_0: float = None,
    sat_lat_0: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the x, y coordinates in the AHI projection for given latitudes and
        longitudes
    """
    p = get_ahi_proj(sat_height=sat_height, sat_lon_0=sat_lon_0, sat_lat_0=sat_lat_0)
    x, y = p(lon, lat)
    return (
        x,
        y,
    )


def get_himawari_image(
    files: list[str],
) -> xr.Dataset:
    """
    Get the HIMAWARI image for a given timestamp.

    Args:
        files (str): The path to the HIMAWARI files.

    Returns:
        xr.Dataset: The HIMAWARI dataset for the specified timestamp.
    """

    # Load with satpy
    scn = Scene(
        [f"s3://{f}" for f in files],  # select all files at one time
        reader="ahi_hsd",
        reader_kwargs=dict(
            storage_options={
                "anon": True,
                "default_block_size": 100 * 1024 * 1024,  # 100MB blocks for large files
                "default_cache_type": "readahead",  # Optimize for sequential reading
            }
        ),
    )
    # load available datasets
    scn.load(scn.all_dataset_names())

    # Resample to 2km resolution
    new_scn = scn.resample(scn.coarsest_area(), resampler="native")

    # Convert to xarray
    ds = new_scn.to_xarray()
    return ds


def get_himawari_patch(
    lat: float,
    lon: float,
    dataset: xr.Dataset,
    patch_size: int,
    sat_height: float = None,
    sat_lon_0: float = None,
    sat_lat_0: float = None,
) -> xr.Dataset:
    """
    Get a patch of HIMAWARI data centered around a given latitude and longitude.
    """
    x, y = get_ahi_x_y(
        np.array([lat]),
        np.array([lon]),
        sat_height=sat_height,
        sat_lon_0=sat_lon_0,
        sat_lat_0=sat_lat_0,
    )
    x_dif = dataset.x.diff("x").values[0]
    y_dif = dataset.y.diff("y").values[0]
    return dataset.sel(
        x=slice(
            x[0] - (abs(x_dif) * patch_size / 2), x[0] + (abs(x_dif) * patch_size / 2)
        ),
        y=slice(
            y[0] + (abs(y_dif) * patch_size / 2), y[0] - (abs(y_dif) * patch_size / 2)
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="The row to process from the HIMAWARI file",
    )
    parser.add_argument(
        "--HIMAWARI-file",
        type=str,
        required=True,
        help="Path to the file containing all the files to process",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the processed HIMAWARI patches",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of the patch to extract from the HIMAWARI dataset",
    )
    parser.add_argument(
        "--cyclone",
        type=bool,
        default=False,
        help="Whether to add the storm_id to the name",
    )
    args = parser.parse_args()

    # Load the HIMAWARI dataset
    logger.info(f"Loading HIMAWARI file from {args.HIMAWARI_file}...")
    himawari_data = pd.read_csv(args.HIMAWARI_file)
    himawari_data.columns = himawari_data.columns.str.lower()

    # Extract row to process
    row = himawari_data.iloc[args.num]

    if args.cyclone:
        logger.info(f"Processing HIMAWARI cyclone data for row {args.num} ...")
    else:
        logger.info(f"Processing HIMAWARI cloud data for row {args.num} ...")

    if "start" in row.keys():
        row["date"] = pd.to_datetime(row["start"])

    if "cloud3d:storm_id" in row.keys():
        row["sid"] = row["cloud3d:storm_id"]

    if row["all_available"]:
        logger.info(f"Processing row {args.num} with date {row['date']} ...")

        lat = row["lat"]
        lon = row["lon"]
        time = pd.to_datetime(row["date"])
        time_str = pd.to_datetime(row["date"]).strftime("%Y%m%d%H%M%S")

        # Create output directory if it doesn't exist
        save_path = pathlib.Path(args.save_path)
        # save_path = pathlib.Path(f"./{time_str}")
        os.makedirs(save_path, exist_ok=True)

        lat_str = f"{lat:+.3f}deg"
        lon_str = f"{lon:+.3f}deg"

        if args.cyclone:
            storm_id = row["usa_atcf_id"]
            patch_filename = f"{time_str}_{storm_id}_[{lat_str}_{lon_str}]_{args.patch_size}_patch.nc"
        else:
            patch_filename = (
                f"{time_str}_[{lat_str}_{lon_str}]_{args.patch_size}_patch.nc"
            )
        save_file_name = save_path / patch_filename

        ahi_files = ast.literal_eval(row["ahi_file"])

        # logger.info(f"Loading HIMAWARI image {ahi_file}...")
        ds = get_himawari_image(files=ahi_files)

        sat_height = ast.literal_eval(ds.B01.orbital_parameters).get(
            "projection_altitude"
        )
        sat_lon_0 = ast.literal_eval(ds.B01.orbital_parameters).get(
            "projection_longitude"
        )
        sat_lat_0 = ast.literal_eval(ds.B01.orbital_parameters).get(
            "projection_latitude"
        )

        logger.info(f"Extracting patch ...")
        ds_patch = get_himawari_patch(
            lat,
            lon,
            ds,
            args.patch_size,
            sat_height=sat_height,
            sat_lon_0=sat_lon_0,
            sat_lat_0=sat_lat_0,
        )

        # Loading additional variable
        logger.info(f"Adding additional variables:")

        # Add cloud height
        ds_var = xr.open_dataset(
            fs.open(row["height_file"], **fsspec_caching), engine="h5netcdf"
        )
        if "x" not in ds_var.coords or "y" not in ds_var.coords:
            ds_var = ds_var.assign_coords(
                {
                    "x": ("Columns", ds.x.values),
                    "y": ("Rows", ds.y.values),
                }
            )
            ds_var = ds_var.swap_dims({"Columns": "x", "Rows": "y"})
        ds_var_patch = get_himawari_patch(
            lat,
            lon,
            ds_var,
            args.patch_size,
            sat_height=sat_height,
            sat_lon_0=sat_lon_0,
            sat_lat_0=sat_lat_0,
        )
        ds_patch["height"] = ds_var_patch["CldTopHght"]
        ds_patch["height_DQF"] = ds_var_patch["CloudHgtQF"]
        logger.info(f"Added cloud height ...")

        # Add cloud optical depth
        ds_patch["optical_depth"] = ds_var_patch["CldOptDpth"]
        logger.info(f"Added cloud optical depth ...")

        # Add cloud top temprature
        ds_patch["temperature"] = ds_var_patch["CldTopTemp"]
        logger.info(f"Added cloud top temperature ...")

        # Add cloud pressure
        ds_patch["pressure"] = ds_var_patch["CldTopPres"]
        logger.info(f"Added cloud pressure ...")

        # Add cloud masks
        ds_var = xr.open_dataset(
            fs.open(row["mask_file"], **fsspec_caching), engine="h5netcdf"
        )
        if "x" not in ds_var.coords or "y" not in ds_var.coords:
            ds_var = ds_var.assign_coords(
                {
                    "x": ("Columns", ds.x.values),
                    "y": ("Rows", ds.y.values),
                }
            )
            ds_var = ds_var.swap_dims({"Columns": "x", "Rows": "y"})
        ds_var_patch = get_himawari_patch(
            lat,
            lon,
            ds_var,
            args.patch_size,
            sat_height=sat_height,
            sat_lon_0=sat_lon_0,
            sat_lat_0=sat_lat_0,
        )
        ds_patch["mask_binary"] = ds_var_patch["CloudMaskBinary"]
        ds_patch["mask_advanced"] = ds_var_patch["CloudMask"]
        ds_patch["mask_DQF"] = ds_var_patch["CloudMaskQualFlag"]
        logger.info(f"Added cloud masks ...")

        # Add cloud phase
        ds_var = xr.open_dataset(
            fs.open(row["phase_file"], **fsspec_caching), engine="h5netcdf"
        )
        if "x" not in ds_var.coords or "y" not in ds_var.coords:
            ds_var = ds_var.assign_coords(
                {
                    "x": ("Columns", ds.x.values),
                    "y": ("Rows", ds.y.values),
                }
            )
            ds_var = ds_var.swap_dims({"Columns": "x", "Rows": "y"})
        ds_var_patch = get_himawari_patch(
            lat,
            lon,
            ds_var,
            args.patch_size,
            sat_height=sat_height,
            sat_lon_0=sat_lon_0,
            sat_lat_0=sat_lat_0,
        )
        ds_patch["phase"] = ds_var_patch["CloudPhase"]
        ds_patch["phase_DQF"] = ds_var_patch["CloudPhaseFlag"]
        logger.info(f"Added cloud phase ...")

        # Add cloud type
        ds_patch["type"] = ds_var_patch["CloudType"]
        logger.info(f"Added cloud type ...")

        # Add cloud particle size
        # NOTE: Cloud particle size is not available...

        # Reprocess file:
        logger.info(f"Reprocessing patch ...")
        new_ds = reprocess_himawari(
            ds=ds_patch,
            time=pd.to_datetime(row["date"]),
            chunksize=64,
            zlib=True,
            shuffle=True,
            complevel=5,
        )  # default in the reprocessing script

        # Save the patched dataset with the specified encoding
        logger.info(f"Saving patched dataset to {save_file_name} ...")
        new_ds.to_netcdf(save_file_name, engine="netcdf4")

        logger.info("Finished successfully ...")

    else:
        logger.warning(f"Not all files available for {row['date']}. Skipping ...")
