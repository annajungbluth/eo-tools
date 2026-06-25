#!/home/users/annaju/miniforge3/envs/jasmin-env/bin/python
import argparse
import pathlib

import fsspec
import os
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from process_utils import (
    GOES_WAVELENGTHS,
    encode_and_clip,
    get_abi_lat_lon,
    get_satellite_viewing_angles,
    get_sza_and_azi,
)
from pyproj import Proj

GOES_EAST_PROJ4 = "+proj=geos +lon_0=-75 +h=35786023 +x_0=0 +y_0=0 +sweep=x +datum=WGS84 +units=m +no_defs"

goes_satellite_height = 35786023  # in meters
goes_satellite_longitude = -75  # in degrees
goes_satellite_latitude = 0  # in degrees


def reprocess_goes(ds, time, chunksize, **encoding_kwargs):
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

    def _encode_categorical_var(
        var: xr.DataArray,
        min_value: int,
        max_value: int,
        target_dtype,
        chunksizes,
    ) -> xr.DataArray:
        """Encode categorical/integer flags as compact integers (no scale/offset)."""
        da = _prepare_aux_var(var, np.float32)
        fill_value = np.iinfo(target_dtype).max

        da = da.where(np.isfinite(da))
        da = da.clip(min_value, max_value).round()
        da = da.fillna(fill_value).astype(target_dtype)
        da = da.reset_encoding()

        da.attrs.update(valid_range=[target_dtype(min_value), target_dtype(max_value)])
        da.encoding.update(
            dict(dtype=target_dtype.__name__, _FillValue=target_dtype(fill_value))
        )

        chunks = [
            chunksizes[dim] if dim in chunksizes else da[dim].size for dim in da.dims
        ]
        da.encoding.update(dict(chunksizes=chunks, preferred_chunks=chunksizes))
        da.encoding.update(encoding_kwargs)

        for enc_key in list(da.encoding):
            _ = da.attrs.pop(enc_key, None)

        return da

    # Load and stack data vars and apply quality flags
    data = xr.concat(
        (ds[var] for var in GOES_WAVELENGTHS.keys()), dim="channel"
    ).assign_coords(channel=("channel", list(GOES_WAVELENGTHS.keys())))

    new_ds = data.to_dataset(name="data").assign_attrs(ds.attrs)
    new_ds = new_ds.assign_coords(t=time)
    new_ds["data"] = encode_and_clip(
        new_ds.data,
        0,
        400,
        np.uint16,
        {"x": chunksize, "y": chunksize},
        **encoding_kwargs,
    )

    lats, lons = get_abi_lat_lon(ds)

    new_ds["latitude"] = (("y", "x"), lats.astype(np.float32))
    new_ds["latitude"] = encode_and_clip(
        new_ds.latitude,
        -90,
        90,
        np.uint16,
        {"x": chunksize, "y": chunksize},
        **encoding_kwargs,
    )

    new_ds["longitude"] = (("y", "x"), lons.astype(np.float32))
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
        sat_lat=goes_satellite_latitude,
        sat_lon=goes_satellite_longitude,
        sat_alt=goes_satellite_height / 1e3,  # convert to km
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
    # Reprocess additional geophysical variables.
    float_var_specs = {
        "height": (0, 20000),
        "optical_depth": (0, 160),
        "temperature": (180, 340),
        "pressure": (0, 1100),
        "particle_size": (0, 160)
    }
    for var_name, (vmin, vmax) in float_var_specs.items():
        new_ds[var_name] = _prepare_aux_var(ds[var_name], np.float32)
        new_ds[var_name] = encode_and_clip(
            new_ds[var_name],
            vmin,
            vmax,
            np.uint16,
            {"x": chunksize, "y": chunksize},
            **encoding_kwargs,
        )

    chunk_map = {"x": chunksize, "y": chunksize}

    int_var_specs = {
        "height_DQF": (0, 3, np.uint8),
        "optical_depth_DQF": (0, 16, np.uint8),
        "mask_binary": (0, 1, np.uint8),
        "mask_advanced": (0, 3, np.uint8),
        "mask_DQF": (0, 6, np.uint8),
        "particle_size_DQF": (0, 16, np.uint8),
        "temperature_DQF": (0, 3, np.uint8),
        "phase": (0, 5, np.uint8),
        "phase_DQF": (0, 63, np.uint8),
        "pressure_DQF": (0, 3, np.uint8),
    }
    for var_name, (vmin, vmax, dtype) in int_var_specs.items():
        new_ds[var_name] = _encode_categorical_var(
            ds[var_name], vmin, vmax, dtype, chunk_map
        )

    # drop all coordinates except for ['x', 'y', 't'] and the channel coordinate
    coords_to_drop = [
        coord for coord in new_ds.coords if coord not in ["x", "y", "t", "channel"]
    ]
    new_ds = new_ds.drop_vars(coords_to_drop)

    return new_ds


def get_abi_proj(dataset: xr.Dataset) -> Proj:
    """
    Return a pyproj projection from the information contained within an ABI file
    """
    return Proj(
        proj="geos",
        h=dataset.goes_imager_projection.perspective_point_height,
        lon_0=dataset.goes_imager_projection.longitude_of_projection_origin,
        lat_0=dataset.goes_imager_projection.latitude_of_projection_origin,
        sweep=dataset.goes_imager_projection.sweep_angle_axis,
    )


def get_abi_x_y(
    lat: np.ndarray, lon: np.ndarray, dataset: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the x, y coordinates in the ABI projection for given latitudes and
        longitudes
    """
    p = get_abi_proj(dataset)
    x, y = p(lon, lat)
    return (
        x / dataset.goes_imager_projection.perspective_point_height,
        y / dataset.goes_imager_projection.perspective_point_height,
    )


def get_goes_image(
    file: str,
) -> xr.Dataset:
    """
    Get the GOES image for a given timestamp.

    Args:
        file (str): The path to the GOES image.

    Returns:
        xr.Dataset: The GOES dataset for the specified timestamp.
    """
    # Create filesystem object inside worker process to avoid fork-safety issues
    fs = fsspec.filesystem("s3", anon=True)
    fsspec_caching = {
        "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
        "block_size": 8
        * 1024
        * 1024,  # size in bytes per block, adjust depends on the file size but the recommended size is in the MB}
    }
    ds = xr.open_dataset(fs.open(file, **fsspec_caching), engine="h5netcdf")
    return ds


def get_goes_patch(
    lat: float, lon: float, dataset: xr.Dataset, patch_size: int
) -> xr.Dataset:
    """
    Get a patch of GOES data centered around a given latitude and longitude.
    """
    x, y = get_abi_x_y(np.array([lat]), np.array([lon]), dataset)
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
        "--num", type=int, required=True, help="The row to process from the GOES file"
    )
    parser.add_argument(
        "--GOES_file",
        type=str,
        default="./files/matched-filtered-goes-east-[2023-2025]-with-additional-variables.csv",
        help="Path to the file containing all the files to process",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of the patch to extract from the GOES dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/users/annaju/data/esl2026/goes/",
        help="Path to save the processed GOES patches",
    )
    parser.add_argument(
        "--cyclone",
        type=bool,
        default=True,
        help="Whether to add the storm_id to the name",
    )
    args = parser.parse_args()

    # Load the GOES dataset
    logger.info(f"Loading GOES file from {args.GOES_file}...")
    goes_data = pd.read_csv(args.GOES_file)
    goes_data.columns = goes_data.columns.str.lower()

    # Extract row to process
    row = goes_data.iloc[args.num]

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
        save_path.mkdir(parents=True, exist_ok=True)

        lat_str = f"{lat:+.3f}deg"
        lon_str = f"{lon:+.3f}deg"

        if args.cyclone:
            storm_id = row["usa_atcf_id"]
            patch_filename = f"{time_str}_{storm_id}_[{lat_str}_{lon_str}]_{args.patch_size}_patch.nc"
        else:
            patch_filename = (
                f"{time_str}_[{lat_str}_{lon_str}]_{args.patch_size}_patch.nc"
            )

        if args.cyclone:
            save_path = os.path.join(save_path, storm_id)

        os.makedirs(save_path, exist_ok=True)

        save_file_name = os.path.join(save_path, patch_filename)


        abi_file = row["abi_file"]

        logger.info(f"Loading GOES image {abi_file}...")
        ds = get_goes_image(file=abi_file)
        logger.info(f"Extracting patch ...")
        ds_patch = get_goes_patch(lat, lon, ds, args.patch_size)

        # Loading additional variable
        logger.info(f"Adding additional variables:")

        # Add cloud height
        ds_var = get_goes_image(file=row["acha_file"])
        ds_var_patch = get_goes_patch(lat, lon, ds_var, args.patch_size)
        ds_patch["height"] = ds_var_patch["HT"]
        ds_patch["height_DQF"] = ds_var_patch["DQF"]
        logger.info(f"Added cloud height ...")

        # Add cloud optical depth
        ds_var = get_goes_image(file=row["cod_file"])
        ds_var_patch = get_goes_patch(lat, lon, ds_var, args.patch_size)
        ds_patch["optical_depth"] = ds_var_patch["COD"]
        ds_patch["optical_depth_DQF"] = ds_var_patch["DQF"]
        logger.info(f"Added cloud optical depth ...")

        # Add cloud masks
        ds_var = get_goes_image(file=row["acm_file"])
        ds_var_patch = get_goes_patch(lat, lon, ds_var, args.patch_size)
        ds_patch["mask_binary"] = ds_var_patch["BCM"]
        ds_patch["mask_advanced"] = ds_var_patch["ACM"]
        ds_patch["mask_DQF"] = ds_var_patch["DQF"]
        logger.info(f"Added cloud masks ...")

        # Add cloud particle size
        ds_var = get_goes_image(file=row["cps_file"])
        ds_var_patch = get_goes_patch(lat, lon, ds_var, args.patch_size)
        try:
            ds_patch["particle_size"] = ds_var_patch["CPS"]
        except KeyError:
            logger.warning(
                f"CPS variable not found in {row['cps_file']}. Trying 'PSD' variable instead..."
            )
            ds_patch["particle_size"] = ds_var_patch["PSD"]

        ds_patch["particle_size_DQF"] = ds_var_patch["DQF"]
        logger.info(f"Added cloud particle size ...")

        # Add cloud top temperature
        ds_var = get_goes_image(file=row["acht_file"])
        ds_var_patch = get_goes_patch(lat, lon, ds_var, args.patch_size)
        ds_patch["temperature"] = ds_var_patch["TEMP"]
        ds_patch["temperature_DQF"] = ds_var_patch["DQF"]
        logger.info(f"Added cloud top temperature ...")

        # Add cloud phase
        ds_var = get_goes_image(file=row["actp_file"])
        ds_var_patch = get_goes_patch(lat, lon, ds_var, args.patch_size)
        ds_patch["phase"] = ds_var_patch["Phase"]
        ds_patch["phase_DQF"] = ds_var_patch["DQF"]
        logger.info(f"Added cloud phase ...")

        # Add cloud pressure
        ds_var = get_goes_image(file=row["achp_file"])
        ds_var_patch = get_goes_patch(lat, lon, ds_var, args.patch_size)
        ds_patch["pressure"] = ds_var_patch["PRES"]
        ds_patch["pressure_DQF"] = ds_var_patch["DQF"]
        logger.info(f"Added cloud pressure ...")

        # Reprocess file:
        logger.info(f"Reprocessing patch ...")
        new_ds = reprocess_goes(
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
