#!/home/users/annaju/miniforge3/envs/jasmin-env/bin/python
import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from process_utils import (
    CenterWeightedCropDatasetEditor,
    get_satellite_viewing_angles,
    get_sza_and_azi,
    read_zipped_msg,
)

# MSG wavelengths in nanometers
MSG_WAVELENGTHS = {
    "IR_016": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 1578.4,
        "center_wavelength": 1640.0,
        "max_wavelength": 1696.0,
    },  # 1.64
    "IR_039": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 3638.4,
        "center_wavelength": 3920.0,
        "max_wavelength": 4201.6,
    },  # 3.92,
    "IR_087": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 8540.0,
        "center_wavelength": 8700.0,
        "max_wavelength": 8892.0,
    },  # 8.70,
    "IR_097": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 9548.0,
        "center_wavelength": 9660.0,
        "max_wavelength": 9783.2,
    },  # 9.66,
    "IR_108": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10280.0,
        "center_wavelength": 10800.0,
        "max_wavelength": 11280.0,
    },
    "IR_120": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 11520.0,
        "center_wavelength": 12000.0,
        "max_wavelength": 12440.0,
    },
    "IR_134": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 12680.0,
        "center_wavelength": 13400.0,
        "max_wavelength": 14000.0,
    },
    "VIS006": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 602.0,
        "center_wavelength": 640.0,
        "max_wavelength": 677.0,
    },
    "VIS008": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 782.0,
        "center_wavelength": 810.0,
        "max_wavelength": 838.0,
    },
    "WV_062": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 5854.0,
        "center_wavelength": 6250.0,
        "max_wavelength": 6718.0,
    },
    "WV_073": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 7150.0,
        "center_wavelength": 7350.0,
        "max_wavelength": 7590.0,
    },
}

CHANNELS = [
    "IR_016",
    "IR_039",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
    "VIS006",
    "VIS008",
    "WV_062",
    "WV_073",
]


def encode_and_clip(da, min, max, target_dtype, chunksizes={}, **encoding_kwargs):
    # Drop any existing encoding
    da = da.reset_encoding()

    input_type = da.dtype.type
    da.attrs.update(valid_range=[input_type(min), input_type(max)])
    da.encoding.update(encoding_attrs(da.valid_range, input_type, target_dtype))
    chunks = [chunksizes[dim] if dim in chunksizes else da[dim].size for dim in da.dims]
    da.encoding.update(dict(chunksizes=chunks, preferred_chunks=chunksizes))
    da.encoding.update(encoding_kwargs)
    for enc_key in da.encoding:
        _ = da.attrs.pop(enc_key, None)
    da.data = np.clip(da.data, *da.valid_range)

    return da


def encoding_attrs(valid_range, input_dtype, target_dtype):
    valid_range = np.array(valid_range, dtype=input_dtype)
    range = valid_range.max() - valid_range.min()
    max_value = np.iinfo(target_dtype).max
    min_value = np.iinfo(target_dtype).min

    if np.issubdtype(target_dtype, np.unsignedinteger):
        return dict(
            scale_factor=input_dtype(range / (max_value - 1)),
            add_offset=input_dtype(valid_range.min()),
            _FillValue=max_value,
            dtype=str(target_dtype.__name__),
        )
    elif np.issubdtype(target_dtype, np.integer):
        return dict(
            scale_factor=input_dtype(range / max_value),
            add_offset=input_dtype(valid_range.min()),
            _FillValue=min_value,
            dtype=str(target_dtype.__name__),
        )


def interpolate_na_2d(da, **kwargs):
    out = (
        xr.concat(
            [
                da.interpolate_na(dim="x", use_coordinate=False, limit=1),
                da[:, ::-1].interpolate_na(dim="y", use_coordinate=False, limit=1)[
                    :, ::-1
                ],
            ],
            dim="_mean",
        )
        .mean("_mean")
        .assign_attrs(da.attrs)
    )
    out.encoding.update(**da.encoding)
    return out


def reprocess_msg(ds, time, chunksize, **encoding_kwargs):
    # Load and stack data vars and apply quality flags
    data = xr.concat(
        (ds[var] for var in MSG_WAVELENGTHS.keys()), dim="channel"
    ).assign_coords(channel=("channel", list(MSG_WAVELENGTHS.keys())))

    # Interpolate to fill NaN values if only a few pixelss
    data = interpolate_na_2d(data, limit=1)

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
        sat_lat=ds.msg_seviri_fes_3km.latitude_of_projection_origin,
        sat_lon=ds.msg_seviri_fes_3km.longitude_of_projection_origin,
        sat_alt=ds.msg_seviri_fes_3km.perspective_point_height / 1e3,  # convert to km
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

    if "cloudsat_overpass_mask" in ds.data_vars:
        new_ds["cloudsat_overpass_mask"] = (
            ("y", "x"),
            ds.cloudsat_overpass_mask.values,
        )
        new_ds.cloudsat_overpass_mask.encoding.update(**encoding_kwargs)

    # Reprocess additional variables without encoding and clipping
    new_ds["height"] = xr.DataArray(
        ds.height.data.astype(np.float32), dims=("y", "x"), attrs=ds.height.attrs
    )
    new_ds["optical_depth"] = xr.DataArray(
        ds.optical_depth.data.astype(np.float32),
        dims=("y", "x"),
        attrs=ds.optical_depth.attrs,
    )
    new_ds["temperature"] = xr.DataArray(
        ds.temperature.data.astype(np.float32),
        dims=("y", "x"),
        attrs=ds.temperature.attrs,
    )
    new_ds["pressure"] = xr.DataArray(
        ds.pressure.data.astype(np.float32), dims=("y", "x"), attrs=ds.pressure.attrs
    )
    new_ds["mask_binary"] = xr.DataArray(
        ds.mask_binary.data.astype(np.float32),
        dims=("y", "x"),
        attrs=ds.mask_binary.attrs,
    )
    new_ds["phase"] = xr.DataArray(
        ds.phase.data.astype(np.float32), dims=("y", "x"), attrs=ds.phase.attrs
    )
    new_ds["DQF"] = xr.DataArray(
        ds.DQF.data.astype(np.float32), dims=("y", "x"), attrs=ds.DQF.attrs
    )
    new_ds["type"] = xr.DataArray(
        ds.type.data.astype(np.float32), dims=("y", "x"), attrs=ds.type.attrs
    )
    new_ds["effective_radius"] = xr.DataArray(
        ds.effective_radius.data.astype(np.float32),
        dims=("y", "x"),
        attrs=ds.effective_radius.attrs,
    )
    new_ds["liquid_water_path"] = xr.DataArray(
        ds.liquid_water_path.data.astype(np.float32),
        dims=("y", "x"),
        attrs=ds.liquid_water_path.attrs,
    )

    coords_to_drop = [
        coord for coord in new_ds.coords if coord not in ["x", "y", "t", "channel"]
    ]
    new_ds = new_ds.drop_vars(coords_to_drop)

    return new_ds


def patch_msg(ds, patch_size, fov_radius):
    """
    Load and patch MSG file using Satpy.

    Args:
        ds (xarray.Dataset): MSG dataset.
        patch_size (int): Size of the patch to crop from the dataset.
        fov_radius (float): Field of view radius for cropping.

    Returns:
        xarray.Dataset: Patched dataset.
    """
    # Crop dataset into patch
    crop = CenterWeightedCropDatasetEditor(
        patch_shape=(patch_size, patch_size), fov_radius=fov_radius, satellite="msg"
    )

    result, xmin, ymin = crop(ds)

    if result is None:  # i.e. if no valid patch was found
        logger.warning(f"Could not find valid patch ...")
        return None, None, None

    return result, xmin, ymin


if __name__ == "__main__":
    # Add argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num", type=int, required=True, help="The row to process from the MSG file"
    )
    parser.add_argument(
        "--MSG-file",
        type=str,
        default="/home/users/annaju/eo-tools/scripts/files/pretraining-test-msg-[2019]-with-additional-variables-shuffled.csv",
        help="Path to the file containing all the files to process",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/users/annaju/data/clouds/msg",
        help="Path to save the processed MSG patches",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of the patch to extract from the MSG dataset",
    )
    parser.add_argument(
        "--fov_radius",
        type=float,
        default=0.6,
        help="Field of view radius for cropping",
    )
    args = parser.parse_args()

    # Load the MSG dataset
    logger.info(f"Loading MSG file from {args.MSG_file}...")
    msg_data = pd.read_csv(args.MSG_file)

    # Extract row to process
    row = msg_data.iloc[args.num]

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    row["date"] = pd.to_datetime(row["date"], format="%Y-%m-%d %H:%M:%S")

    print(row["cloud_file"])
    print(row["seviri_file"])

    if row["all_available"]:
        logger.info(f"Processing row {args.num} with date {row['date']} ...")

        ds = read_zipped_msg(row["seviri_file"])

        logger.info(f"Adding additional variables:")

        ds_var = xr.open_dataset(row["cloud_file"], decode_times=False)

        ds["height"] = xr.DataArray(
            ds_var["cth_corrected"].values,
            dims=["y", "x"],
            attrs=ds_var["cth_corrected"].attrs,
        )
        logger.info(f"Added cloud height ...")

        ds["optical_depth"] = xr.DataArray(
            ds_var["cot"].values, dims=["y", "x"], attrs=ds_var["cot"].attrs
        )
        logger.info(f"Added cloud optical depth ...")

        ds["temperature"] = xr.DataArray(
            ds_var["ctt_corrected"].values,
            dims=["y", "x"],
            attrs=ds_var["ctt_corrected"].attrs,
        )
        logger.info(f"Added cloud temperature ...")

        ds["pressure"] = xr.DataArray(
            ds_var["ctp_corrected"].values,
            dims=["y", "x"],
            attrs=ds_var["ctp_corrected"].attrs,
        )
        logger.info(f"Added cloud pressure ...")

        ds["mask_binary"] = xr.DataArray(
            ds_var["cldmask"].sel(views=0).values,
            dims=["y", "x"],
            attrs=ds_var["cldmask"].attrs,
        )
        logger.info(f"Added cloud mask ...")

        ds["phase"] = xr.DataArray(
            ds_var["phase"].values, dims=["y", "x"], attrs=ds_var["phase"].attrs
        )
        logger.info(f"Added cloud phase ...")

        ds["DQF"] = xr.DataArray(
            ds_var["qcflag"].values, dims=["y", "x"], attrs=ds_var["qcflag"].attrs
        )
        logger.info(f"Added cloud quality flag ...")

        ds["type"] = xr.DataArray(
            ds_var["cldtype"].sel(views=0).values,
            dims=["y", "x"],
            attrs=ds_var["cldtype"].attrs,
        )
        logger.info(f"Added cloud type ...")

        ds["effective_radius"] = xr.DataArray(
            ds_var["cer"].values, dims=["y", "x"], attrs=ds_var["cer"].attrs
        )
        logger.info(f"Added cloud effective radius ...")

        ds["liquid_water_path"] = xr.DataArray(
            ds_var["cwp"].values, dims=["y", "x"], attrs=ds_var["cwp"].attrs
        )
        logger.info(f"Added cloud liquid water path ...")

        ds_patch, xmin, ymin = patch_msg(ds, args.patch_size, args.fov_radius)

        # Reprocess file:
        logger.info(f"Reprocessing patch ...")
        new_ds = reprocess_msg(
            ds=ds_patch,
            time=pd.to_datetime(row["date"]),
            chunksize=64,
            zlib=True,
            shuffle=True,
            complevel=5,
        )  # default in the reprocessing script

        central_lat = ds_patch.latitude.values[
            args.patch_size // 2, args.patch_size // 2
        ]
        central_lon = ds_patch.longitude.values[
            args.patch_size // 2, args.patch_size // 2
        ]
        time_str = pd.to_datetime(row["date"]).strftime("%Y%m%d%H%M%S")
        lat_str = f"{central_lat:+.3f}deg"
        lon_str = f"{central_lon:+.3f}deg"

        patch_filename = f"{time_str}_[{lat_str}_{lon_str}]_patch.nc"
        save_file_name = pathlib.Path(save_path) / patch_filename

        # Save the patched dataset with the specified encoding
        logger.info(f"Saving patched dataset to {save_file_name} ...")
        new_ds.to_netcdf(save_file_name, engine="netcdf4")

        logger.info("Finished successfully ...")

    else:
        logger.warning(f"Not all files available for {row['date']}. Skipping ...")
