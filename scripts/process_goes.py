#!/home/users/annaju/miniforge3/envs/jasmin-env/bin/python
import pathlib
import goes2go
import pandas as pd
import numpy as np
import xarray as xr
import argparse
from loguru import logger
from tqdm import tqdm
from google.cloud import storage
from satpy import Scene
from datetime import datetime, timedelta
from process_utils import random_datetime, CenterWeightedCropDatasetEditor
from multiprocessing import Pool, cpu_count, set_start_method
import os
import contextlib
import warnings

@contextlib.contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

import s3fs
import fsspec

def reduce_file_size(ds, compression_level=9):
    """
    Reduce the file size of the dataset by converting to float32 and compressing.
    
    Args:
        ds (xarray.Dataset): The dataset to reduce.
        compression_level (int): Compression level for saving the dataset.
    
    Returns:
        xarray.Dataset: Reduced dataset.
    """
    # Create encoding for both data variables and coordinates to avoid warnings
    encoding = {}
    # Handle data variables
    for var in ds.data_vars:
        encoding[var] = {
            "zlib": True,
            "complevel": 9,
            "dtype": "float32",
        }
    # Handle coordinates (especially x, y to avoid serialization warnings)
    for coord in ds.coords:
        if coord in ['x', 'y']:
            encoding[coord] = {
                "dtype": "float32",
                "_FillValue": None  # This prevents the warning
            }
    return ds, encoding


def download_mcmip(dt, patch_size, fov_radius):
    """
    Download GOES MCMIP data. 
    """
    # Create filesystem object inside worker process to avoid fork-safety issues
    fs = fsspec.filesystem('s3', anon=True)
    fsspec_caching = {
    "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
    "block_size": 8
    * 1024
    * 1024,  # size in bytes per block, adjust depends on the file size but the recommended size is in the MB
    }

    # try:
    # Get the list of ABI files for the random datetime
    # Will raise an error if files are not available
    with suppress_warnings():
        abi_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=dt, 
            end=dt + timedelta(minutes=30),
            download=False,
            domain='F',
            product="ABI-L2-MCMIP",
        )
    
    # Check that we have MCMIP files for the selected date
    if len(abi_files) == 0:
        logger.warning(f"No MCMIP files found for date {dt}.")
        return None
    logger.info(f"Found {len(abi_files)} MCMIP files for date {dt}.")
    
    # Load the first file into an xarray dataset
    ds = xr.open_dataset(fs.open(abi_files['file'][0], **fsspec_caching), engine="h5netcdf")
    meas_time_str = datetime.strftime(abi_files['start'][0], '%Y%m%d%H%M%S')
    logger.info(f"Loaded MCMIP file: {abi_files['file'][0]} ...")
    
    # Crop dataset into patch
    crop = CenterWeightedCropDatasetEditor(
        patch_shape=(patch_size, patch_size), 
        fov_radius=fov_radius,
        satellite = 'goes')
    result = crop(ds)

    logger.info(f"Cropping dataset to patch of size {patch_size} with FOV radius {fov_radius} ...")
    
    if result is None: # i.e. if no valid patch was found
        logger.warning(f"Could not find valid patch for {dt}")
        return None
    else:
        patch_ds, xmin, ymin = result
        return (patch_ds, xmin, ymin, meas_time_str)

    # except:
    #     logger.error(f"Could not process GOES file for {dt} ...")
    #     return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date in YYYY-MM-DD format")
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of the patch to crop from the dataset")
    parser.add_argument("--fov_radius", type=float, default=0.6, help="Field of view radius for cropping")

    args = parser.parse_args()  

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")

    # Create output directory if it doesn't exist
    # save_path = pathlib.Path("/work/scratch-nopw2/annaju/goes_temp/")
    save_path = pathlib.Path("./")
    save_path.mkdir(parents=True, exist_ok=True)

    # Convert start and end dates to datetime objects
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # Select random datetime
    dt = random_datetime(start, end)

    result = download_mcmip (dt, 
                              patch_size=args.patch_size, 
                              fov_radius=args.fov_radius)
    
    if result is not None:
        patch_ds, xmin, ymin, dt_str = result

        # Reduce file size
        patch_ds, encoding = reduce_file_size(patch_ds, compression_level=9)

        logger.info(f"Saving patch for {dt_str} at coordinates ({xmin}, {ymin})...")

        # Save patch to netcdf file
        patch_filename = f"{dt_str}_patch_{xmin}_{ymin}.nc"
        patch_ds.to_netcdf(f"{save_path}/{patch_filename}", encoding=encoding)

        # try loading the file to check if it was saved correctly
        try:
            xr.open_dataset(f"{save_path}/{patch_filename}")
            logger.info(f"Patch saved successfully: {patch_filename}")
        except Exception as e:
            raise RuntimeError(f"Failed to load saved patch file: {patch_filename}. Error: {e}")
        
        # Upload to GCP
        logger.info(f"Uploading file to GCP...")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/users/annaju/esl-3d-clouds-extremes-baa3a73d57dc.json"  # TODO: Add credentials
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("2025-esl-3dclouds-extremes-datasets")
        blob = bucket.blob(f'pre-training/goes/mcmip-update/{patch_filename}')
        blob.upload_from_filename(f"{save_path}/{patch_filename}")

        # remove local file
        (save_path / patch_filename).unlink()

        logger.info("Finished successfully ...")
    else:
        logger.warning(f"No valid patch found for number {args.seed} and {dt}. Skipping upload ...")
