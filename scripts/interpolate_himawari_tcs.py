import argparse
import os
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import goes2go
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

# from goes2go.himawari_data import _himawari_file_df
from loguru import logger
from matplotlib.collections import LineCollection
from pyproj import Proj
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

# Connect to AWS public buckets
fs = s3fs.S3FileSystem(anon=True)

# Define parameter options and aliases
# ------------------------------------
_himawari_satellite = {
    "noaa-himawari8": [8, "8", "H8", "HIMAWARI8", "HIMAWARI-8"],
    "noaa-himawari9": [9, "9", "H9", "HIMAWARI9", "HIMAWARI-9"],
}

_himawari_domain = {
    "Japan": ["C", "CONUS", "JAPAN"],
    "FLDK": ["F", "FULL", "FULLDISK", "FULL DISK"],
    "Target": ["M", "MESOSCALE", "M1", "M2", "TARGET"],
}

_himawari_resolution = {
    "R05": [0.5, 500, "0.5", "500"],
    "R10": [1, 1000, "1", "1000"],
    "R20": [2, 2000, "2", "2000"],
}

_himawari_bands = dict(
    zip(
        [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B09",
            "B10",
            "B11",
            "B12",
            "B13",
            "B14",
            "B15",
            "B16",
        ],
        range(1, 16 + 1),
    )
)


def check_time_in_range(dt, query_dt, delta_minutes=2):
    start_dt = query_dt - timedelta(minutes=delta_minutes)
    end_dt = query_dt + timedelta(minutes=delta_minutes)
    return (start_dt <= dt).all() and (dt <= end_dt).all()


def get_correct_files(files, query_dt):
    if len(files) == 0:
        return None
    else:
        files = files.copy()
        files["diff"] = abs(pd.to_datetime(files["time"]) - query_dt)
        correct_files = files[files["diff"] == files["diff"].min()]
        if check_time_in_range(pd.to_datetime(correct_files["time"]), query_dt):
            return correct_files
        else:
            return None


def _himawari_file_df(
    satellite,
    domain,
    query_dt,
    bands=None,
    resolutions=None,
    refresh=True,
    ignore_missing=False,
):
    """Get list of requested GOES files as pandas.DataFrame.

    Parameters
    ----------
    satellite : str
    domain : str
    query_dt : datetime
    bands : None, int, or list
        Specify the AHI channels to retrieve.
    resolutions : None, str, or list
        Specify the AHI resolutions to retrieve.
    refresh : bool
        Refresh the s3fs.S3FileSystem object when files are listed.
        Default True will refresh and not use a cached list.
    """
    params = locals()

    query_dt = pd.to_datetime(query_dt)

    # List all files for each date
    # ----------------------------
    files = []
    path = f"{satellite}/AHI-L1b-{domain}/{query_dt.year}/{query_dt.month:02d}/{query_dt.day:02d}/{query_dt.hour:02d}{query_dt.minute:02d}"
    if ignore_missing is True:
        try:
            files += fs.ls(path, refresh=refresh)
        except FileNotFoundError:
            print(f"Ignored missing dir: {path}")
    else:
        files += fs.ls(path, refresh=refresh)

    # Build a table of the files
    # --------------------------
    if len(files) == 0:
        return pd.DataFrame()  # Return an empty DataFrame if no files are found
    else:
        df = pd.DataFrame(files, columns=["file"])

    df = df.loc[df["file"].str.contains(".DAT.bz2", na=False)].copy()
    if df.empty:
        return df

    df[
        [
            "data_format",
            "satellite",
            "date",
            "time",
            "band",
            "domain",
            "resolution",
            "sector",
        ]
    ] = (
        df["file"]
        .str.rsplit("/", expand=True)
        .iloc[:, -1]
        .str.rsplit(".", expand=True)
        .loc[:, 0]
        .str.rsplit("_", expand=True)
    )

    # Filter files by band number
    # ---------------------------
    if bands is not None:
        if not hasattr(bands, "__len__") or isinstance(bands, (str, bytes, bytearray)):
            bands = [bands]
        for i_band, band in enumerate(bands):
            if band not in _himawari_bands:
                try:
                    bands[i_band] = dict(
                        zip(_himawari_bands.values(), _himawari_bands.keys())
                    )[band]
                except KeyError:
                    raise ValueError(f"Band {band} is not a valid AHI channel")
        df = df.loc[df.band.isin(bands)]

    # Filter files by resolution
    # --------------------------
    if resolutions is not None:
        if not hasattr(resolutions, "__len__") or isinstance(
            resolutions, (str, bytes, bytearray)
        ):
            resolutions = [resolutions]
        for i_resolution, resolution in enumerate(resolutions):
            if resolution not in _himawari_resolution:
                for key, aliases in _himawari_resolution.items():
                    if resolution in aliases:
                        resolutions[i_resolution] = key
                        break
                else:
                    raise ValueError(
                        f"Resolution {resolution} is not a valid AHI resolution"
                    )
        df = df.loc[df.resolution.isin(resolutions)]
    elif not df.empty:
        # If None pick the highest resolution for each
        df = df.loc[
            df.resolution
            == df.groupby("band").resolution.unique().str[0][df.band].to_numpy()
        ]

    # Filter files by requested time range
    # ------------------------------------
    # Convert filename datetime string to datetime object
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df.date + df.time, format="%Y%m%d%H%M")

    for i in params:
        df.attrs[i] = params[i]

    return df


def get_selected_tc(
    ibtracs: pd.DataFrame,
    SID: str,
):
    """
    Function to select the tropical cyclone track from the IBTrACS dataset.
    Args:
        ibtracs (pd.DataFrame): The IBTrACS dataset.
        SID (str): The storm ID of the tropical cyclone.
    Returns:
        tuple: A tuple containing the longitudes, latitudes, and timestamps of the tropical cyclone track.
    """
    ibtracs_sel = ibtracs[ibtracs.SID == SID].reset_index(drop=True)
    longitudes = ibtracs_sel["LON"].values
    latitudes = ibtracs_sel["LAT"].values
    timestamps = ibtracs_sel["ISO_TIME"].values
    return longitudes, latitudes, timestamps


def interpolate_track(
    latitudes: np.ndarray, longitudes: np.ndarray, timestamps: np.ndarray, s: float = 0
):
    """
    Interpolate the track of a tropical cyclone.

    Args:
        latitudes (np.ndarray): Array of latitudes.
        longitudes (np.ndarray): Array of longitudes.
        timestamps (np.ndarray): Array of timestamps.

    Returns:
        np.ndarray: Interpolated latitudes, longitudes, and timestamps.
    """
    # Convert timestamps to integer seconds for SciPy interpolation.
    ts_sec = np.asarray(timestamps).astype("datetime64[s]").astype(np.int64)
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)

    # IBTrACS can contain duplicated or unsorted timestamps; make x strictly increasing.
    order = np.argsort(ts_sec)
    ts_sec = ts_sec[order]
    latitudes = latitudes[order]
    longitudes = longitudes[order]

    unique_ts, inverse = np.unique(ts_sec, return_inverse=True)
    if len(unique_ts) < 2:
        raise ValueError("Need at least two unique timestamps to interpolate track")

    # Average points sharing the same timestamp to preserve one value per x.
    latitudes = np.bincount(inverse, weights=latitudes) / np.bincount(inverse)
    longitudes = np.bincount(inverse, weights=longitudes) / np.bincount(inverse)
    ts_sec = unique_ts

    # Normalize into [-180, 180] first, then unwrap so tracks crossing 180/-180 stay continuous.
    longitudes_norm = ((longitudes + 180.0) % 360.0) - 180.0
    longitudes_unwrapped = np.rad2deg(
        np.unwrap(np.deg2rad(longitudes_norm), discont=np.deg2rad(180.0))
    )

    # Fall back to linear interpolation for short tracks.
    spline_k = min(3, len(ts_sec) - 1)

    f_lat_spline = UnivariateSpline(ts_sec, latitudes, s=s, k=spline_k)
    f_lon_spline = UnivariateSpline(ts_sec, longitudes_unwrapped, s=s, k=spline_k)

    def f_lat(t):
        t_sec = np.asarray(t).astype("datetime64[s]").astype(np.int64)
        return f_lat_spline(t_sec)

    def f_lon(t):
        t_sec = np.asarray(t).astype("datetime64[s]").astype(np.int64)
        lon_unwrapped = f_lon_spline(t_sec)
        # Wrap back to [-180, 180] for downstream compatibility.
        return ((lon_unwrapped + 180.0) % 360.0) - 180.0

    return f_lat, f_lon


def get_available_himawari_times(
    start: pd.Timestamp, end: pd.Timestamp, satellite: str = "noaa-himawari8"
):
    """
    Get available HIMAWARI times within a specified range.
    Args:
        start (pd.Timestamp): Start time.
        end (pd.Timestamp): End time.
        satellite (str): Satellite name.
    Returns:
        pd.DataFrame: DataFrame containing available HIMAWARI times.
    """

    rows = []
    for query_dt in pd.date_range(
        start=start.floor("10min"), end=end.ceil("10min"), freq="10min"
    ):
        try:
            ahi_df = _himawari_file_df(
                satellite,
                "FLDK",
                query_dt,
                ignore_missing=True,
                refresh=False,
            )
        except ValueError as e:
            logger.error(f"Error fetching AHI files for {query_dt}: {e}")
            ahi_df = pd.DataFrame()  # Create an empty DataFrame to avoid further errors

        if ahi_df is None or ahi_df.empty:
            logger.warning(f"No AHI files found for {query_dt}")
        else:
            ahi_df = get_correct_files(ahi_df, query_dt)
            if ahi_df is None:
                logger.warning(f"No AHI files found within 2 minutes of {query_dt}")
                ahi_df = pd.DataFrame()

        if ahi_df.empty:
            continue

        assert (
            len(ahi_df["time"].unique()) == 1
        )  # Check that there is only one unique timestamp

        if len(ahi_df["file"].tolist()) >= 16:
            rows.append(
                {
                    "start": ahi_df["time"].iloc[0],
                    "files": ahi_df["file"].tolist(),
                    "data_format": ahi_df["data_format"].iloc[0],
                    "satellite": ahi_df["satellite"].iloc[0],
                    "query_dt": query_dt,
                    "domain": ahi_df["domain"].iloc[0],
                    "date": ahi_df["date"].iloc[0],
                }
            )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------------
# VARIABLES TO UPDATE:
tc_all = pd.read_csv(
    "./files/matched-ibtracs-himawari-[2023-2025].csv"
)  # TODO: UPDATE THESE!
save_path = "tmp-himawari-[2023-2025].csv"

fast_compile = True
# -------------------------------------------------------------------------------------------

tc_all["ISO_TIME"] = pd.to_datetime(tc_all["ISO_TIME"])

logger.info(
    f"Successfully loaded dataset with {len(tc_all)} records and {len(tc_all.SID.unique())} unique storms."
)

SIDs = tc_all.SID.unique()

if fast_compile:
    sum_df = pd.DataFrame(
        columns=[
            "start",
            "satellite",
            "LAT",
            "LON",
            "SID",
            "NAME",
            "USA_ATCF_ID",
        ]
    )
else:
    sum_df = pd.DataFrame(
        columns=[
            "start",
            "satellite",
            "date",
            "domain",
            "LAT",
            "LON",
            "SID",
            "NAME",
            "USA_ATCF_ID",
        ]
    )

for sid in tqdm(SIDs):
    # Get the tropical cyclone track
    longitudes, latitudes, timestamps = get_selected_tc(tc_all, sid)
    satellites = tc_all[tc_all.SID == sid]["SATELLITE"].unique()
    if len(satellites) != 1:
        raise ValueError(
            f"More/less than one satellite found for SID {sid}: {satellites}"
        )
    satellite = satellites[0]
    name = tc_all[tc_all.SID == sid]["NAME"].unique()[0]
    usa_atcf_id = tc_all[tc_all.SID == sid]["USA_ATCF_ID"].unique()[0]

    # Interpolate the track
    # TODO: There is a bug in the interpolation track over the longitude -180/+180 line that needs fixing
    f_lat, f_lon = interpolate_track(latitudes, longitudes, timestamps)

    if fast_compile:  # Fill in dataset without querying the actual files
        ahi_df = pd.DataFrame()
        ahi_df["start"] = list(
            pd.date_range(
                start=pd.to_datetime(timestamps.min()).floor("10min"),
                end=pd.to_datetime(timestamps.max()).ceil("10min"),
                freq="10min",
            )
        )
        ahi_df["satellite"] = [satellite] * len(ahi_df)
    else:
        # Get available HIMAWARI times
        ahi_df = get_available_himawari_times(
            start=pd.to_datetime(timestamps.min()),
            end=pd.to_datetime(timestamps.max()),
            satellite=satellite,
        )

    interpolated_latitudes = []
    interpolated_longitudes = []
    sid_list = []
    name_list = []
    usa_atcf_id_list = []

    for i, him_time in tqdm(enumerate(ahi_df.start.values)):
        t = np.datetime64(him_time, "s")
        interpolated_latitudes.append(np.round(float(f_lat(t)), 5))
        interpolated_longitudes.append(np.round(float(f_lon(t)), 5))
        sid_list.append(sid)
        name_list.append(name)
        usa_atcf_id_list.append(usa_atcf_id)

    ahi_df["LAT"] = interpolated_latitudes
    ahi_df["LON"] = interpolated_longitudes
    ahi_df["SID"] = sid_list
    ahi_df["NAME"] = name_list
    ahi_df["USA_ATCF_ID"] = usa_atcf_id_list

    sum_df = pd.concat([sum_df, ahi_df])

# Save the summary DataFrame to a CSV file
sum_df.to_csv(save_path, index=False)

# logger.info(f"Filtering test set...")
# sum_df['start'] = pd.to_datetime(sum_df['start'])
# df_subset = sum_df[sum_df['start'].dt.year.isin([2020, 2021, 2022]) & sum_df['start'].dt.day.isin([28, 29, 30, 31])]
# df_subset.columns = df_subset.columns.str.lower()

# logger.info(f"Reducing to 30-minute cadence...")
# df_subset = df_subset[df_subset['start'].dt.minute.isin([0, 30])]
# df_subset.to_csv('pretraining-test-himawari-cyclones-[2020-2022].csv', index=False)

logger.info(f"Finished processing. Final number of records: {len(sum_df)}")
