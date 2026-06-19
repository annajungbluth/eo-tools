from datetime import timedelta
from functools import lru_cache

import pandas as pd
import s3fs
from loguru import logger
from tqdm import tqdm

# Connect to AWS public buckets
fs = s3fs.S3FileSystem(anon=True)


@lru_cache(maxsize=50000)
def _ls_path_cached(path):
    """Cache S3 directory listings to avoid repeated network calls."""
    try:
        return tuple(fs.ls(path, refresh=False))
    except FileNotFoundError:
        return tuple()


def _list_s3_path(path, refresh=False, ignore_missing=False):
    """List files in an S3 path, optionally using cache for speed."""
    if refresh:
        try:
            return fs.ls(path, refresh=True)
        except FileNotFoundError:
            if ignore_missing:
                return []
            raise

    files = _ls_path_cached(path)
    if not files and ignore_missing:
        return []
    return list(files)


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
    files += _list_s3_path(path, refresh=refresh, ignore_missing=ignore_missing)

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


def _himawari_l2_df(
    satellite,
    domain,
    query_dt,
    level="L2",
    bands=None,
    resolutions=None,
    refresh=True,
    ignore_missing=False,
):
    """Get list of requested GOES files as pandas.DataFrame.

    Parameters
    ----------
    satellite : str
    product : str
    query_dt : datetime
    bands : None, int, or list
        Specify the ABI channels to retrieve.
    refresh : bool
        Refresh the s3fs.S3FileSystem object when files are listed.
        Default True will refresh and not use a cached list.
    """
    params = locals()

    query_dt = pd.to_datetime(query_dt)

    # List all files for each date
    # ----------------------------
    files = []
    if level == "L2":
        path = f"{satellite}/AHI-{level}-FLDK-{domain}/{query_dt.year}/{query_dt.month:02d}/{query_dt.day:02d}/{query_dt.hour:02d}{query_dt.minute:02d}"
    else:
        raise ValueError(f"Level {level} is not supported. Only 'L2' is supported.")
    files += _list_s3_path(path, refresh=refresh, ignore_missing=ignore_missing)

    # Build a table of the files
    # --------------------------
    df = pd.DataFrame(files, columns=["file"])
    if df.empty:
        return df
    if len(df["file"].iloc[0].split("/")[-1].split(".")[0].split("_")) == 9:
        df[
            [
                "satellite",
                "instrument",
                "domain",
                "date",
                "time",
                "xxx",
                "xxxx",
                "variable",
                "language",
            ]
        ] = (
            df["file"]
            .str.rsplit("/", expand=True)
            .iloc[:, -1]
            .str.rsplit(".", expand=True)
            .loc[:, 0]
            .str.rsplit("_", expand=True)
        )
        variable_mapping = {"CHGT": "HEIGHT", "CMSK": "MASK", "CPHS": "PHASE"}
        df["variable"] = df["variable"].map(variable_mapping).fillna(df["variable"])
        df["datetime"] = pd.to_datetime(df.date + df.time, format="%Y%j%H%M")
    elif len(df["file"].iloc[0].split("/")[-1].split(".")[0].split("_")) == 10:
        df[
            [
                "satellite",
                "instrument",
                "resolution",
                "domain",
                "date",
                "time",
                "xxx",
                "xxxx",
                "variable",
                "language",
            ]
        ] = (
            df["file"]
            .str.rsplit("/", expand=True)
            .iloc[:, -1]
            .str.rsplit(".", expand=True)
            .loc[:, 0]
            .str.rsplit("_", expand=True)
        )
        variable_mapping = {"CHGT": "HEIGHT", "CMSK": "MASK", "CPHS": "PHASE"}
        df["variable"] = df["variable"].map(variable_mapping).fillna(df["variable"])
        df["datetime"] = pd.to_datetime(df.date + df.time, format="%Y%j%H%M")
    elif len(df["file"].iloc[0].split("/")[-1].split(".")[0].split("_")) == 6:
        df[["satellite-product", "version", "satellite", "start", "end", "xxx"]] = (
            df["file"]
            .str.rsplit("/", expand=True)
            .iloc[:, -1]
            .str.rsplit(".", expand=True)
            .loc[:, 0]
            .str.rsplit("_", expand=True)
        )
        satellite_mapping = {
            "AHI-CHGT": "HEIGHT",
            "AHI-CMSK": "MASK",
            "AHI-CPHS": "PHASE",
        }
        df["variable"] = df["satellite-product"].map(satellite_mapping)
        # Remove the leading "s" and tolerate filenames with an extra trailing digit.
        df["start"] = df["start"].astype(str).str.extract(r"(\d{14})", expand=False)
        df["datetime"] = pd.to_datetime(
            df["start"], format="%Y%m%d%H%M%S", errors="coerce"
        )
        df = df.dropna(subset=["datetime"])
    else:
        raise ValueError(f"Unexpected filename format: {df['file'].iloc[0]}")
    return df


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
            return correct_files["file"].tolist()
        else:
            return None


def get_correct_l2_file(files, variable):
    assert variable in [
        "height",
        "mask",
        "phase",
    ], f"Variable {variable} is not supported"
    if len(files) == 0:
        return None
    else:
        var_file = files["file"][files["variable"].str.lower() == variable]
        if len(var_file) == 0:
            return None
        else:
            return var_file.iloc[0]


def main():
    # -------------------------------------------------------------------------------------------
    # VARIABLES TO UPDATE:
    path = "./files/tmp-himawari-[2023-2025].csv"
    save_path = "matched-himawari-[2023-2025]-with-additional-variables.csv"
    # -------------------------------------------------------------------------------------------

    df = pd.read_csv(path)
    df_copy = df.copy()

    logger.info(f"Loaded dataframe with {len(df_copy)} rows from {path}")

    if "start" in df_copy.columns:
        df_copy["date"] = pd.to_datetime(df_copy["start"])

    results_by_time = {}
    times = pd.to_datetime(df_copy["date"])

    for i, query_dt in tqdm(enumerate(times), total=len(times)):
        satellite = (
            df_copy["satellite"].iloc[i]
            if "satellite" in df_copy.columns
            else "noaa-himawari8"
        )
        # Check AHI files
        try:
            ahi_files = _himawari_file_df(
                satellite,
                "FLDK",
                query_dt,
                ignore_missing=True,
                refresh=False,
            )
        except ValueError as e:
            logger.error(f"Error fetching AHI files for {query_dt}: {e}")
            ahi_files = (
                pd.DataFrame()
            )  # Create an empty DataFrame to avoid further errors
        if ahi_files is None or ahi_files.empty:
            logger.warning(f"No AHI files found for {query_dt}")
        else:
            ahi_files = get_correct_files(ahi_files, query_dt)
            if ahi_files is None:
                logger.warning(f"No AHI files found within 2 minutes of {query_dt}")

        # Cloud products at 2 km
        try:
            cloud_files_ = _himawari_l2_df(
                satellite,
                "Clouds",
                query_dt,
                level="L2",
                ignore_missing=True,
                refresh=False,
            )
        except ValueError as e:
            logger.error(f"Error fetching Cloud files for {query_dt}: {e}")
            cloud_files_ = (
                pd.DataFrame()
            )  # Create an empty DataFrame to avoid further errors

        height_file = get_correct_l2_file(cloud_files_, "height")
        mask_file = get_correct_l2_file(cloud_files_, "mask")
        phase_file = get_correct_l2_file(cloud_files_, "phase")

        has_ahi_files = ahi_files is not None and len(ahi_files) > 0
        has_l2_files = all(x is not None for x in [height_file, mask_file, phase_file])

        results_by_time[query_dt] = {
            "ahi_file": list(ahi_files) if ahi_files is not None else None,
            "height_file": height_file,
            "mask_file": mask_file,
            "phase_file": phase_file,
            "all_available": has_ahi_files and has_l2_files,
        }

    mapped_results = pd.to_datetime(df_copy["date"]).map(results_by_time)
    df_copy["ahi_file"] = mapped_results.map(
        lambda x: x["ahi_file"] if isinstance(x, dict) else None
    )
    df_copy["height_file"] = mapped_results.map(
        lambda x: x["height_file"] if isinstance(x, dict) else None
    )
    df_copy["mask_file"] = mapped_results.map(
        lambda x: x["mask_file"] if isinstance(x, dict) else None
    )
    df_copy["phase_file"] = mapped_results.map(
        lambda x: x["phase_file"] if isinstance(x, dict) else None
    )
    df_copy["all_available"] = mapped_results.map(
        lambda x: x["all_available"] if isinstance(x, dict) else False
    )

    df_copy.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
