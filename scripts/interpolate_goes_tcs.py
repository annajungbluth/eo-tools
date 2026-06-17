import goes2go
import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import make_splrep
from tqdm import tqdm


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
    # Implement interpolation logic here
    latitudes_intp = make_splrep(timestamps.astype("datetime64[s]"), latitudes, s=s)
    longitudes_intp = make_splrep(timestamps.astype("datetime64[s]"), longitudes, s=s)
    # For now, just return the input arrays
    return latitudes_intp, longitudes_intp


def get_available_goes_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
    reduction_factor: int = 1,  # No reduction if set to 1
    satellite: str = "noaa-goes16",
):
    """
    Get available GOES times within a specified range.
    Args:
        start (pd.Timestamp): Start time.
        end (pd.Timestamp): End time.
        reduction_factor (int): Factor to reduce the number of times.
    Returns:
        pd.DataFrame: DataFrame containing available GOES times.
    """
    goes_df = goes2go.goes_timerange(
        satellite=satellite,
        start=start,
        end=end,
        download=False,
        product="ABI-L2-MCMIP",
        domain="F",
    )

    goes_df["mid_time"] = goes_df.start + (goes_df.end - goes_df.start) / 2

    goes_df = goes_df[::reduction_factor].reset_index(drop=True)
    return goes_df


# -------------------------------------------------------------------------------------------
# VARIABLES TO UPDATE:
tc_all = pd.read_csv(
    "./files/matched-ibtracs-goes-east-[2023-2025].csv"
)  # TODO: UPDATE THESE!
save_path = "matched-goes-east-[2023-2025].csv"
# -------------------------------------------------------------------------------------------

tc_all["ISO_TIME"] = pd.to_datetime(tc_all["ISO_TIME"])

logger.info(
    f"Successfully loaded dataset with {len(tc_all)} records and {len(tc_all.SID.unique())} unique storms."
)

SIDs = tc_all.SID.unique()

sum_df = pd.DataFrame(
    columns=[
        "file",
        "product_mode",
        "satellite",
        "start",
        "end",
        "creation",
        "product",
        "mode_bands",
        "mode",
        "band",
        "mid_time",
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
    try:
        f_lat, f_lon = interpolate_track(latitudes, longitudes, timestamps)
    except ValueError as e:
        logger.warning(
            f"Interpolation failed for SID {sid} with error: {e}. Skipping this storm."
        )
        continue

    # Get available GOES times
    goes_df = get_available_goes_times(
        start=pd.to_datetime(timestamps.min()),
        end=pd.to_datetime(timestamps.max()),
        satellite=satellite,
    )

    interpolated_latitudes = []
    interpolated_longitudes = []
    sid_list = []
    name_list = []
    usa_atcf_id_list = []

    for i, goes_time in tqdm(enumerate(goes_df.mid_time.values)):
        interpolated_latitudes.append(
            np.round(f_lat(goes_time.astype("datetime64[s]")), 5)
        )
        interpolated_longitudes.append(
            np.round(f_lon(goes_time.astype("datetime64[s]")), 5)
        )
        sid_list.append(sid)
        name_list.append(name)
        usa_atcf_id_list.append(usa_atcf_id)

    goes_df["LAT"] = interpolated_latitudes
    goes_df["LON"] = interpolated_longitudes
    goes_df["SID"] = sid_list
    goes_df["NAME"] = name_list
    goes_df["USA_ATCF_ID"] = usa_atcf_id_list

    sum_df = pd.concat([sum_df, goes_df], ignore_index=True)

sum_df.to_csv(save_path, index=False)

# logger.info(f"Filtering test set...")
# sum_df['start'] = pd.to_datetime(sum_df['start'])
# df_subset = sum_df[sum_df['start'].dt.year.isin([2023, 2024]) & sum_df['start'].dt.day.isin([28, 29, 30, 31])]
# df_subset.columns = df_subset.columns.str.lower()

# logger.info(f"Reducing to 30-minute cadence...")
# df_subset = df_subset[df_subset['start'].dt.minute.isin([0, 30])]
# df_subset.to_csv('pretraining-test-goes-cyclones-[2023-2024].csv', index=False)

logger.info(f"Finished processing. Final number of records: {len(sum_df)}")
