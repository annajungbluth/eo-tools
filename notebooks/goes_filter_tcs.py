import numpy as np
import pandas as pd
import goes2go
from tqdm import tqdm
from scipy.interpolate import make_splrep
from loguru import logger

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
    ibtracs_sel = ibtracs[ibtracs.SID==SID].reset_index(drop=True)
    longitudes = ibtracs_sel['LON'].values
    latitudes = ibtracs_sel['LAT'].values
    timestamps = ibtracs_sel['ISO_TIME'].values
    return longitudes, latitudes, timestamps

def interpolate_track(
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        timestamps: np.ndarray,
        s: float = 0
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
    latitudes_intp = make_splrep(timestamps.astype('datetime64[s]'), latitudes, s=s)
    longitudes_intp = make_splrep(timestamps.astype('datetime64[s]'), longitudes, s=s)
    # For now, just return the input arrays
    return latitudes_intp, longitudes_intp

def get_available_goes_times(
        start: pd.Timestamp,
        end: pd.Timestamp, 
        reduction_factor: int = 1,
        satellite: str='noaa-goes16',
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
    goes_df = goes2go.goes_timerange(satellite = satellite, start = start, end = end,
                            download = False, product = 'ABI-L2-MCMIP', domain = 'F')
    
    goes_df['mid_time'] = goes_df.start + (goes_df.end - goes_df.start)/2

    goes_df = goes_df[::reduction_factor].reset_index(drop=True)
    return goes_df


tc_all = pd.read_csv('./IBTrACs-filtered/goes_ibtracs.NA-EP.list.v04r01.csv') # all named storms in the GOES field-of-view
tc_all['ISO_TIME'] = pd.to_datetime(tc_all['ISO_TIME'])

logger.info(f"Successfully loaded IBTrACS dataset with {len(tc_all)} records and {len(tc_all.SID.unique())} unique storms.")

SIDs = tc_all.SID.unique()

sum_df = pd.DataFrame(columns=['file', 'product_mode', 'satellite', 'start', 'end', 'creation',
       'product', 'mode_bands', 'mode', 'band', 'mid_time', 'LAT', 'LON',
       'SID'])

for sid in tqdm(SIDs):
    # Get the tropical cyclone track
    longitudes, latitudes, timestamps = get_selected_tc(tc_all, sid)
    
    # Interpolate the track
    try:
        f_lat, f_lon = interpolate_track(latitudes, longitudes, timestamps)
    except ValueError as e:
        logger.warning(f"Interpolation failed for SID {sid} with error: {e}. Skipping this storm.")
        continue

    # Get available GOES times
    goes_df = get_available_goes_times(pd.to_datetime(timestamps.min()), pd.to_datetime(timestamps.max()))
    
    interpolated_latitudes = []
    interpolated_longitudes = []
    sid_list = []

    for i, goes_time in tqdm(enumerate(goes_df.mid_time.values)):
        interpolated_latitudes.append(np.round(f_lat(goes_time.astype('datetime64[s]')), 5))
        interpolated_longitudes.append(np.round(f_lon(goes_time.astype('datetime64[s]')), 5))
        sid_list.append(sid)
    goes_df['LAT'] = interpolated_latitudes
    goes_df['LON'] = interpolated_longitudes
    goes_df['SID'] = sid_list

    sum_df = pd.concat([sum_df, goes_df], ignore_index=True)

save_path = 'jasmin.goes_ibtracs.NA-EP.list.v04r01.csv'

sum_df.to_csv(save_path, index=False)

logger.info(f"Filtering test set...")

sum_df['start'] = pd.to_datetime(sum_df['start'])   
df_subset = sum_df[sum_df['start'].dt.year.isin([2023, 2024]) & sum_df['start'].dt.day.isin([28, 29, 30, 31])]
df_subset.columns = df_subset.columns.str.lower()

logger.info(f"Reducing to 30-minute cadence...")
df_subset = df_subset[df_subset['start'].dt.minute.isin([0, 30])]
df_subset.to_csv('pretraining-test-goes-cyclones-[2023-2024].csv', index=False)

logger.info(f"Finished filtering test set. Final number of records: {len(df_subset)}")
