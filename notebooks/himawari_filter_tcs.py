import pandas as pd
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from datetime import datetime

import argparse
import os

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import goes2go
import pandas as pd
from tqdm import tqdm

from pyproj import Proj
from scipy.interpolate import make_splrep
from goes2go.himawari_data import _himawari_file_df
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
    latitudes_intp = make_splrep((timestamps).astype(np.datetime64), latitudes, s=s)
    longitudes_intp = make_splrep((timestamps).astype(np.datetime64), longitudes, s=s)
    # For now, just return the input arrays
    return latitudes_intp, longitudes_intp

def get_available_himawari_times(
        start: pd.Timestamp,
        end: pd.Timestamp, 
):
    """
    Get available GOES times within a specified range.
    Args:
        start (pd.Timestamp): Start time.
        end (pd.Timestamp): End time.
    Returns:
        pd.DataFrame: DataFrame containing available GOES times.
    """
    ahi_df = _himawari_file_df(
            "noaa-himawari8", 
            "FLDK", 
            start = start,
            end = end,
            ignore_missing=True)

    # convert file paths to datetime
    years = ahi_df.file.apply(lambda x: x.split('/')[2])
    months = ahi_df.file.apply(lambda x: x.split('/')[3])
    days = ahi_df.file.apply(lambda x: x.split('/')[4])
    times = ahi_df.file.apply(lambda x: x.split('/')[5][:2] + ':' + x.split('/')[5][2:4] + ':00')

    ahi_df['time'] = pd.to_datetime(
        years + '-' + months + '-' + days + ' ' + times,
        format='%Y-%m-%d %H:%M:%S')
    ahi_df['time'] = pd.to_datetime(ahi_df['time'], format='%Y-%m-%d %H:%M:%S')

    files = []
    for time, group in ahi_df.groupby('time'):
        if len(group['file'].tolist()) >= 16:  # Ensure at least 16 files for a valid time
            files.append({
                'start': time,
                'files': group['file'].tolist(),
                'data_format': group['data_format'].tolist()[0],
                'satellite': group['satellite'].tolist()[0],
                'date': group['date'].tolist()[0],
                'domain': group['domain'].tolist()[0],
            })

    df = pd.DataFrame(files)
    return df

# Define the paths to the IBTrACS files
ibtracs_wp_file = '/home/users/annaju/eo-tools/notebooks/IBTrACs/ibtracs.WP.list.v04r01.csv'
ibtracs_sp_file = '/home/users/annaju/eo-tools/notebooks/IBTrACs/ibtracs.SP.list.v04r01.csv'

df_wp = pd.read_csv(ibtracs_wp_file)
df_sp = pd.read_csv(ibtracs_sp_file)

# Filter western pacific storms
df_wp_filtered = df_wp[df_wp.SEASON.isin([2020, 2021, 2022])]
# df_wp_filtered = df_wp[df_wp.SEASON.isin([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])]
df_wp_filtered = df_wp_filtered[df_wp_filtered.NATURE.isin(['TS', 'DS'])]
df_wp_filtered = df_wp_filtered[df_wp_filtered.NAME != 'UNNAMED']
# filter for HIMAWARI FOV
df_wp_filtered = df_wp_filtered[df_wp_filtered.LAT.between(-50, 50) & df_wp_filtered.LON.between(91, 191)]
# filter out storms with less than 8 points
grouped = df_wp_filtered.groupby('SID')
lengths = grouped.size()
df_wp_filtered = df_wp_filtered[df_wp_filtered.SID.isin(lengths[lengths >= 8].index)]

# Filter southern pacific storms
df_sp_filtered = df_sp[df_sp.SEASON.isin([2020, 2021, 2022])]
# df_sp_filtered = df_sp[df_sp.SEASON.isin([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])]
df_sp_filtered = df_sp_filtered[df_sp_filtered.NATURE.isin(['TS', 'DS'])]
df_sp_filtered = df_sp_filtered[df_sp_filtered.NAME != 'UNNAMED']
# filter for HIMAWARI FOV
df_sp_filtered = df_sp_filtered[df_sp_filtered.LAT.between(-50, 50) & df_sp_filtered.LON.between(91, 191)]
# filter out storms with less than 8 points
grouped = df_sp_filtered.groupby('SID')
lengths = grouped.size()
df_sp_filtered = df_sp_filtered[df_sp_filtered.SID.isin(lengths[lengths >= 8].index)]

# Combine the filtered dataframes
df_all = pd.concat([df_wp_filtered, df_sp_filtered], ignore_index=True)

# filter out all storms that started before 2015-07-07 or after 2022-12-12
_ = df_all['ISO_TIME']
df_all['ISO_TIME'] = pd.to_datetime(df_all['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')
df_all['ISO_DATE'] = df_all['ISO_TIME'].dt.date
df_all = df_all[(df_all['ISO_DATE'] >= datetime(2015, 7, 7).date()) & (df_all['ISO_DATE'] <= datetime(2022, 12, 12).date())]
df_all['ISO_TIME'] = _

# # Sample based on intensity
# df_all['USA_SSHS'] = df_all['USA_SSHS'].astype(int)
# max_intensity = df_all['USA_SSHS'].max()
# print(f"Maximum intensity in the dataset: {max_intensity}") 
# # pick 10 storms with intensity > 4  
# intense_storms = df_all[df_all['USA_SSHS'] > 4].SID.unique()
# print(f"Number of storms with intensity > 4: {len(intense_storms)}")
# # sample storms based on intensity
# sampled_storms = df_all[df_all.SID.isin(intense_storms)]

sampled_storms = df_all

SIDs = sampled_storms.SID.unique()
sum_df = pd.DataFrame(columns=['start', 'files', 'data_format', 'satellite', 'date', 'domain', 'LAT', 'LON', 'SID'])

for sid in tqdm(SIDs):
    # Get the tropical cyclone track
    longitudes, latitudes, timestamps = get_selected_tc(sampled_storms, sid)
    
    # Interpolate the track
    f_lat, f_lon = interpolate_track(latitudes, longitudes, timestamps)
    
    # Get available GOES times
    ahi_df = get_available_himawari_times(pd.to_datetime(timestamps.min()), pd.to_datetime(timestamps.max()))
    
    interpolated_latitudes = []
    interpolated_longitudes = []
    sid_list = []

    for i, goes_time in tqdm(enumerate(ahi_df.start.values)):
        interpolated_latitudes.append(np.round(f_lat(goes_time.astype('datetime64[s]')), 5))
        interpolated_longitudes.append(np.round(f_lon(goes_time.astype('datetime64[s]')), 5))
        sid_list.append(sid)
    ahi_df['LAT'] = interpolated_latitudes
    ahi_df['LON'] = interpolated_longitudes
    ahi_df['SID'] = sid_list

    sum_df = pd.concat([sum_df, ahi_df])

# Save the summary DataFrame to a CSV file
sum_df.to_csv('jasmin.himawari_ibtracs-[2020-2022].SP-WP.list.v04r01.csv', index=False)

logger.info(f"Filtering test set...")

sum_df['start'] = pd.to_datetime(sum_df['start'])   
df_subset = sum_df[sum_df['start'].dt.year.isin([2020, 2021, 2022]) & sum_df['start'].dt.day.isin([28, 29, 30, 31])]
df_subset.columns = df_subset.columns.str.lower()

logger.info(f"Reducing to 30-minute cadence...")
df_subset = df_subset[df_subset['start'].dt.minute.isin([0, 30])]
df_subset.to_csv('pretraining-test-himawari-cyclones-[2020-2022].csv', index=False)

logger.info(f"Finished filtering test set. Final number of records: {len(df_subset)}")
