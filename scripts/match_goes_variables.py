import os 
import pandas as pd
import goes2go
from datetime import datetime, timedelta
from tqdm import tqdm
from loguru import logger


path = './files/pretraining-test-goes-[2023-2024].csv'
df = pd.read_csv(path)
df_copy = df.copy()

logger.info(f"Loaded dataframe with {len(df)} rows from {path}")

def check_time_in_range(dt, query_dt, delta_minutes=2):
    start_dt = query_dt - timedelta(minutes=delta_minutes)
    end_dt = query_dt + timedelta(minutes=delta_minutes)
    return start_dt <= dt <= end_dt

def get_correct_file(files, query_dt):
    if len(files) == 0:
        return None
    else:
        files['diff'] = abs(pd.to_datetime(files['start']) - query_dt)
        correct_file = files.loc[files['diff'].idxmin()]
        if check_time_in_range(pd.to_datetime(correct_file['start']), query_dt):
            return correct_file['file']
        else:
            return None

abi_list = []
acha_list = []
achp_list = []
cod_list = []
acht_list = []
acm_list = []
actp_list = []
cps_list = []
all_available = []

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    query_dt = pd.to_datetime(row['date'])
    start_dt = query_dt - timedelta(minutes=10)
    end_dt = query_dt + timedelta(minutes=10)

    # Check ABI files
    try:
        abi_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=start_dt, 
            end=end_dt,
            download=False,
            domain='F',
            product="ABI-L2-MCMIP",
        )
    except ValueError as e:
        logger.error(f"Error fetching ABI files for {query_dt}")
        abi_files = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
    abi_file = get_correct_file(abi_files, query_dt)
    abi_list.append(abi_file)

    # Cloud height at 2 km
    try:
        acha_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=query_dt - timedelta(minutes=10), 
            end=query_dt + timedelta(minutes=10),
            download=False,
            domain='F',
            product="ABI-L2-ACHA2KMF",
        )
    except ValueError as e:
        logger.error(f"Error fetching ACHA files for {query_dt}")
        acha_files = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
    acha_file = get_correct_file(acha_files, query_dt)
    acha_list.append(acha_file)

    # Cloud pressure at 2 km
    try:
        achp_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=query_dt - timedelta(minutes=10), 
            end=query_dt + timedelta(minutes=10),
            download=False,
            domain='F',
            product="ABI-L2-ACHP2KMF",
        )
    except ValueError as e:
        logger.error(f"Error fetching ACHP files for {query_dt}")
        achp_files = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
    achp_file = get_correct_file(achp_files, query_dt)
    achp_list.append(achp_file)

    # Cloud optical depth at 2 km
    try:
        cod_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=query_dt - timedelta(minutes=10), 
            end=query_dt + timedelta(minutes=10),
            download=False,
            domain='F',
            product="ABI-L2-COD2KMF",
        )
    except ValueError as e:
        logger.error(f"Error fetching COD files for {query_dt}")
        cod_files = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
    cod_file = get_correct_file(cod_files, query_dt)
    cod_list.append(cod_file)

    # Cloud temperature at 2 km
    try:
        acht_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=query_dt - timedelta(minutes=10), 
            end=query_dt + timedelta(minutes=10),
            download=False,
            domain='F',
            product="ABI-L2-ACHTF",
        )
    except ValueError as e:
        logger.error(f"Error fetching ACHT files for {query_dt}")
        acht_files = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
    acht_file = get_correct_file(acht_files, query_dt)
    acht_list.append(acht_file)

    # Clear sky mask at 2 km
    try:
        acm_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=query_dt - timedelta(minutes=10), 
            end=query_dt + timedelta(minutes=10),
            download=False,
            domain='F',
            product="ABI-L2-ACMF",
        )
    except ValueError as e:
        logger.error(f"Error fetching ACM files for {query_dt}")
        acm_files = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
    acm_file = get_correct_file(acm_files, query_dt)
    acm_list.append(acm_file)

    # Cloud phase at 2 km
    try:
        actp_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=query_dt - timedelta(minutes=10), 
            end=query_dt + timedelta(minutes=10),
            download=False,
            domain='F',
            product="ABI-L2-ACTPF",
        )
    except ValueError as e:
        logger.error(f"Error fetching ACTP files for {query_dt}")
        actp_files = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
    actp_file = get_correct_file(actp_files, query_dt)
    actp_list.append(actp_file)

    # Cloud particle size at 2 km
    try:
        cps_files = goes2go.goes_timerange(
            satellite='noaa-goes16',
            start=query_dt - timedelta(minutes=10), 
            end=query_dt + timedelta(minutes=10),
            download=False,
            domain='F',
            product="ABI-L2-CPSF",
        )
    except ValueError as e:
        logger.error(f"Error fetching CPS files for {query_dt}")
        cps_files = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
    cps_file = get_correct_file(cps_files, query_dt)
    cps_list.append(cps_file)

    if all([abi_file, acha_file, achp_file, cod_file, acht_file, acm_file, actp_file, cps_file]):
        all_available.append(True)
    else:
        all_available.append(False)

df_copy['abi_file'] = abi_list
df_copy['acha_file'] = acha_list
df_copy['achp_file'] = achp_list
df_copy['cod_file'] = cod_list
df_copy['acht_file'] = acht_list
df_copy['acm_file'] = acm_list
df_copy['actp_file'] = actp_list
df_copy['cps_file'] = cps_list
df_copy['all_available'] = all_available

save_path = './files/pretraining-test-goes-[2023-2024]-with-additional-variables.csv'

df_copy.to_csv(save_path, index=False)
