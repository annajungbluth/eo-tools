import pathlib
import tempfile
import warnings
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pyproj import Proj
from satpy import Scene

warnings.simplefilter("ignore")

GOES_WAVELENGTHS = {
    "CMI_C01": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 450.5,
        "center_wavelength": 470.0,
        "max_wavelength": 490.6,
    },  # 0.47,
    "CMI_C02": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 596.3,
        "center_wavelength": 640.0,
        "max_wavelength": 682.1,
    },  # 0.64,
    "CMI_C03": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 846.9,
        "center_wavelength": 870.0,
        "max_wavelength": 882.0,
    },  # 0.87,
    "CMI_C04": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 1366.3,
        "center_wavelength": 1380.0,
        "max_wavelength": 1380.3,
    },  # 1.38,
    "CMI_C05": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 1587.6,
        "center_wavelength": 1610.0,
        "max_wavelength": 1632.4,
    },  # 1.61,
    "CMI_C06": {
        "reso_og": 3000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 2220.2,
        "center_wavelength": 2250.0,
        "max_wavelength": 2265.5,
    },  # 2.25,
    "CMI_C07": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 3802.7,
        "center_wavelength": 3890.0,
        "max_wavelength": 3992.2,
    },  # 3.89,
    "CMI_C08": {
        "reso_og": 3000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 5790.4,
        "center_wavelength": 6170.0,
        "max_wavelength": 6590.7,
    },  # 6.17,
    "CMI_C09": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 6725.0,
        "center_wavelength": 6930.0,
        "max_wavelength": 7142.9,
    },  # 6.93,
    "CMI_C10": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 7242.7,
        "center_wavelength": 7340.0,
        "max_wavelength": 7431.1,
    },  # 7.34,
    "CMI_C11": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 8226.4,
        "center_wavelength": 8440.0,
        "max_wavelength": 8663.3,
    },  # 8.44,
    "CMI_C12": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 9423.3,
        "center_wavelength": 9610.0,
        "max_wavelength": 9800.1,
    },  # 9.61,
    "CMI_C13": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10177.1,
        "center_wavelength": 10330.0,
        "max_wavelength": 10481.1,
    },  # 10.33,
    "CMI_C14": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10815.5,
        "center_wavelength": 11190.0,
        "max_wavelength": 11603.6,
    },  # 11.19,
    "CMI_C15": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 11825.9,
        "center_wavelength": 12270.0,
        "max_wavelength": 12747.0,
    },  # 12.27,
    "CMI_C16": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 12990.4,
        "center_wavelength": 13270.0,
        "max_wavelength": 13559.3,
    },  # 13.27,
}


# HIMAWARI wavelengths in nanometers
HIMAWARI_WAVELENGTHS = {
    "B01": {
        "reso_og": 1000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 450.0,
        "center_wavelength": 470.0,
        "max_wavelength": 490.7,
    },
    "B02": {
        "reso_og": 500,
        "band_type": "TOA Reflectance",
        "min_wavelength": 495.1,
        "center_wavelength": 510.0,
        "max_wavelength": 525.9,
    },
    "B03": {
        "reso_og": 1000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 599.1,
        "center_wavelength": 640.0,
        "max_wavelength": 680.6,
    },
    "B04": {
        "reso_og": 2000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 839.1,
        "center_wavelength": 860.0,
        "max_wavelength": 873.5,
    },
    "B05": {
        "reso_og": 2000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 1589.6,
        "center_wavelength": 1600.0,
        "max_wavelength": 1630.3,
    },
    "B06": {
        "reso_og": 2000,
        "band_type": "TOA Reflectance",
        "min_wavelength": 2235.1,
        "center_wavelength": 2300.0,
        "max_wavelength": 2278.9,
    },
    "B07": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 3784.6,
        "center_wavelength": 3900.0,
        "max_wavelength": 3985.0,
    },
    "B08": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 5827.5,
        "center_wavelength": 6200.0,
        "max_wavelength": 6648.9,
    },
    "B09": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 6739.0,
        "center_wavelength": 6900.0,
        "max_wavelength": 7140.3,
    },
    "B10": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 7253.7,
        "center_wavelength": 7300.0,
        "max_wavelength": 7440.5,
    },
    "B11": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 8404.8,
        "center_wavelength": 8600.0,
        "max_wavelength": 8776.6,
    },
    "B12": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 9446.4,
        "center_wavelength": 9600.0,
        "max_wavelength": 9823.2,
    },
    "B13": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10193.7,
        "center_wavelength": 10400.0,
        "max_wavelength": 10612.3,
    },
    "B14": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 10909.9,
        "center_wavelength": 11200.0,
        "max_wavelength": 11576.8,
    },
    "B15": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 11900.5,
        "center_wavelength": 12400.0,
        "max_wavelength": 12865.0,
    },
    "B16": {
        "reso_og": 2000,
        "band_type": "TOA Normalised Brightness Temperature",
        "min_wavelength": 13003.9,
        "center_wavelength": 13300.0,
        "max_wavelength": 13564.8,
    },
}


def random_date(start, end):
    """
    Generate a random datetime between two datetime objects.
    """
    delta = end - start
    random_days = np.random.randint(0, delta.days + 1)
    return start + timedelta(days=random_days)


def random_time(start, end):
    """
    Generate a random time between two time objects.
    """
    start_minutes = (start.hour * 60) + (start.minute)
    end_minutes = (end.hour * 60) + (end.minute)
    random_minutes = np.random.randint(start_minutes, end_minutes + 1)
    return datetime(2000, 1, 1, random_minutes // 60, random_minutes % 60, 0)


def random_datetime(start, end):
    """
    Generate a random datetime between two datetime objects.
    """
    random_date_value = random_date(start, end)
    random_time_value = random_time(
        datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 23, 59, 00)
    )
    return datetime(
        random_date_value.year,
        random_date_value.month,
        random_date_value.day,
        random_time_value.hour,
        random_time_value.minute,
        random_time_value.second,
    )


def create_fov_mask(shape, fov_radius, patch_shape=None):
    """
    Function to create mask for specified field of view.
    """
    # Create coordinate grids
    y, x = np.ogrid[: shape[0], : shape[1]]
    # Calculate center points
    center_y, center_x = shape[0] // 2, shape[1] // 2
    # Calculate distance from center for each point
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    # Normalize distances by max possible distance (corner to center)
    max_dist = np.sqrt((center_x) ** 2 + (center_y) ** 2)
    normalized_dist = dist_from_center / max_dist
    # Create mask for specified field of view
    mask = normalized_dist <= fov_radius

    # If specified, ensure the mask also covers the patch size
    if patch_shape is not None:
        patch_shape_half_x = patch_shape[0] // 2 + 1
        patch_shape_half_y = patch_shape[1] // 2 + 1
        # Create a square mask for the patch size
        patch_mask = np.ones(shape, dtype=bool)
        # Mask out the corner area to keep everything beyond patch_size_half
        patch_mask[:patch_shape_half_x, :] = False
        patch_mask[-patch_shape_half_x:, :] = False
        patch_mask[:, :patch_shape_half_y] = False
        patch_mask[:, -patch_shape_half_y:] = False
        # Combine the two masks
        mask = mask & patch_mask
    return mask


def check_quality_flags_goes(ds):
    """
    Function to check quality flags in GOES data.
    0 --> good pixel quality
    1 --> conditionally usable pixel quality
    2 --> out of range pixel quality
    3 --> no value pixel quality
    4 --> focal plane temperature threshold exceeded pixel quality
    """
    # Check each channel individually - exit early if bad quality found
    for i in range(1, 17):
        if (ds[f"DQF_C{i:02d}"] > 0).any().item():
            logger.info(f"Did not pass quality check for channel DQF_C{i:02d}.")
            return False

    # If we get here, all channels passed the quality check
    # Also check whether there are any NaN or inf values in the dataset
    if (
        np.isnan(ds.x).any()
        or np.isinf(ds.x).any()
        or np.isnan(ds.y).any()
        or np.isinf(ds.y).any()
    ):
        logger.info("Dataset contains NaN or inf values in x or y.")
        return False
    # Check if any channels have NaN values
    for channel in ds.data_vars:
        if np.isnan(ds[channel].values).any():
            logger.info(f"Dataset contains NaN values in channel {channel}.")
            return False
    return True


def check_quality_flags_msg(ds, min_valid_fraction=0.999):
    """
    Function to check quality in MSG data.

    Args:
        ds (xarray.Dataset): The dataset to check.
        min_valid_fraction (float): Minimum fraction of valid data required for the dataset to pass the quality check.
            From experimentation, patches around the limb might have around 8000 to 10000 NaN values close to the disk edge
            even if the data on disk is valid. To not filter out all edge disk images, we emperically set the default to allow 0.1% of the data to be NaN.
    Returns:
        bool: True if the dataset passes the quality check, False otherwise.
    """
    channels = [
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
    # OPTION 1:
    # # create a mask where the latitude value is inf
    # mask_lat = ~np.isinf(ds.latitude)
    # mask_lon = ~np.isinf(ds.longitude)
    # # combine both masks to find all points with valid lat/lon values
    # mask = mask_lat & mask_lon
    # valid_pixels = np.count_nonzero(mask)

    # # loop through each channel to check for NaN values
    # for channel in channels:
    #     # Check for NaN only where mask is True, i.e. where the lat/lon values are valid
    #     nan_in_valid_region = np.isnan(ds[channel].values[mask])
    #     nan_fraction = np.count_nonzero(nan_in_valid_region) / valid_pixels

    #     # If the fraction of NaN values exceeds the allowed threshold, return False
    #     if nan_fraction > (1 - min_valid_fraction):
    #         return False
    # # If we get here, all channels passed the quality check
    # return True

    # OPTION 2:
    # check if any coordinates are NaN or inf
    if (
        np.isnan(ds.latitude).any()
        or np.isinf(ds.latitude).any()
        or np.isnan(ds.longitude).any()
        or np.isinf(ds.longitude).any()
    ):
        return False

    # loop through each channel to check for NaN values
    for channel in channels:
        # check if any values in the channel are NaN
        if np.isnan(ds[channel].values).any():
            return False
    return True


def check_quality_flags_himawari(ds, min_valid_fraction=0.999):
    """
    Function to check quality in HIMAWARI data.

    Args:
        ds (xarray.Dataset): The dataset to check.
        min_valid_fraction (float): Minimum fraction of valid data required for the dataset to pass the quality check.
            From experimentation, patches around the limb might have around 8000 to 10000 NaN values close to the disk edge
            even if the data on disk is valid. To not filter out all edge disk images, we emperically set the default to allow 0.1% of the data to be NaN.
    Returns:
        bool: True if the dataset passes the quality check, False otherwise.
    """
    channels = [
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
    ]

    ## OPTION 1:
    # # create a mask where the latitude value is inf
    # mask_lat = ~np.isinf(ds.latitude)
    # mask_lon = ~np.isinf(ds.longitude)
    # # combine both masks to find all points with valid lat/lon values
    # mask = mask_lat & mask_lon
    # valid_pixels = np.count_nonzero(mask)
    # loop through each channel to check for NaN values
    # for channel in channels:
    #     # Check for NaN only where mask is True, i.e. where the lat/lon values are valid
    #     nan_in_valid_region = np.isnan(ds[channel].values[mask])
    #     nan_fraction = np.count_nonzero(nan_in_valid_region) / valid_pixels

    #     # If the fraction of NaN values exceeds the allowed threshold, return False
    #     if nan_fraction > (1 - min_valid_fraction):
    #         return False
    # # If we get here, all channels passed the quality check
    # return True

    ## OPTION 2:
    # check if any coordinates are NaN or inf
    if (
        np.isnan(ds.latitude).any()
        or np.isinf(ds.latitude).any()
        or np.isnan(ds.longitude).any()
        or np.isinf(ds.longitude).any()
    ):
        return False

    # loop through each channel to check for NaN values
    for channel in channels:
        # check if any values in the channel are NaN
        if np.isnan(ds[channel].values).any():
            return False
    return True


class CenterWeightedCropDatasetEditor:
    def __init__(self, patch_shape, satellite, fov_radius=0.6, max_attempts=10):
        self.satellite = satellite
        self.patch_shape = patch_shape
        self.fov_radius = fov_radius
        self.max_attempts = max_attempts

    def __call__(self, ds):
        assert (
            ds["x"].shape[0] >= self.patch_shape[0]
        ), "Invalid dataset shape: %s" % str(ds["x"].shape)
        assert (
            ds["y"].shape[0] >= self.patch_shape[1]
        ), "Invalid dataset shape: %s" % str(ds["y"].shape)

        # get x/y grid
        x_grid, y_grid = np.meshgrid(
            np.arange(0, ds.x.shape[0], 1), np.arange(0, ds.y.shape[0], 1)
        )

        # create mask for valid coordinates within desired field of view
        # NOTE: This masks from the center to the image edge, rather than disk edge
        valid_mask = create_fov_mask(
            shape=(ds.x.shape[0], ds.y.shape[0]),
            fov_radius=self.fov_radius,
            patch_shape=self.patch_shape,
        )

        # get coordinate pairs for valid points
        coords_on_disk = np.column_stack((x_grid[valid_mask], y_grid[valid_mask]))
        del x_grid, y_grid

        attempts = 0
        while attempts <= self.max_attempts:
            # pick random x/y index
            random_idx = np.random.randint(0, len(coords_on_disk))
            x, y = tuple(coords_on_disk[random_idx])
            # define patch boundaries
            xmin = x - self.patch_shape[0] // 2
            ymin = y - self.patch_shape[1] // 2
            xmax = x + self.patch_shape[0] // 2
            ymax = y + self.patch_shape[1] // 2

            # crop patch
            patch_ds = ds.sel(
                {
                    "x": slice(ds["x"][xmin], ds["x"][xmax - 1]),
                    "y": slice(ds["y"][ymin], ds["y"][ymax - 1]),
                }
            )
            # check data quality flags
            if self.satellite.lower() == "goes":
                quality = check_quality_flags_goes(patch_ds)
            elif self.satellite.lower() == "msg":
                quality = check_quality_flags_msg(patch_ds)
            elif self.satellite.lower() == "himawari":
                quality = check_quality_flags_himawari(patch_ds)
            else:
                raise ValueError(f"Unknown satellite type: {self.satellite}")

            if quality == False:
                # logger.info('Found patch with bad quality flags, trying again ...')
                # try new set of indices
                attempts += 1
                continue
            else:
                # exit loop and return patch
                return patch_ds, xmin, ymin

        logger.info(
            "Could not find patch without bad quality flags after %d cropping attempts"
            % self.max_attempts
        )
        return None, None, None


def read_zipped_msg(filename, channels=None):
    """
    Function to read a zipped MSG file and return an xarray Dataset.

    Args:
        filename (str): Path to the zipped MSG file.
        channels (list, optional): List of channels to load. If None, defaults to a
            predefined list of channels.
    Returns:
        xarray.Dataset: Dataset containing the MSG data for the specified channels.
    """
    if channels is None:
        channels = [
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

    zf = zipfile.ZipFile(filename)

    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        seviri_file = list(pathlib.Path(tempdir).glob("MSG*-NA.nat"))[0]
        scn = Scene([seviri_file], reader="seviri_l1b_native")
        scn.load(channels)
        msg_ds = scn.to_xarray().load()

    return msg_ds


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
            add_offset=input_dtype(-valid_range.min()),
            _FillValue=max_value,
            dtype=str(target_dtype.__name__),
        )
    elif np.issubdtype(target_dtype, np.integer):
        return dict(
            scale_factor=input_dtype(range / max_value),
            add_offset=input_dtype(-valid_range.min()),
            _FillValue=min_value,
            dtype=str(target_dtype.__name__),
        )


def get_satellite_viewing_angles(
    lat: np.ndarray,
    lon: np.ndarray,
    sat_lat: float,
    sat_lon: float,
    sat_alt: float,  # in km
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate satellite zenith and azimuth angles.

    Satellite zenith angle measures the angle from vertical that an observation
    is made at the surface. 0 means that the satellite is directly overhead. 90
    means that the surface point is on the horizon of the satellite view.

    Satellite azimuth angle measures the angle from North from the surface point
    to the satellite, measured clockwise. 0 is due N, 90 is E, 180 is S and 270
    is W.

    Parameters
    ----------
    lat : np.ndarray
        latitudes of surface point in degrees
    lon : np.ndarray
        longitudes of surface point in degrees
    sat_lat : float, optional
        latitude of sub-satellite point in degrees, by default 0
    sat_lon : float, optional
        longitude of sub-satellite point in degrees, by default 0
    sat_alt : float, optional
        altitude of satellite in km, by default 35_793 (geostationary orbit
        height over average earth radius)

    Returns
    -------
    tuple[float, float]
        satellite zenith and azimuth angles in degrees
    """
    # TODO test for inf / nan coordinates
    # Approximate spherical Earth so use radius of 6,371 km
    Re = 6_371
    Rgeo = sat_alt + Re

    # Caclulate the beta angle
    cos_beta = np.cos(np.radians(lat - sat_lat)) * np.cos(np.radians(lon - sat_lon))
    sin_beta = np.sin(np.arccos(cos_beta))

    # Calculate satellite zenith angle
    geo_dist = (
        Rgeo**2 + Re**2 - 2 * Rgeo * Re * cos_beta
    ) ** 0.5  # distance from surface to satellite
    sin_theta = (Rgeo * sin_beta) / geo_dist
    zenith_angle = np.degrees(np.arcsin(sin_theta))

    # Find where satellite-surface path intersects the earth and make these > 90
    zenith_angle = np.where(
        geo_dist**2 < (Rgeo**2 - Re**2), zenith_angle, 180 - zenith_angle
    )

    # Calculate satellite azimuthal angle
    x_sat = np.cos(np.radians(lat - sat_lat)) * np.sin(np.radians(lon - sat_lon))
    y_sat = np.sin(np.radians(lat - sat_lat))
    azimuth_angle = np.where(
        np.isfinite(x_sat), np.degrees(np.arctan2(x_sat, y_sat)) % 360, np.nan
    )

    return zenith_angle, azimuth_angle


def get_sza_and_azi(
    date: datetime, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get the solar zenith angle at a specific time/lat/lon

    Parameters
    ----------
    date : datetime | array of datetime like
        Dates of the points
    lat : np.ndarray
        Latitudes
    lon : np.ndarray
        Longitudes

    Returns
    -------
    sza: np.ndarray
        The solar zenith angle in degrees, where 0 is directly above, 90 is on
        the horizon and 180 is directly below
    saa: np.ndarray
        The solar azimuth angle in degrees, clockwise from North
    """
    # TODO test for inf / nan coordinates
    try:
        date = pd.DatetimeIndex(date)
    except TypeError:
        date = pd.DatetimeIndex([date])
    day_of_year = date.dayofyear.to_numpy()
    hour_of_day = (date.hour + date.minute / 60 + date.second / 60 / 60).to_numpy()

    # calculate approx time equation as angle for 365 day year
    equation_of_time_approx = 2.0 * np.pi * day_of_year / 365.0

    # calculate the solar declination for the given day
    # the declination varies due to the fact that the earth rotation axis
    # is not perpendicular to the ecliptic plane
    solar_declination = (
        0.006918
        - 0.399912 * np.cos(equation_of_time_approx)
        - 0.006758 * np.cos(2.0 * equation_of_time_approx)
        - 0.002697 * np.cos(3.0 * equation_of_time_approx)
        + 0.070257 * np.sin(equation_of_time_approx)
        + 0.000907 * np.sin(2.0 * equation_of_time_approx)
        + 0.001480 * np.sin(3.0 * equation_of_time_approx)
    )

    # equation of time, used to compensate for the earth's elliptical orbit
    # around the sun and its axial tilt when calculating solar time
    # eqt is the correction in hours
    equation_of_time = 2.0 * np.pi * day_of_year / 366.0
    equation_of_time = (
        0.0072 * np.cos(equation_of_time)
        - 0.0528 * np.cos(2.0 * equation_of_time)
        - 0.0012 * np.cos(3.0 * equation_of_time)
        - 0.1229 * np.sin(equation_of_time)
        - 0.1565 * np.sin(2.0 * equation_of_time)
        - 0.0041 * np.sin(3.0 * equation_of_time)
    )

    # calculate the solar zenith angle
    omega = np.radians(
        (360.0 / 24.0) * (hour_of_day + lon / 15.0 + equation_of_time - 12.0)
    )
    sunh = np.sin(solar_declination) * np.sin(np.radians(lat)) + np.cos(
        solar_declination
    ) * np.cos(np.radians(lat)) * np.cos(omega)

    solar_elevation = np.arcsin(np.clip(sunh, -1, 1))
    solar_zenith_angle = np.pi / 2.0 - solar_elevation

    # Solar azimuth added by yaswant
    azimuth = (
        np.sin(solar_declination) * np.cos(np.radians(lat))
        - np.cos(solar_declination) * np.sin(np.radians(lat)) * np.cos(omega)
    ) / np.cos(np.pi / 2.0 - solar_zenith_angle)

    solar_azimuth_angle = np.arccos(np.clip(azimuth, -1, 1))

    return np.degrees(solar_zenith_angle), np.degrees(solar_azimuth_angle)


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


def get_abi_lat_lon(
    dataset: xr.Dataset, dtype: type = float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns latitude and longitude for each location in an ABI dataset
    """
    p = get_abi_proj(dataset)
    xx, yy = np.meshgrid(
        (
            dataset.x.data * dataset.goes_imager_projection.perspective_point_height
        ).astype(dtype),
        (
            dataset.y.data * dataset.goes_imager_projection.perspective_point_height
        ).astype(dtype),
    )
    lons, lats = p(xx, yy, inverse=True)
    lons[lons >= 1e30] = np.nan
    lats[lats >= 1e30] = np.nan
    return lats, lons
