# functions that load and operate on the data saved in the dataframes,
# for example, from the computed contour, calculate area, centroid, perimeter,
# percentage of fluo area, onset of gene expression, counting poles, ....

import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def get_file_list(folder_path, ending=''):
    """
    Retrieve a list of files with a specific ending from a given folder. The function
    provides both a relative and an absolute file path list of the files that match
    the specified ending. The files are sorted alphabetically.

    :param folder_path: Path to the directory containing files.
        The path should end with a delimiter (e.g., slash for Linux/macOS
        or backslash for Windows).
    :type folder_path: str
    :param ending: File ending to filter by. Default is an empty string,
        which retrieves all files in the folder.
    :type ending: str
    :return: A tuple containing two lists:
        - A list of file names (relative paths) that match the ending.
        - A list of absolute paths to these files in the specified folder.
    :rtype: tuple[list[str], list[str]]
    """
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(ending)])
    abs_file_list = [folder_path + f for f in file_list]
    return file_list, abs_file_list

def load_processed_df(path, all=True):
    """
    Loads and processes a JSON file into a pandas DataFrame. The function reads the JSON file
    from the specified file path and attempts to create a DataFrame from the data. If the
    `all` parameter is set to False, the DataFrame is filtered to include only rows where the
    'Flag' column has the value 'Keep'. The total number of rows in the resulting DataFrame is
    printed.

    :param path: The file path to the JSON file to be loaded.
    :type path: str
    :param all: Boolean flag indicating whether to keep all rows in the DataFrame or filter
        rows where the 'Flag' column has the value 'Keep'. Default is True.
    :type all: bool
    :return: A pandas DataFrame created from the contents of the JSON file. If `all` is False,
        only rows with the 'Keep' flag are included.
    :rtype: pd.DataFrame
    """
    with open(path) as project_file:
        df = json.load(project_file)
        try:
            df = pd.DataFrame(df["data"], columns=df["columns"])
        except:
            try:
                df = pd.DataFrame(df["data"])
            except:
                df = pd.DataFrame.from_dict(df)
        if not all:
            df = df[df['Flag'] == 'Keep']
        print('# rows: ', len(df))

    return df

def filter_outliers(outlier_dict, df, label='outlier'):
    """
    Filters outliers from a DataFrame based on the provided dictionary of outlier keys and
    updates a specified flag column for matching records.

    This function iterates through the `outlier_dict`, constructs a list of outlier identifiers,
    and checks if the 'imageID' field in the DataFrame matches those identifiers. If a match is
    found, it assigns the provided label to a 'Flag' column for the corresponding rows.

    :param outlier_dict: A dictionary where keys represent a base identifier and values are lists of
        suffixes. Each key-suffix combination is used to construct full identifiers of outliers.
    :type outlier_dict: dict
    :param df: The DataFrame containing the data records, including the 'imageID' column, which will
        be matched with constructed outlier identifiers.
    :type df: pandas.DataFrame
    :param label: The value to assign to the 'Flag' column for rows identified as outliers. Defaults
        to 'outlier'.
    :type label: str
    :return: The updated DataFrame, where the 'Flag' column is modified to include the `label` for
        rows flagged as outliers.
    """
    for k, v in outlier_dict.items():
        outliers = [k + '_S' + o for o in v]
        index = df[df['imageID'].isin(outliers)].index
        df.loc[index, 'Flag'] = label

    return df

def convert_units_to_um(df, args=[]):
    """
    Convert specific measurement values to micrometer (μm) or other units by applying conversion
    factors based on the 'um_per_pixel' value present in the dataframe.
    This function creates new columns in the dataframe for the converted values.

    :param df: Input dataframe containing measurement data and the conversion factor
        'um_per_pixel'.
    :type df: pandas.DataFrame
    :param args: List of measurement types to convert. Possible values include:
        'Length_MA', 'Volume_MA', 'Areas', 'cnt_Area', 'cnt_Perimeter', and
        'centroid_dist'. Each value adds respective converted columns.
    :type args: list[str], optional
    :return: The dataframe with additional columns containing converted values.
    :rtype: pandas.DataFrame
    """
    if 'Length_MA' in args:
        df['Length_MA_um'] = df['Length_MA'] * df['um_per_pixel']
    if 'Volume_MA' in args:
        df['Volume_MA_mm3'] = df['Volume_MA'] * df['um_per_pixel'] ** 3 / (10 ** 9)
        df['Volume_eq_mm3'] = df['Length_MA'] ** 3 * np.pi / 6 * df['um_per_pixel'] ** 3 / (10 ** 9)
    if 'Areas' in args:
        df['Area_um2'] = df['Areas'].apply(np.sum)
        df['Area_um2'] = df['Area_um2'] * df['um_per_pixel'] ** 2
    if 'cnt_Area' in args:
        df['cnt_Area_um2'] = df['cnt_Area'] * df['um_per_pixel'] ** 2
    if 'cnt_Perimeter' in args:
        df['cnt_Perimeter_um'] = df['cnt_Perimeter'] * df['um_per_pixel']
    if 'centroid_dist' in args:
        df['centroid_dist_um'] = df['centroid_dist'] * df['um_per_pixel']

    return df


# clustering functions for time series data
def find_onset_time_cost_function(time_values, data):
    """
    Find the onset time based on minimizing the cost function C = sigma1^2 + sigma2^2.
    :param time_values: Array of time points
    :param data: Array of data points corresponding to time points
    :return: Onset time where the cost function is minimal, cost values, sigma1 squared, sigma2 squared
    """
    # Initialize arrays to store cost values and variances with NaNs
    cost_values = np.full(len(time_values), np.nan)
    sigma1_squared_values = np.full(len(time_values), np.nan)
    sigma2_squared_values = np.full(len(time_values), np.nan)

    # Loop through each time point except the first two and last two
    for t_boundary in range(2, len(time_values) - 2):
        # Split data into two parts
        data1 = data[:t_boundary]
        data2 = data[t_boundary:]

        # Calculate variances for each segment
        sigma1_squared = np.var(data1) #np.nanvar(data1) for individual curves, to ignore nan when calculating the variance
        sigma2_squared = np.var(data2) ##np.nanvar(data2) for individual curves, to ignore nan when calculating the variance

        # Calculate cost function
        cost = sigma1_squared + sigma2_squared

        # Fill the arrays with calculated values
        cost_values[t_boundary] = cost
        sigma1_squared_values[t_boundary] = sigma1_squared
        sigma2_squared_values[t_boundary] = sigma2_squared

    # Find index of minimum cost value, ignoring NaNs
    min_cost_index = np.nanargmin(cost_values)

    # The corresponding onset time
    onset_time = time_values[min_cost_index]

    return onset_time, cost_values, sigma1_squared_values, sigma2_squared_values

def find_onset_time_midpoint(time_values, data, weights, s_factor=0.5):
    # Fit a spline to the data
    spline = UnivariateSpline(time_values, data, w=weights, s=s_factor)

    # Calculate the spline values
    spline_values = spline(time_values)

    # Find the minimum and maximum values of the spline
    min_val = np.min(spline_values)
    max_val = np.max(spline_values)

    # Determine the midpoint value
    midpoint_value = (min_val + max_val) / 2

    # Find the index where the spline first reaches the midpoint value
    onset_idx = np.where(spline_values >= midpoint_value)[0][0]

    # Determine onset time
    onset_time = time_values[onset_idx]

    return onset_time, spline, spline_values, midpoint_value



# operations to compute profiles IF experiment
def div_by_area(df, channels=['DAPI']):
    """
    Normalizes the intensity values in the specified channels by dividing them
    by the 'Areas' column values. This operation is performed for each provided
    channel, and new columns are created with the names formatted as
    "<channel>_area_norm". The function modifies the DataFrame in-place and
    returns the updated DataFrame.

    :param df:
        The pandas DataFrame containing the data to be processed. Must include
        the specified channels and an 'Areas' column.
    :param channels:
        A list of string channel names for which normalization by area
        will be applied. Defaults to ['DAPI'].
    :return:
        The updated DataFrame with new columns added for the normalized values.
    """
    for ch in channels:
        newcol = ch + '_' + 'area_norm'
        print(newcol)
        df[newcol] = None
        dff = df[(df[ch].notnull()) & (df['Areas'].notnull())]
        dff[newcol] = dff.apply(lambda x: np.divide(x[ch], x['Areas']), axis=1)
        df.update(dff)
    return df

def get_profile_orientation(df, orientation_channel=None, method='median'):
    """
    Determines the orientation of profiles in the provided DataFrame based on the selected
    method. The function evaluates a specified orientation channel, or determines one
    automatically if not provided, and processes the profiles to assess whether they should
    be flipped or not.

    :param df: The input pandas DataFrame containing the profile data and channel information.
    :param orientation_channel: Optional; The specific channel to evaluate orientation. If not provided, the channel with index 0
        in the last row's `channels` dictionary will be used.
    :param method: Optional; The calculation method for determining orientation. Can be 'median' or 'mean'.
        Default is 'median'.
    :return: A list indicating the orientation for each profile. Values are 1 if flipping is required,
        0 otherwise, and None for profiles where orientation could not be determined.
    :rtype: list
    """
    channel_dict = df.iloc[-1]['channels']

    if not orientation_channel:
        orientation_channel = list(channel_dict.keys())[list(channel_dict.values()).index(0)]

    flip = list()
    for i, profile in enumerate(df[f'{orientation_channel}_area_norm']):
        try:
            profile = profile[round(len(profile) * 0.1):round(len(profile) * 0.9)]
            profile = np.array(profile, dtype=float)

            if method == 'median':
                half1 = np.median(profile[: round(len(profile) / 3)])
                half2 = np.median(profile[round(len(profile) / 3) * 2:])

            if method == 'mean':
                half1 = np.mean(profile[: round(len(profile) / 3)])
                half2 = np.mean(profile[round(len(profile) / 3) * 2:])

            if half1 < half2:
                flip.append(1)
            else:
                flip.append(0)
        except:
            flip.append(None)
    return flip
