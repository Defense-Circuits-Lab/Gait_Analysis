# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_twoD_02_preprocess.ipynb.

# %% auto 0
__all__ = ['get_preprocessing_relevant_marker_ids', 'get_all_unique_marker_ids', 'smooth_tracked_coords_and_likelihood',
           'interpolate_low_likelihood_intervals', 'get_low_likelihood_interval_border_idxs',
           'add_new_marker_derived_from_existing_markers', 'get_corner_coords_with_likelihoods',
           'get_most_reliable_marker_position_with_likelihood', 'get_translation_vector',
           'evaluate_maze_shape_using_open_corners', 'compute_error_proportion', 'compute_angle_error',
           'compute_distance_ratio_error', 'get_distance_between_two_points', 'get_conversion_factor_px_to_cm',
           'get_rotation_angle_with_open_corner', 'get_rotation_angle_with_closed_corners_only', 'normalize_df',
           'translate_df', 'rotate_df', 'convert_df_to_cm', 'create_bodypart_objects']

# %% ../../nbs/01_twoD_02_preprocess.ipynb 3
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import math

from ..core import Bodypart
from . import utils

# %% ../../nbs/01_twoD_02_preprocess.ipynb 4
def get_preprocessing_relevant_marker_ids(
    df: pd.DataFrame,  # DataFrame with x, y, and likelihood for tracked marker_ids
    marker_ids_to_exclude: Optional[
        List[str]
    ] = None,  # list of marker_ids to exclude; optional default None
) -> List[str]:
    all_marker_ids = get_all_unique_marker_ids(df=df)
    relevant_marker_ids = all_marker_ids
    if marker_ids_to_exclude != None:
        for marker_id_to_exclude in marker_ids_to_exclude:
            if marker_id_to_exclude in relevant_marker_ids:
                relevant_marker_ids.remove(marker_id_to_exclude)
    return relevant_marker_ids

# %% ../../nbs/01_twoD_02_preprocess.ipynb 5
def get_all_unique_marker_ids(df: pd.DataFrame) -> List[str]:
    unique_marker_ids = []
    for column_name in df.columns:
        marker_id, _ = column_name.split("_")
        if marker_id not in unique_marker_ids:
            unique_marker_ids.append(marker_id)
    return unique_marker_ids

# %% ../../nbs/01_twoD_02_preprocess.ipynb 6
def smooth_tracked_coords_and_likelihood(
    df: pd.DataFrame,  # DataFrame to smooth
    window_length: int,  # Odd integer (!) of sliding window size in frames to consider for smoothing
    marker_ids: List[str] = [
        "all"
    ],  # List of markers that will be smoothed; optional default ['all'] to smooth all marker_ids
    polyorder: int = 3,  # Order of the polynom used for the savgol filter
) -> pd.DataFrame:
    """
    Smoothes the DataFrame basically using the implementation from DLC2kinematics:
    https://github.com/AdaptiveMotorControlLab/DLC2Kinematics/blob/82e7e60e00e0efb3c51e024c05a5640c91032026/src/dlc2kinematics/preprocess.py#L64
    However, with one key change: likelihoods will also be smoothed.
    In addition, we will not smooth the columns for the tracked LEDs and the MazeCorners.

    Note: window_length has to be an odd integer!
    """
    smoothed_df = df.copy()
    column_names = utils.get_column_names(
        df=smoothed_df,
        column_identifiers=["x", "y", "likelihood"],
        marker_ids=marker_ids,
    )
    column_idxs_to_smooth = smoothed_df.columns.get_indexer(column_names)
    smoothed_df.iloc[:, column_idxs_to_smooth] = savgol_filter(
        x=smoothed_df.iloc[:, column_idxs_to_smooth],
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
    )
    return smoothed_df

# %% ../../nbs/01_twoD_02_preprocess.ipynb 7
def interpolate_low_likelihood_intervals(
    df: pd.DataFrame,
    marker_ids: List[str],
    max_interval_length: int,
    framerate: float,
) -> pd.DataFrame:
    interpolated_df = df.copy()
    for marker_id in marker_ids:
        low_likelihood_interval_border_idxs = get_low_likelihood_interval_border_idxs(
            likelihood_series=interpolated_df[f"{marker_id}_likelihood"],
            max_interval_length=max_interval_length,
            framerate=framerate,
        )
        for start_idx, end_idx in low_likelihood_interval_border_idxs:
            if (start_idx - 1 >= 0) and (end_idx + 2 < interpolated_df.shape[0]):
                interpolated_df[f"{marker_id}_x"][
                    start_idx - 1 : end_idx + 2
                ] = interpolated_df[f"{marker_id}_x"][
                    start_idx - 1 : end_idx + 2
                ].interpolate()
                interpolated_df[f"{marker_id}_y"][
                    start_idx - 1 : end_idx + 2
                ] = interpolated_df[f"{marker_id}_y"][
                    start_idx - 1 : end_idx + 2
                ].interpolate()
                interpolated_df[f"{marker_id}_likelihood"][
                    start_idx : end_idx + 1
                ] = 0.5
    return interpolated_df

# %% ../../nbs/01_twoD_02_preprocess.ipynb 8
def get_low_likelihood_interval_border_idxs(
    likelihood_series: pd.Series,
    framerate: float,
    max_interval_length: int,
    min_likelihood_threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    all_low_likelihood_idxs = np.where(
        likelihood_series.values < min_likelihood_threshold
    )[0]
    short_low_likelihood_interval_border_idxs = utils.get_interval_border_idxs(
        all_matching_idxs=all_low_likelihood_idxs,
        framerate=framerate,
        max_interval_duration=max_interval_length * framerate,
    )
    return short_low_likelihood_interval_border_idxs

# %% ../../nbs/01_twoD_02_preprocess.ipynb 9
def add_new_marker_derived_from_existing_markers(
    df: pd.DataFrame,
    existing_markers: List[str],
    new_marker_id: str,
    likelihood_threshold: float = 0.5,
) -> None:
    df_with_new_marker = df.copy()
    for coordinate in ["x", "y"]:
        df_with_new_marker[f"{new_marker_id}_{coordinate}"] = (
            sum(
                [
                    df_with_new_marker[f"{marker_id}_{coordinate}"]
                    for marker_id in existing_markers
                ]
            )
        ) / len(existing_markers)
    df_with_new_marker[f"{new_marker_id}_likelihood"] = 0
    row_idxs_where_all_likelihoods_exceeded_threshold = (
        utils.get_idxs_where_all_markers_exceed_likelihood(
            df=df_with_new_marker, marker_ids=existing_markers, likelihood_threshold=0.5
        )
    )
    df_with_new_marker.iloc[row_idxs_where_all_likelihoods_exceeded_threshold, -1] = 1
    return df_with_new_marker

# %% ../../nbs/01_twoD_02_preprocess.ipynb 10
def get_corner_coords_with_likelihoods(df: pd.DataFrame) -> Dict:
    corner_coords_with_likelihood = {}
    for corner_marker_id in [
        "MazeCornerClosedRight",
        "MazeCornerClosedLeft",
        "MazeCornerOpenRight",
        "MazeCornerOpenLeft",
    ]:
        xy_coords, min_likelihood = get_most_reliable_marker_position_with_likelihood(
            df=df, marker_id=corner_marker_id
        )
        corner_coords_with_likelihood[corner_marker_id] = {
            "coords": xy_coords,
            "min_likelihood": min_likelihood,
        }
    return corner_coords_with_likelihood

# %% ../../nbs/01_twoD_02_preprocess.ipynb 11
def get_most_reliable_marker_position_with_likelihood(
    df: pd.DataFrame, marker_id: str, percentile: float = 99.95
) -> Tuple[np.array, float]:
    likelihood_threshold = np.nanpercentile(
        df[f"{marker_id}_likelihood"].values, percentile
    )
    df_most_reliable_frames = df.loc[
        df[f"{marker_id}_likelihood"] >= likelihood_threshold
    ].copy()
    most_reliable_x, most_reliable_y = (
        df_most_reliable_frames[f"{marker_id}_x"].median(),
        df_most_reliable_frames[f"{marker_id}_y"].median(),
    )
    return np.array([most_reliable_x, most_reliable_y]), likelihood_threshold

# %% ../../nbs/01_twoD_02_preprocess.ipynb 12
def get_translation_vector(coords_to_become_origin: np.ndarray) -> np.ndarray:
    return -coords_to_become_origin

# %% ../../nbs/01_twoD_02_preprocess.ipynb 13
def evaluate_maze_shape_using_open_corners(
    corners_and_likelihoods: Dict, tolerance: float
) -> Dict:
    best_result = {
        "valid": False,
        "mean_error": tolerance + 1,
        "open_corner_id": None,
        "side_id": None,
    }
    all_open_corner_marker_ids = [
        corner_marker_id
        for corner_marker_id in corners_and_likelihoods.keys()
        if "Open" in corner_marker_id
    ]
    for open_corner_marker_id in all_open_corner_marker_ids:
        valid_positions = False
        side_id = open_corner_marker_id[open_corner_marker_id.find("Open") + 4 :]
        if side_id == "Left":
            opposite_side_id = "Right"
        else:
            opposite_side_id = "Left"
        closed_corner_opposite_side = f"MazeCornerClosed{opposite_side_id}"
        angle_error = compute_angle_error(
            a=corners_and_likelihoods[f"MazeCornerClosed{opposite_side_id}"]["coords"],
            b=corners_and_likelihoods[f"MazeCornerClosed{side_id}"]["coords"],
            c=corners_and_likelihoods[open_corner_marker_id]["coords"],
        )
        distance_ratio_error = compute_distance_ratio_error(
            corners_and_likelihoods=corners_and_likelihoods,
            open_corner_marker_id=open_corner_marker_id,
            side_id=side_id,
        )
        if (angle_error <= tolerance) & (distance_ratio_error <= tolerance):
            valid_positions = True
        mean_error = (angle_error + distance_ratio_error) / 2
        if mean_error < best_result["mean_error"]:
            best_result["valid"] = valid_positions
            best_result["mean_error"] = mean_error
            best_result["open_corner_id"] = open_corner_marker_id
            best_result["side_id"] = side_id
    return best_result

# %% ../../nbs/01_twoD_02_preprocess.ipynb 14
def compute_error_proportion(query_value: float, target_value: float) -> float:
    return abs(query_value - target_value) / target_value

# %% ../../nbs/01_twoD_02_preprocess.ipynb 15
def compute_angle_error(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # b is point at the joint that connects the other two
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return compute_error_proportion(query_value=angle, target_value=90)

# %% ../../nbs/01_twoD_02_preprocess.ipynb 16
def compute_distance_ratio_error(
    corners_and_likelihoods: Dict, open_corner_marker_id: str, side_id: str
) -> float:
    maze_width = get_distance_between_two_points(
        corners_and_likelihoods["MazeCornerClosedLeft"]["coords"],
        corners_and_likelihoods["MazeCornerClosedRight"]["coords"],
    )
    maze_length = get_distance_between_two_points(
        corners_and_likelihoods[f"MazeCornerClosed{side_id}"]["coords"],
        corners_and_likelihoods[open_corner_marker_id]["coords"],
    )
    distance_ratio = maze_length / maze_width
    return compute_error_proportion(query_value=distance_ratio, target_value=50 / 4)

# %% ../../nbs/01_twoD_02_preprocess.ipynb 17
def get_distance_between_two_points(
    coords_point_a: np.ndarray, coords_point_b: np.ndarray
) -> float:
    return (
        (coords_point_a[0] - coords_point_b[0]) ** 2
        + (coords_point_a[1] - coords_point_b[1]) ** 2
    ) ** 0.5

# %% ../../nbs/01_twoD_02_preprocess.ipynb 18
def get_conversion_factor_px_to_cm(
    coords_point_a: np.ndarray, coords_point_b: np.ndarray, distance_in_cm: float
) -> float:
    distance = get_distance_between_two_points(coords_point_a, coords_point_b)
    return distance_in_cm / distance

# %% ../../nbs/01_twoD_02_preprocess.ipynb 19
def get_rotation_angle_with_open_corner(
    corners: Dict,
    side_id: str,
    translation_vector: np.ndarray,
    conversion_factor: float,
) -> float:
    """
    Function, that calculates the rotation angle of the maze considering the best matching open corner
    and the corresponding closed corner on the same side.

    Returns:
        float: angle in radians
    """
    if side_id == "Left":
        side_specific_y = 0
    else:
        side_specific_y = 4
    translated_closed_corner = (
        corners[f"MazeCornerClosed{side_id}"]["coords"] + translation_vector
    )
    translated_open_corner = (
        corners[f"MazeCornerOpen{side_id}"]["coords"] + translation_vector
    )
    target_rotated_open_corner = np.asarray(
        [50 / conversion_factor, side_specific_y / conversion_factor]
    )
    length_a = (
        get_distance_between_two_points(
            translated_open_corner, target_rotated_open_corner
        )
        * conversion_factor
    )
    length_b = (
        get_distance_between_two_points(
            translated_open_corner, translated_closed_corner
        )
        * conversion_factor
    )
    length_c = 50
    angle = math.acos(
        (length_b**2 + length_c**2 - length_a**2) / (2 * length_b * length_c)
    )
    return angle

# %% ../../nbs/01_twoD_02_preprocess.ipynb 20
def get_rotation_angle_with_closed_corners_only(
    corners: Dict, translation_vector: np.ndarray, conversion_factor: float
) -> float:
    translated_closed_left = (
        corners["MazeCornerClosedLeft"]["coords"] + translation_vector
    )
    translated_closed_right = (
        corners["MazeCornerClosedRight"]["coords"] + translation_vector
    )
    target_rotated_closed_right = np.asarray([0, 4 / conversion_factor])

    length_a = (
        get_distance_between_two_points(
            translated_closed_right, target_rotated_closed_right
        )
        * conversion_factor
    )
    length_b = (
        get_distance_between_two_points(translated_closed_left, translated_closed_right)
        * conversion_factor
    )
    length_c = 4
    angle = math.acos(
        (length_b**2 + length_c**2 - length_a**2) / (2 * length_b * length_c)
    )
    return angle

# %% ../../nbs/01_twoD_02_preprocess.ipynb 21
def normalize_df(df: pd.DataFrame, normalization_parameters) -> None:
    unadjusted_df = df.copy()
    translated_df = translate_df(
        df=unadjusted_df,
        translation_vector=normalization_parameters["translation_vector"],
    )
    rotated_and_translated_df = rotate_df(
        df=translated_df, rotation_angle=normalization_parameters["rotation_angle"]
    )
    final_df = convert_df_to_cm(
        df=rotated_and_translated_df,
        conversion_factor=normalization_parameters["conversion_factor"],
    )
    return final_df

# %% ../../nbs/01_twoD_02_preprocess.ipynb 22
def translate_df(df: pd.DataFrame, translation_vector: np.array) -> pd.DataFrame:
    for marker_id in get_all_unique_marker_ids(df=df):
        df.loc[:, [f"{marker_id}_x", f"{marker_id}_y"]] += translation_vector
    return df

# %% ../../nbs/01_twoD_02_preprocess.ipynb 23
def rotate_df(
    df: pd.DataFrame,  # DataFrame with 2D coordinates to be rotated
    rotation_angle: float,  # rotation angle in radians
) -> pd.DataFrame:
    df_rotated = df.copy()
    cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
    for marker_id in get_all_unique_marker_ids(df=df):
        df_rotated[f"{marker_id}_x"] = (
            df[f"{marker_id}_x"] * cos_theta - df[f"{marker_id}_y"] * sin_theta
        )
        df_rotated[f"{marker_id}_y"] = (
            df[f"{marker_id}_x"] * sin_theta + df[f"{marker_id}_y"] * cos_theta
        )
    return df_rotated

# %% ../../nbs/01_twoD_02_preprocess.ipynb 24
def convert_df_to_cm(df: pd.DataFrame, conversion_factor: float) -> pd.DataFrame:
    for marker_id in get_all_unique_marker_ids(df=df):
        df.loc[:, [f"{marker_id}_x", f"{marker_id}_y"]] *= conversion_factor
    return df

# %% ../../nbs/01_twoD_02_preprocess.ipynb 25
def create_bodypart_objects(normalized_df: pd.DataFrame, fps: int) -> Dict:
    bodyparts = {}
    for marker_id in get_all_unique_marker_ids(df=normalized_df):
        bodyparts[marker_id] = Bodypart(
            bodypart_id=marker_id, df=normalized_df, fps=fps
        )
    return bodyparts