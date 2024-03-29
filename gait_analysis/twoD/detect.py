# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_twoD_03_detect.ipynb.

# %% auto 0
__all__ = ['create_behavior_df', 'add_orientation_to_behavior_df', 'add_immobility_based_on_several_bodyparts_to_behavior_df',
           'get_immobility_related_events', 'create_event_objects', 'add_event_bouts_to_behavior_df', 'get_gait_events',
           'get_gait_disruption_events', 'get_interval_border_idxs_from_event_type_and_id']

# %% ../../nbs/01_twoD_03_detect.ipynb 3
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from ..core import EventBout
from .. import twoD

# %% ../../nbs/01_twoD_03_detect.ipynb 4
def create_behavior_df(
    normalized_df: pd.DataFrame, bodyparts_to_include: List[str]
) -> pd.DataFrame:
    column_names = twoD.utils.get_column_names(
        df=normalized_df,
        column_identifiers=["x", "y", "likelihood"],
        marker_ids=bodyparts_to_include,
    )
    return normalized_df[column_names].copy()

# %% ../../nbs/01_twoD_03_detect.ipynb 5
def add_orientation_to_behavior_df(
    behavior_df: pd.DataFrame,
    all_bodyparts: Dict,
    bodyparts_for_direction_front_to_back: List[str],
) -> pd.DataFrame:
    assert (
        len(bodyparts_for_direction_front_to_back) == 2
    ), '"bodyparts_for_direction_front_to_back" must be a list of exact 2 marker_ids!'
    front_marker_id = bodyparts_for_direction_front_to_back[0]
    back_marker_id = bodyparts_for_direction_front_to_back[1]
    behavior_df.loc[
        all_bodyparts[front_marker_id].df["x"] > all_bodyparts[back_marker_id].df["x"],
        "facing_towards_open_end",
    ] = True
    return behavior_df

# %% ../../nbs/01_twoD_03_detect.ipynb 6
def add_immobility_based_on_several_bodyparts_to_behavior_df(
    behavior_df: pd.DataFrame,
    all_bodyparts: Dict,
    bodyparts_critical_for_freezing: List[str],
) -> pd.DataFrame:

    # ToDo: Shares some code witht he "filter_dataframe" function, can it be reused here?
    #       However, here we iterate through several dfs and use the shared indices across
    #       These dataframes, so the behavior is different and adaptations would be required.
    valid_idxs_per_marker_id = []
    for bodypart_id in bodyparts_critical_for_freezing:
        tmp_df = all_bodyparts[bodypart_id].df.copy()
        valid_idxs_per_marker_id.append(
            tmp_df.loc[tmp_df["immobility"] == True].index.values
        )
    shared_valid_idxs_for_all_markers = valid_idxs_per_marker_id[0]
    if len(valid_idxs_per_marker_id) > 1:
        for next_set_of_valid_idxs in valid_idxs_per_marker_id[1:]:
            shared_valid_idxs_for_all_markers = np.intersect1d(
                shared_valid_idxs_for_all_markers, next_set_of_valid_idxs
            )
    behavior_df.loc[shared_valid_idxs_for_all_markers, "immobility"] = True
    return behavior_df

# %% ../../nbs/01_twoD_03_detect.ipynb 7
def get_immobility_related_events(
    behavior_df: pd.DataFrame, fps: float, min_interval_duration: float, event_type: str
) -> List[EventBout]:
    all_immobility_idxs = np.where(behavior_df["immobility"].values == True)[0]
    immobility_interval_border_idxs = twoD.utils.get_interval_border_idxs(
        all_matching_idxs=all_immobility_idxs,
        framerate=1 / fps,
        min_interval_duration=min_interval_duration,
    )
    immobility_related_events = create_event_objects(
        interval_border_idxs=immobility_interval_border_idxs,
        fps=fps,
        event_type=event_type,
    )
    return immobility_related_events

# %% ../../nbs/01_twoD_03_detect.ipynb 8
def create_event_objects(
    interval_border_idxs: List[Tuple[int, int]], fps: int, event_type: str
) -> List[EventBout]:
    events = []
    event_id = 0
    for start_idx, end_idx in interval_border_idxs:
        single_event = EventBout(
            event_id=event_id,
            start_idx=start_idx,
            end_idx=end_idx,
            fps=fps,
            event_type=event_type,
        )
        events.append(single_event)
        event_id += 1
    return events

# %% ../../nbs/01_twoD_03_detect.ipynb 9
def add_event_bouts_to_behavior_df(
    behavior_df: pd.DataFrame, event_type: str, events: List[EventBout]
) -> pd.DataFrame:
    assert event_type not in list(
        behavior_df.columns
    ), f"{event_type} was already a column in self.behavior_df!"
    behavior_df[event_type] = np.nan
    behavior_df[f"{event_type}_id"] = np.nan
    behavior_df[f"{event_type}_duration"] = np.nan
    if len(events) > 0:
        for event_bout in events:
            assert (
                event_bout.event_type == event_type
            ), f"Event types didn´t match! Expected {event_type} but found {event_bout.event_type}."
            behavior_df.iloc[event_bout.start_idx : event_bout.end_idx + 1, -3] = True
            behavior_df.iloc[
                event_bout.start_idx : event_bout.end_idx + 1, -2
            ] = event_bout.id
            behavior_df.iloc[
                event_bout.start_idx : event_bout.end_idx + 1, -1
            ] = event_bout.duration
    return behavior_df

# %% ../../nbs/01_twoD_03_detect.ipynb 10
def get_gait_events(
    all_bodyparts: Dict,
    fps: int,
    gait_min_rolling_speed: float,
    gait_min_duration: float,
) -> List[EventBout]:
    idxs_with_sufficient_speed = np.where(
        all_bodyparts["CenterOfGravity"].df["rolling_speed_cm_per_s"].values
        >= gait_min_rolling_speed
    )[0]
    gait_interval_border_idxs = twoD.utils.get_interval_border_idxs(
        all_matching_idxs=idxs_with_sufficient_speed,
        framerate=1 / fps,
        min_interval_duration=gait_min_duration,
    )
    gait_events = create_event_objects(
        interval_border_idxs=gait_interval_border_idxs, fps=fps, event_type="gait_bout"
    )
    return gait_events

# %% ../../nbs/01_twoD_03_detect.ipynb 11
def get_gait_disruption_events(
    behavior_df: pd.DataFrame,
    fps: int,
    gait_events: List[EventBout],
    gait_disruption_max_time_to_immobility: float,
) -> List[EventBout]:
    n_frames_max_distance = int(gait_disruption_max_time_to_immobility * fps)
    gait_disruption_interval_border_idxs = []
    for gait_bout in gait_events:
        end_idx = gait_bout.end_idx
        unique_immobility_bout_values = behavior_df.loc[
            end_idx : end_idx + n_frames_max_distance + 1, "immobility_bout"
        ].unique()
        if True in unique_immobility_bout_values:
            closest_immobility_bout_id = (
                behavior_df.loc[
                    end_idx : end_idx + n_frames_max_distance + 1, "immobility_bout_id"
                ]
                .dropna()
                .unique()
                .min()
            )
            immobility_interval_border_idxs = (
                get_interval_border_idxs_from_event_type_and_id(
                    behavior_df=behavior_df,
                    event_type="immobility_bout",
                    event_id=closest_immobility_bout_id,
                )
            )
            gait_disruption_interval_border_idxs.append(immobility_interval_border_idxs)
    gait_disruption_events = create_event_objects(
        interval_border_idxs=gait_disruption_interval_border_idxs,
        fps=fps,
        event_type="gait_disruption_bout",
    )
    return gait_disruption_events

# %% ../../nbs/01_twoD_03_detect.ipynb 12
def get_interval_border_idxs_from_event_type_and_id(
    behavior_df: pd.DataFrame, event_type: str, event_id: int
) -> Tuple[int, int]:
    interval_idxs = behavior_df.loc[
        behavior_df[f"{event_type}_id"] == event_id
    ].index.values
    return interval_idxs[0], interval_idxs[-1]
