# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_inspect.ipynb.

# %% auto 0
__all__ = ['get_only_matching_xlsx_files', 'get_metadata_from_filename', 'collect_all_available_data', 'filter_dataframe', 'plot',
           'check_data_availability']

# %% ../nbs/02_inspect.ipynb 3
from pathlib import Path, PosixPath
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% ../nbs/02_inspect.ipynb 6
def get_only_matching_xlsx_files(
    dir_path: Path, paradigm_id: str, week_id: Optional[int] = None
) -> List[Path]:
    filtered_filepaths = []
    for filepath in dir_path.iterdir():
        if filepath.name.endswith(".xlsx") and (paradigm_id in filepath.name):
            if week_id != None:
                if f"week-{week_id}." in filepath.name:
                    filtered_filepaths.append(filepath)
            else:
                filtered_filepaths.append(filepath)
    return filtered_filepaths

# %% ../nbs/02_inspect.ipynb 7
def get_metadata_from_filename(
    filepath_session_results: Path, group_assignment_filepath: Path
) -> Dict:
    metadata = {}
    (
        metadata["line_id"],
        mouse_id,
        metadata["paradigm_id"],
        week_string_with_file_extension,
    ) = filepath_session_results.name.split("_")
    metadata["subject_id"] = f'{metadata["line_id"]}_{mouse_id}'
    metadata["week_id"] = week_string_with_file_extension[
        week_string_with_file_extension.find("-")
        + 1 : week_string_with_file_extension.find(".")
    ]
    df_group_assignment = pd.read_excel(group_assignment_filepath)
    if metadata["subject_id"] in df_group_assignment["subject_id"].unique():
        metadata["group_id"] = df_group_assignment.loc[
            df_group_assignment["subject_id"] == metadata["subject_id"], "group_id"
        ].iloc[0]
    elif (
        metadata["subject_id"] in df_group_assignment["alternative_subject_id"].unique()
    ):
        metadata["group_id"] = df_group_assignment.loc[
            df_group_assignment["alternative_subject_id"] == metadata["subject_id"],
            "group_id",
        ].iloc[0]
    else:
        metadata["group_id"] = "unknown"
    return metadata

# %% ../nbs/02_inspect.ipynb 9
def collect_all_available_data(
    root_dir_path: Path,  # Filepath to the directory that contains the exported results .xlsx files
    group_assignment_filepath: Path,  # Filepath to the group_assignments.xlsx file
    paradigm_ids: List[str],  # List of paradigms of which the results shall be loaded
    week_ids: List[str],  # List of weeks from which the results shall be loaded
    sheet_name: str,  # Tab name of the exported results sheet to load, e.g. "session_overview"
) -> pd.DataFrame:
    all_recording_results_dfs = []
    for week_id in week_ids:
        for paradigm_id in paradigm_ids:
            tmp_matching_filepaths = get_only_matching_xlsx_files(
                dir_path=root_dir_path, paradigm_id=paradigm_id, week_id=week_id
            )
            for filepath in tmp_matching_filepaths:
                metadata = get_metadata_from_filename(
                    filepath_session_results=filepath,
                    group_assignment_filepath=group_assignment_filepath,
                )
                tmp_xlsx = pd.ExcelFile(filepath)
                tmp_df = pd.read_excel(tmp_xlsx, sheet_name=sheet_name, index_col=0)
                for key, value in metadata.items():
                    tmp_df[key] = value
                all_recording_results_dfs.append(tmp_df)
    df = pd.concat(all_recording_results_dfs)
    df.reset_index(drop=True, inplace=True)
    return df

# %% ../nbs/02_inspect.ipynb 13
def filter_dataframe(df: pd.DataFrame, filter_criteria: List[Tuple]) -> pd.DataFrame:
    # assert all list have equal lenghts
    valid_idxs_per_criterion = []
    for column_name, comparison_method, reference_value in filter_criteria:
        # assert valid key in comparison methods
        # assert column name exists
        if comparison_method == "greater":
            valid_idxs_per_criterion.append(
                df.loc[df[column_name] > reference_value].index.values
            )
        elif comparison_method == "smaller":
            valid_idxs_per_criterion.append(
                df.loc[df[column_name] < reference_value].index.values
            )
        elif comparison_method == "equal_to":
            valid_idxs_per_criterion.append(
                df.loc[df[column_name] == reference_value].index.values
            )
        elif comparison_method == "is_in_list":
            valid_idxs_per_criterion.append(
                df.loc[df[column_name].isin(reference_value)].index.values
            )
        elif comparison_method == "is_nan":
            valid_idxs_per_criterion.append(
                df.loc[df[column_name].isnull()].index.values
            )
    shared_valid_idxs_across_all_criteria = valid_idxs_per_criterion[0]
    if len(valid_idxs_per_criterion) > 1:
        for i in range(1, len(valid_idxs_per_criterion)):
            shared_valid_idxs_across_all_criteria = np.intersect1d(
                shared_valid_idxs_across_all_criteria, valid_idxs_per_criterion[i]
            )
    df_filtered = df.loc[shared_valid_idxs_across_all_criteria, :].copy()
    return df_filtered

# %% ../nbs/02_inspect.ipynb 19
def plot(
    df: pd.DataFrame,  # the DataFrame that contains the data you´d like to plot
    x_column: str,  # the column name of the data that should be visualized on the x-axis
    y_column: str,  # the column name of the data that should be visualized on the y-axis
    plot_type: str = "violinplot",  # currently only "violinplot" and "stripplot" are implemented
    hue_column: Optional[
        str
    ] = None,  # if you´d like to use the data of a column to color-code the plotted, you can specify it here (see example below)
    hide_legend: bool = True,  # pass along as `False` if you´d like the legend of the plot to be displayed
) -> None:
    fig = plt.figure(figsize=(8, 5), facecolor="white")
    ax = fig.add_subplot(111)
    if plot_type == "violinplot":
        sns.violinplot(data=df, x=x_column, y=y_column, hue=hue_column)
        if df.shape[0] < 10_000:
            sns.stripplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue_column,
                dodge=True,
                color="black",
                alpha=0.3,
            )
    elif plot_type == "stripplot":
        sns.stripplot(data=df, x=x_column, y=y_column, hue=hue_column, dodge=True)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if hide_legend:
        ax.get_legend().remove()
    plt.show()

# %% ../nbs/02_inspect.ipynb 22
def check_data_availability(
    root_dir_path: Path, all_week_ids: List[int], all_paradigm_ids: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame({}, index=all_week_ids)
    root_dir_path = root_dir_path
    for week_id in all_week_ids:
        for paradigm_id in all_paradigm_ids:
            all_matching_filepaths = get_only_matching_xlsx_files(
                dir_path=root_dir_path, paradigm_id=paradigm_id, week_id=week_id
            )
            for filepath in all_matching_filepaths:
                subject_id = get_metadata_from_filename(filepath)["subject_id"]
                if f"{subject_id}_{paradigm_id}" not in df.columns:
                    df[f"{subject_id}_{paradigm_id}"] = False
                df.loc[week_id, f"{subject_id}_{paradigm_id}"] = True
    return df
