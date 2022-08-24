from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import pickle


class TestPositionsGroundTruth:
    
    def __init__(self) -> None:
        self.marker_ids_with_distances = {}
        self.unique_marker_ids = []
        self._add_maze_corners()

    
    def add_new_marker_id(self, marker_id: str, other_marker_ids_with_distances: List[Tuple[str, Union[int, float]]]) -> None:
        for other_marker_id, distance in other_marker_ids_with_distances:
            self._add_ground_truth_information(marker_id_a = marker_id, marker_id_b = other_marker_id, distance = distance)
            self._add_ground_truth_information(marker_id_a = other_marker_id, marker_id_b = marker_id, distance = distance)
            
    
    def _add_ground_truth_information(self, marker_id_a: str, marker_id_b: str, distance: Union[int, float]) -> None:
        if marker_id_a not in self.marker_ids_with_distances.keys():
            self.marker_ids_with_distances[marker_id_a] = {}
            self.unique_marker_ids.append(marker_id_a)
        self.marker_ids_with_distances[marker_id_a][marker_id_b] = distance
        
    
    def _add_maze_corners(self) -> None:
        maze_width, maze_length = 4, 50
        maze_diagonal = (maze_width**2 + maze_length**2)**0.5
        maze_corner_distances = {'maze_corner_open_left': [('maze_corner_open_right', maze_width),
                                                           ('maze_corner_closed_right', maze_diagonal),
                                                           ('maze_corner_closed_left', maze_length)],
                                 'maze_corner_open_right': [('maze_corner_closed_right', maze_length),
                                                            ('maze_corner_closed_left', maze_diagonal)],
                                 'maze_corner_closed_left': [('maze_corner_closed_right', maze_width)]}
        for marker_id, distances in maze_corner_distances.items():
            self.add_new_marker_id(marker_id = marker_id, other_marker_ids_with_distances = distances)

    
    def save_to_disk(self, filepath: Path) -> None:
        # ToDo: save dictionary to disk
        pass
    
    
    def load_from_disk(self, filepath: Path) -> None:
        # ToDo: load dictionary from disk & extract unique marker ids
        pass


class SingleCamDataForAnipose:
    
    def __init__(self, cam_id: str, filepath_synchronized_calibration_video: Path) -> None:
        self.cam_id = cam_id
        self.filepath_synchronized_calibration_video = filepath_synchronized_calibration_video
        
        
    def add_manual_test_position_marker(self, marker_id: str, x_or_column_idx: int, y_or_row_idx: int, likelihood: float, overwrite: bool=False) -> None:
        if hasattr(self, 'manual_test_position_marker_coords_pred') == False:
            self.manual_test_position_marker_coords_pred = {}
        if (marker_id in self.manual_test_position_marker_coords_pred.keys()) & (overwrite == False):
            raise ValueError('There are already coordinates for the marker you '
                             f'tried to add: "{marker_id}: {self.manual_test_position_marker_coords_pred[marker_id]}'
                             '". If you would like to overwrite these coordinates, please pass '
                             '"overwrite = True" as additional argument to this method!')
        self.manual_test_position_marker_coords_pred[marker_id] = {'x': [x_or_column_idx], 'y': [y_or_row_idx], 'likelihood': [likelihood]}
        
        
    def save_manual_marker_coords_as_fake_dlc_output(self, output_filepath: Optional[Path]):
        # ToDo: this could very well be suitable to become extracted to a utils function
        #       it could then easily be re-used as "validate_output_filename_and_path"
        #       if for instance the extension string, the defaults, and the warning message
        #       can be adapted / passed as arguments!
        if type(output_filepath) != Path:
            output_filepath = self.filepath_synchronized_calibration_video.parent
        if output_filepath.name.endswith('.h5') == False:
            if output_filepath.is_dir():
                output_filepath = output_filepath.joinpath(f'{self.cam_id}_manual_test_position_marker_fake.h5')
            else:
                output_filepath = output_filepath.parent.joinpath(f'{self.cam_id}_manual_test_position_marker_fake.h5')
        df_out = self._construct_dlc_output_style_df_from_manual_marker_coords()
        df_out.to_hdf(output_filepath, "df")
        print(f'Your dataframe was successfully saved at: {output_filepath.as_posix()}.')
        self.load_test_position_markers_df_from_dlc_prediction(filepath_deeplabcut_prediction = output_filepath)
        print('Your "fake DLC marker perdictions" were successfully loaded to the SingleCamDataForAnipose object!')
        
        
    def _construct_dlc_output_style_df_from_manual_marker_coords(self) -> pd.DataFrame:
        multi_index = self._get_multi_index()
        df = pd.DataFrame(data = {}, columns = multi_index)
        for scorer, marker_id, key in df.columns:
            df[(scorer, marker_id, key)] = self.manual_test_position_marker_coords_pred[marker_id][key]
        return df

    
    def _get_multi_index(self) -> pd.MultiIndex:
        multi_index_column_names = [[], [], []]
        for marker_id in self.manual_test_position_marker_coords_pred.keys():
            for column_name in ("x", "y", "likelihood"):
                multi_index_column_names[0].append("manually_annotated_marker_positions")
                multi_index_column_names[1].append(marker_id)
                multi_index_column_names[2].append(column_name)
        return pd.MultiIndex.from_arrays(multi_index_column_names, names=('scorer', 'bodyparts', 'coords'))

    
    def load_test_position_markers_df_from_dlc_prediction(self, filepath_deeplabcut_prediction: Path) -> None:
        df = pd.read_hdf(filepath_deeplabcut_prediction)
        setattr(self, 'test_position_markers_df', df)

        
    def validate_test_position_marker_ids(self, test_positions_ground_truth: TestPositionsGroundTruth, add_missing_marker_ids_with_0_likelihood: bool=True) -> None:
        if hasattr(self, 'test_position_markers_df') == False:
            raise ValueError('There was no DLC prediction of the test position markers loaded yet. '
                             'Please load it using the ".load_test_position_markers_df_from_dlc_prediction()" '
                             'method on this object (if you have DLC predictions to load) - or first add '
                             'the positions manually using the ".add_manual_test_position_marker()" method '
                             'on this object, and eventually load these data after adding all marker_ids '
                             'that you could identify via the ".save_manual_marker_coords_as_fake_dlc_output() '
                             'method on this object.')
        ground_truth_marker_ids = test_positions_ground_truth.unique_marker_ids.copy()
        prediction_marker_ids = list(set([marker_id for scorer, marker_id, key in self.test_position_markers_df.columns]))
        marker_ids_not_in_ground_truth = self._find_non_matching_marker_ids(prediction_marker_ids, ground_truth_marker_ids)
        marker_ids_not_in_prediction = self._find_non_matching_marker_ids(ground_truth_marker_ids, prediction_marker_ids)
        if add_missing_marker_ids_with_0_likelihood & (len(marker_ids_not_in_prediction) > 0):
            self._add_missing_marker_ids_to_prediction(missing_marker_ids = marker_ids_not_in_prediction)
            print('The following marker_ids were missing and added to the dataframe with a '
                  f'likelihood of 0: {marker_ids_not_in_prediction}.')
        if len(marker_ids_not_in_ground_truth) > 0:
            self._remove_marker_ids_not_in_ground_truth(marker_ids_to_remove = marker_ids_not_in_ground_truth)
            print('The following marker_ids were deleted from the dataframe, since they were '
                  f'not present in the ground truth: {marker_ids_not_in_ground_truth}.')
            
            
                  
            
    def _add_missing_marker_ids_to_prediction(self, missing_marker_ids: List[str]) -> None:
        df = self.test_position_markers_df
        scorer = list(df.columns)[0][0]
        for marker_id in missing_marker_ids:
            for key in ['x', 'y', 'likelihood']:
                df[(scorer, marker_id, key)] = 0
                

    def _remove_marker_ids_not_in_ground_truth(self, marker_ids_to_remove: List[str]) -> None:
        df = self.test_position_markers_df
        columns_to_remove = [column_name for column_name in df.columns if column_name[1] in marker_ids_to_remove]
        df.drop(columns = columns_to_remove, inplace=True)
            
            
    def _find_non_matching_marker_ids(self, marker_ids_to_match: List[str], template_marker_ids: List[str]) -> List:
        return [marker_id for marker_id in marker_ids_to_match if marker_id not in template_marker_ids]
    
    
    def add_cropping_offsets(self, x_or_column_offset: int=0, y_or_row_offset: int=0) -> None:
        setattr(self, 'cropping_offsets', (x_or_column_offset, y_or_row_offset))
    
    
    def add_flipping_details(self, flipped_horizontally: bool=False, flipped_vertically: bool=False) -> None:
        setattr(self, 'flipped_horizontally', flipped_horizontally)
        setattr(self, 'flipped_vertically', flipped_vertically)
        
    
    def run_intrinsic_camera_calibration(self, filepath_checkerboard_video: Path, save: bool=True, max_frame_count: int=300) -> None:
        # ToDo     
        # run calibration
        # save calibration matrix
        self._set_intrinsic_matrix(intrinsic_matrix = intrinsic_matrix)
        
        
    def load_intrinsic_camera_calibration(self, filepath_intrinsic_calibration: Path) -> None:
        with open(filepath_intrinsic_calibration, 'rb') as io:
            intrinsic_matrix = pickle.load(io)
        self._set_intrinsic_matrix(intrinsic_matrix = intrinsic_matrix)
    
    
    def _set_intrinsic_matrix(self, intrinsic_matrix: np.ndarray) -> None:
        setattr(self, 'intrinsic_matrix', intrinsic_matrix)


#class CalibrationForAnipose3DTracking: