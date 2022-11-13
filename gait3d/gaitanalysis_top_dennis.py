import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import os
import imageio.v3 as iio
from scipy.spatial.transform import Rotation
from scipy import interpolate
from scipy.linalg import norm
from scipy.signal import find_peaks, peak_widths
from statistics import mean
import cv2
import pickle
from scipy.signal import savgol_filter
        
class RecordingTop(ABC):
    """
    Class for Analysing 2D-Position Data of mice in the OpeningTrack.
    
    Attributes:
        full_df_from_file(pandas.DataFrame): the Dataframe containing all bodyparts with x, y-coordinates and likelihood as returned by DLC
        fps(int): fps of the recording
        metadata(Dict): dictionary containing information read from the filename, such as animal_id, recording_date and Opening Track paradigm
    """
       
    @property
    def valid_paradigms(self)->List[str]:
        return ['OTR', 'OTT', 'OTE']
    
    @property
    def valid_mouse_lines(self)->List[str]:
        return ['194', '195', '196', '206', '209']
    
    @property
    def marker_ids_to_exclude_for_smoothing_and_interpolation(self) -> List[str]:
        return ['LED5', 'MazeCornerClosedRight', 'MazeCornerClosedLeft', 'MazeCornerOpenRight', 'MazeCornerOpenLeft']

    
    def __init__(self, filepath: Path, fps: int)->None:
        """
        Constructor for the Recording2D class.
        
        This function calls functions to get the Dataframe from the csv, that is given as filepath argument and to read metadata from the filename.
        
        Parameters:
            filepath(pathlib.Path): the filepath to the h5 containing DLC data
            fps(int): fps of the recording
        """
        self.filepath = filepath
        self.full_df_from_file = self._get_df_from_file(filepath = filepath)
        self.fps = fps
        self.metadata = self._retrieve_metadata(filepath = filepath.name)


    def _get_df_from_file(self, filepath: Path)->pd.DataFrame:
        """
        Reads the Dataframe from the h5-file and drops irrelevant columns and rows.
        
        Parameters:
            filepath(pathlib.Path): the path linked to the h5.file
        Returns:
            pandas.DataFrame: the Dataframe containing all bodyparts with x, y-coordinates and likelihood as returned by DLC
        """
        if filepath.name.endswith('.csv'):
            df = pd.read_csv(filepath, low_memory = False)
            df = df.drop('scorer', axis=1)
            df.columns = df.iloc[0, :]+ '_' + df.iloc[1, :]
            df = df.drop([0, 1], axis=0)
            df = df.reset_index()
            df = df.drop('index', axis=1)
            df = df.astype(float)
        elif filepath.name.endswith('.h5'):
            df = pd.read_hdf(filepath)
            try:
                df = df.drop('scorer', axis=1)
                df.columns = df.iloc[0, :]+ '_' + df.iloc[1, :]
                df = df.drop([0, 1], axis=0)
                df = df.reset_index()
                df = df.drop('index', axis=1)
            except KeyError:
                pass
            df = df.astype(float)
        else:
            raise ValueError('The Path you specified is not linking to a .csv/.h5-file!')
        return df


    def _retrieve_metadata(self, filepath: str)->Dict:
        """
        Function, that slices the Filename to get the encoded metadata.
        
        Relying on file naming like this: 196_F7-27_220826_OTT_Bottom_synchronizedDLC_resnet152_OT_BottomCam_finalSep20shuffle1_550000filtered.h5
        
        Parameters:
            filepath(pathlib.Path): the path linked to the h5.file
        Returns:
            Dict: containing date of recording, animal_id and OT paradigm
        """
        filepath_slices = filepath.split('_')
        animal_line, animal_id, recording_date, paradigm, cam_id = filepath_slices[0], filepath_slices[1], filepath_slices[2], filepath_slices[3][0:3], 'Top'
        self._check_metadata(metadata = (animal_line, animal_id, recording_date, paradigm, cam_id))
        return {'recording_date': self.recording_date, 'animal': self.mouse_line + '_' + self.mouse_id, 'paradigm': self.paradigm, 'cam': self.cam_id}

    
    def _check_metadata(self, metadata = Tuple[str]) -> None: 
        animal_line, animal_id, recording_date, paradigm, cam_id = metadata[0], metadata[1], metadata[2], metadata[3], metadata[4]
        self.cam_id = cam_id
        if animal_line not in self.valid_mouse_lines:
            while True:
                entered_input = input(f'Mouse line for {self.filepath}')
                if entered_input in self.valid_mouse_lines:
                    self.mouse_line = entered_input
                    break
                else:
                    print(f'Entered mouse line does not match any of the defined mouse lines. \nPlease enter one of the following lines: {self.valid_mouse_lines}')
        else:
            self.mouse_line = animal_line
        if not animal_id.startswith('F'):
            while True:
                entered_input = input(f'Mouse ID for {self.filepath}')
                if entered_input.startswith('F'):
                    self.mouse_id = entered_input
                    break
                else:
                    print(f'Animal ID has to start with F. Example: F2-14')
        else:
            self.mouse_id = animal_id
        if paradigm not in self.valid_paradigms:
            while True:
                entered_input = input(f'Paradigm for {self.filepath}')
                if entered_input in self.valid_paradigms:
                    self.paradigm = entered_input
                    break
                else:
                    print(f'Entered paradigm does not match any of the defined paradigms. \nPlease enter one of the following paradigms: {self.valid_paradigms}')
        else:
            self.paradigm = paradigm
        try:
            int(recording_date)
            self.recording_date = recording_date
        except:
            while True:
                entered_input = input(f'Recording date for {self.filepath}')
                try:
                    int(recording_date)
                    self.recording_date = recording_date
                    break
                except:
                    print(f'Entered recording date has to be an integer in shape YYMMDD. Example: 220812')       

    
    def preprocess(self,
                   marker_ids_to_compute_coverage: List[str]=['TailBase', 'Snout'],
                   coverage_threshold: float=0.75, 
                   max_seconds_to_interpolate: float=0.5, 
                   likelihood_threshold: float=0.5,
                   marker_ids_to_compute_center_of_gravity: List[str]=['TailBase', 'Snout'],
                   relative_maze_normalization_error_tolerance: float=0.25
                   ) -> None:
        self.log = {'critical_markers': marker_ids_to_compute_coverage,
                    'coverage_threshold': coverage_threshold,
                    'max_seconds_to_interpolate': max_seconds_to_interpolate,
                    'likelihood_threshold': likelihood_threshold,
                    'center_of_gravity_based_on': marker_ids_to_compute_center_of_gravity, 
                    'relative_error_tolerance_corner_detection': relative_maze_normalization_error_tolerance}
        window_length = self._get_max_odd_n_frames_for_time_interval(fps = self.fps, time_interval = max_seconds_to_interpolate)
        marker_ids_to_preprocess = self._get_preprocessing_relevant_marker_ids(df = self.full_df_from_file)
        smoothed_df = self._smooth_tracked_coords_and_likelihood(marker_ids = marker_ids_to_preprocess, window_length = window_length, polyorder = 3)
        interpolated_df = self._interpolate_low_likelihood_intervals(df = smoothed_df, marker_ids = marker_ids_to_preprocess, max_interval_length = window_length)
        interpolated_df_with_cog = self._add_new_marker_derived_existing_markers(df = interpolated_df,
                                                                                 existing_markers = marker_ids_to_compute_center_of_gravity,
                                                                                 new_marker_id = 'CenterOfGravity',
                                                                                 likelihood_threshold = likelihood_threshold)
        preprocessed_df = self._interpolate_low_likelihood_intervals(df = interpolated_df_with_cog,
                                                                     marker_ids = ['CenterOfGravity'],
                                                                     max_interval_length = window_length)
        self.preprocessed_df = preprocessed_df
        self.log['coverage_critical_markers'] = self._compute_coverage(df = preprocessed_df,
                                                                       critical_marker_ids = marker_ids_to_compute_coverage,
                                                                       likelihood_threshold = likelihood_threshold)
        self.log['coverage_CenterOfGravity'] = self._compute_coverage(df = preprocessed_df,
                                                                      critical_marker_ids = ['CenterOfGravity'],
                                                                      likelihood_threshold = likelihood_threshold)
        if self.log['coverage_critical_markers'] >= coverage_threshold:
            normalization_params = self._get_parameters_to_normalize_maze_coordinates(df = preprocessed_df,
                                                                                      relative_error_tolerance = relative_maze_normalization_error_tolerance)
            self.normalized_df = self._normalize_df(df = preprocessed_df, normalization_parameters = normalization_params)
            self.bodyparts = self._create_bodypart_objects()
            self.log['normalization_parameters'] = normalization_params
            self.log['normalized_maze_corner_coordinates'] = self._get_normalized_maze_corners(normalization_parameters = normalization_params)
        else:
            print(f'Unfortunately, there was insufficient tracking coverage for {self.filepath.name}. We have to skip this recording!')


    def _get_max_odd_n_frames_for_time_interval(self, fps: int, time_interval: 0.5) -> int:
        assert type(fps) == int, '"fps" has to be an integer!'
        frames_per_time_interval = fps * time_interval
        if frames_per_time_interval % 2 == 0:
            max_odd_frame_count = frames_per_time_interval - 1
        elif frames_per_time_interval == int(frames_per_time_interval):
            max_odd_frame_count = frames_per_time_interval
        else:
            frames_per_time_interval = int(frames_per_time_interval)
            if frames_per_time_interval % 2 == 0:
                max_odd_frame_count = frames_per_time_interval - 1
            else:
                max_odd_frame_count = frames_per_time_interval
        assert max_odd_frame_count > 0, f'The specified time interval is too short to fit an odd number of frames'
        return int(max_odd_frame_count)                


    def _get_preprocessing_relevant_marker_ids(self, df: pd.DataFrame) -> List[str]:
        all_marker_ids = self._get_all_unique_marker_ids(df = df)
        relevant_marker_ids = all_marker_ids
        for marker_id_to_exclude in self.marker_ids_to_exclude_for_smoothing_and_interpolation:
            relevant_marker_ids.remove(marker_id_to_exclude)
        return relevant_marker_ids


    def _get_all_unique_marker_ids(self, df: pd.DataFrame) -> List[str]:
        unique_marker_ids = []
        for column_name in df.columns:
            marker_id, _ = column_name.split('_')
            if marker_id not in unique_marker_ids:
                unique_marker_ids.append(marker_id)
        return unique_marker_ids


    def _smooth_tracked_coords_and_likelihood(self, marker_ids: List[str], window_length: int, polyorder: int=3) -> pd.DataFrame:
        """
        Smoothes the DataFrame basically using the implementation from DLC2kinematics:
        https://github.com/AdaptiveMotorControlLab/DLC2Kinematics/blob/82e7e60e00e0efb3c51e024c05a5640c91032026/src/dlc2kinematics/preprocess.py#L64
        However, with one key change: likelihoods will also be smoothed.
        In addition, we will not smooth the columns for the tracked LEDs and the MazeCorners.
        
        Note: window_length has to be an odd integer!
        """
        smoothed_df = self.full_df_from_file.copy()
        column_names = self._get_column_names(df = smoothed_df,
                                              column_identifiers = ['x', 'y', 'likelihood'],
                                              marker_ids = marker_ids)
        column_idxs_to_smooth = smoothed_df.columns.get_indexer(column_names)
        smoothed_df.iloc[:, column_idxs_to_smooth] = savgol_filter(x = smoothed_df.iloc[:, column_idxs_to_smooth],
                                                                   window_length = window_length,
                                                                   polyorder = polyorder,
                                                                   axis = 0)
        return smoothed_df  


    def _get_column_names(self, df: pd.DataFrame, column_identifiers: List[str], marker_ids: List[str]) -> List[str]:
        matching_column_names = []
        for column_name in df.columns:
            marker_id, column_identifier = column_name.split('_')
            if (marker_id in marker_ids) and (column_identifier in column_identifiers):
                matching_column_names.append(column_name)
        return matching_column_names


    def _interpolate_low_likelihood_intervals(self, df: pd.DataFrame, marker_ids: List[str], max_interval_length: int) -> pd.DataFrame:
        interpolated_df = df.copy()
        for marker_id in marker_ids:
            low_likelihood_interval_border_idxs = self._get_low_likelihood_interval_border_idxs(likelihood_series = interpolated_df[f'{marker_id}_likelihood'], 
                                                                                                max_interval_length = max_interval_length)
            for start_idx, end_idx in low_likelihood_interval_border_idxs:
                if (start_idx - 1 >= 0) and (end_idx + 2 < interpolated_df.shape[0]):
                    interpolated_df[f'{marker_id}_x'][start_idx - 1 : end_idx + 2] = interpolated_df[f'{marker_id}_x'][start_idx - 1 : end_idx + 2].interpolate()
                    interpolated_df[f'{marker_id}_y'][start_idx - 1 : end_idx + 2] = interpolated_df[f'{marker_id}_y'][start_idx - 1 : end_idx + 2].interpolate()
                    interpolated_df[f'{marker_id}_likelihood'][start_idx : end_idx + 1] = 0.5
        return interpolated_df    


    def _get_low_likelihood_interval_border_idxs(self, likelihood_series: pd.Series, max_interval_length: int, min_likelihood_threshold: float=0.5) -> List[Tuple[int, int]]:
        all_low_likelihood_idxs = np.where(likelihood_series.values < min_likelihood_threshold)[0]
        last_idxs_of_idx_intervals = np.where(np.diff(all_low_likelihood_idxs) > 1)[0]
        all_interval_end_idxs = np.concatenate([last_idxs_of_idx_intervals, np.array([all_low_likelihood_idxs.shape[0] - 1])])
        all_interval_start_idxs = np.concatenate([np.asarray([0]), last_idxs_of_idx_intervals + 1])
        interval_lengths = all_interval_end_idxs - all_interval_start_idxs
        idxs_of_intervals_matching_length_criterion = np.where(interval_lengths <= max_interval_length)[0]
        selected_interval_start_idxs = all_interval_start_idxs[idxs_of_intervals_matching_length_criterion]
        selected_interval_end_idxs = all_interval_end_idxs[idxs_of_intervals_matching_length_criterion]
        interval_border_idxs = []
        for start_idx, end_idx in zip(selected_interval_start_idxs, selected_interval_end_idxs):
            border_idxs = (all_low_likelihood_idxs[start_idx], all_low_likelihood_idxs[end_idx])
            interval_border_idxs.append(border_idxs)
        return interval_border_idxs


    def _add_new_marker_derived_existing_markers(self, df: pd.DataFrame, existing_markers: List[str], new_marker_id: str, likelihood_threshold: float = 0.5)->None:
        df_with_new_marker = df.copy()
        for coordinate in ['x', 'y']:
            df_with_new_marker[f'{new_marker_id}_{coordinate}'] = (sum([df_with_new_marker[f'{marker_id}_{coordinate}'] for marker_id in existing_markers]))/len(existing_markers)
        df_with_new_marker[f'{new_marker_id}_likelihood'] = 0
        row_idxs_where_all_likelihoods_exceeded_threshold = self._get_idxs_where_all_markers_exceed_likelihood(df = df_with_new_marker, 
                                                                                                           marker_ids = existing_markers, 
                                                                                                           likelihood_threshold = 0.5)
        df_with_new_marker.iloc[row_idxs_where_all_likelihoods_exceeded_threshold, -1] = 1
        return df_with_new_marker
        

    def _get_idxs_where_all_markers_exceed_likelihood( self, df: pd.DataFrame, marker_ids: List[str], likelihood_threshold: float=0.5) -> np.ndarray:
        valid_idxs_per_marker_id = []
        for marker_id in marker_ids:
            valid_idxs_per_marker_id.append(df.loc[df[f'{marker_id}_likelihood'] >= likelihood_threshold].index.values)
        shared_valid_idxs_for_all_markers = valid_idxs_per_marker_id[0]
        if len(valid_idxs_per_marker_id) > 1:
            for i in range(1, len(valid_idxs_per_marker_id)):
                shared_valid_idxs_for_all_markers = np.intersect1d(shared_valid_idxs_for_all_markers, valid_idxs_per_marker_id[i])
        return shared_valid_idxs_for_all_markers

            
    def _compute_coverage(self, df: pd.DataFrame, critical_marker_ids: List[str], likelihood_threshold: float=0.5) -> float:
        idxs_where_all_markers_exceed_likelihood_threshold = self._get_idxs_where_all_markers_exceed_likelihood(df = df, 
                                                                                                                marker_ids = critical_marker_ids,
                                                                                                                likelihood_threshold = likelihood_threshold)
        return idxs_where_all_markers_exceed_likelihood_threshold.shape[0] / df.shape[0]
        

    def _get_parameters_to_normalize_maze_coordinates(self, df: pd.DataFrame, relative_error_tolerance: float) -> Dict:
        corners = self._get_corner_coords_with_likelihoods(df = df)
        translation_vector = self._get_translation_vector(coords_closed_left = corners['MazeCornerClosedLeft']['coords'])
        best_result = self._evaluate_maze_shape_using_open_corners(corners_and_likelihoods = corners, tolerance = relative_error_tolerance)
        if best_result['valid']:
            side_id = best_result['side_id']
            self.log['maze_normalization_based_on'] = f'MazeCornerClosed{side_id}_and_MazeCornerOpen{side_id}'
            conversion_factor = self._get_conversion_factor_px_to_cm(coords_point_a = corners[f'MazeCornerClosed{side_id}']['coords'],
                                                                     coords_point_b = corners[f'MazeCornerOpen{side_id}']['coords'],
                                                                     distance_in_cm = 50)
            rotation_angle = self._get_rotation_angle_with_open_corner(corners = corners,
                                                                       side_id = best_result['side_id'],
                                                                       translation_vector = translation_vector,
                                                                       conversion_factor = conversion_factor)
        else:
            self.log['maze_normalization_based_on'] = f'MazeCornerClosedRight_and_MazeCornerClosedLeft'
            conversion_factor = self._get_conversion_factor_px_to_cm(coords_point_a = corners['MazeCornerClosedLeft']['coords'],
                                                                     coords_point_b = corners['MazeCornerClosedRight']['coords'],
                                                                     distance_in_cm = 4)
            rotation_angle = self._get_rotation_angle_with_closed_corners_only(corners = corners,
                                                                               translation_vector = translation_vector,
                                                                               conversion_factor = conversion_factor)
        return {'translation_vector': translation_vector, 'rotation_angle': rotation_angle, 'conversion_factor': conversion_factor}

        
    def _get_corner_coords_with_likelihoods(self, df: pd.DataFrame) -> Dict:
        corner_coords_with_likelihood = {}
        for corner_marker_id in ['MazeCornerClosedRight', 'MazeCornerClosedLeft', 'MazeCornerOpenRight', 'MazeCornerOpenLeft']:
            xy_coords, min_likelihood = self._get_most_reliable_marker_position_with_likelihood(df = df, marker_id = corner_marker_id)
            corner_coords_with_likelihood[corner_marker_id] = {'coords': xy_coords, 'min_likelihood': min_likelihood}
        return corner_coords_with_likelihood


    def _get_most_reliable_marker_position_with_likelihood(self, df: pd.DataFrame, marker_id: str, percentile: float=99.95) -> Tuple[np.array, float]:
        likelihood_threshold = np.nanpercentile(df[f'{marker_id}_likelihood'].values, percentile)
        df_most_reliable_frames = df.loc[df[f'{marker_id}_likelihood'] >= likelihood_threshold].copy()
        most_reliable_x, most_reliable_y = df_most_reliable_frames[f'{marker_id}_x'].median(), df_most_reliable_frames[f'{marker_id}_y'].median()
        return np.array([most_reliable_x, most_reliable_y]), likelihood_threshold    
            

    def _get_translation_vector(self, coords_closed_left: np.ndarray) -> np.ndarray:
        """
        Function that calculates the offset of the left closed mazecorner to (0, 0).
        
        Parameters:
            coords_closed_right (np.ndarray): containing the coordinat (i.e. vector) of the right closed maze corner that shall become 0/0
            
        Returns:
            translation_vector(np.array): vector with offset in each dimension
        """
        return -coords_closed_left


    def _evaluate_maze_shape_using_open_corners(self, corners_and_likelihoods: Dict, tolerance: float) -> Dict:
        best_result = {'valid': False, 'mean_error': tolerance + 1, 'open_corner_id': None, 'side_id': None}
        for open_corner_marker_id in [corner_marker_id for corner_marker_id in corners_and_likelihoods.keys() if 'Open' in corner_marker_id]:
            valid_positions = False
            side_id = open_corner_marker_id[open_corner_marker_id.find('Open') + 4:]
            if side_id == 'Left': opposite_side_id = 'Right'
            else: opposite_side_id = 'Left'
            closed_corner_opposite_side = f'MazeCornerClosed{opposite_side_id}'
            angle_error = self._compute_angle_error(a = corners_and_likelihoods[f'MazeCornerClosed{opposite_side_id}']['coords'],
                                              b = corners_and_likelihoods[f'MazeCornerClosed{side_id}']['coords'],
                                              c = corners_and_likelihoods[open_corner_marker_id]['coords'])
            distance_ratio_error = self._compute_distance_ratio_error(corners_and_likelihoods = corners_and_likelihoods,
                                                                open_corner_marker_id = open_corner_marker_id,
                                                                side_id = side_id)
            if (angle_error <= tolerance) & (distance_ratio_error <= tolerance):
                valid_positions = True
            mean_error = (angle_error + distance_ratio_error) / 2
            if mean_error < best_result['mean_error']:
                best_result['valid'] = valid_positions
                best_result['mean_error'] = mean_error
                best_result['open_corner_id'] = open_corner_marker_id
                best_result['side_id'] = side_id
        return best_result
    
    
    def _compute_angle_error(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        # b is point at the joint that connects the other two
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))
        return self._compute_error_proportion(query_value = angle, target_value = 90)


    def _compute_error_proportion(self, query_value: float, target_value: float) -> float:
        return abs(query_value - target_value) / target_value


    def _compute_distance_ratio_error(self, corners_and_likelihoods: Dict, open_corner_marker_id: str, side_id: str) -> float:
        maze_width = self._get_distance_between_two_points(corners_and_likelihoods['MazeCornerClosedLeft']['coords'],
                                                           corners_and_likelihoods['MazeCornerClosedRight']['coords'])
        maze_length = self._get_distance_between_two_points(corners_and_likelihoods[f'MazeCornerClosed{side_id}']['coords'],
                                                           corners_and_likelihoods[open_corner_marker_id]['coords'])
        distance_ratio = maze_length/maze_width
        return self._compute_error_proportion(query_value = distance_ratio, target_value = 50/4)


    def _get_distance_between_two_points(self, coords_point_a: np.ndarray, coords_point_b: np.ndarray) -> float:
        return ((coords_point_a[0] - coords_point_b[0])**2 + (coords_point_a[1] - coords_point_b[1])**2)**0.5


    def _get_conversion_factor_px_to_cm(self, coords_point_a: np.ndarray, coords_point_b: np.ndarray, distance_in_cm: float) -> float:
        distance = self._get_distance_between_two_points(coords_point_a, coords_point_b)
        return distance_in_cm / distance

    
    def _get_rotation_angle_with_open_corner(self, corners: Dict, side_id: str, translation_vector: np.ndarray, conversion_factor: float) -> float:
        """
        Function, that calculates the rotation angle of the maze considering the best matching open corner
        and the corresponding closed corner on the same side.
            
        Returns:
            float: angle in radians
        """
        if side_id == 'Left':
            side_specific_y = 0
        else:
            side_specific_y = 4
        translated_closed_corner = corners[f'MazeCornerClosed{side_id}']['coords'] + translation_vector
        translated_open_corner = corners[f'MazeCornerOpen{side_id}']['coords'] + translation_vector
        target_rotated_open_corner = np.asarray([50 / conversion_factor, side_specific_y / conversion_factor])
        length_a = self._get_distance_between_two_points(translated_open_corner, target_rotated_open_corner) * conversion_factor
        length_b = self._get_distance_between_two_points(translated_open_corner, translated_closed_corner) * conversion_factor
        length_c = 50
        angle = math.acos((length_b**2 + length_c**2 - length_a**2) / (2 * length_b * length_c))
        return angle     

    
    def _get_rotation_angle_with_closed_corners_only(self, corners: Dict, translation_vector: np.ndarray, conversion_factor: float) -> float:
        translated_closed_left = corners['MazeCornerClosedLeft']['coords'] + translation_vector
        translated_closed_right = corners['MazeCornerClosedRight']['coords'] + translation_vector
        target_rotated_closed_right = np.asarray([0, 4 / conversion_factor])
        
        length_a = self._get_distance_between_two_points(translated_closed_right, target_rotated_closed_right) * conversion_factor
        length_b = self._get_distance_between_two_points(translated_closed_left, translated_closed_right) * conversion_factor
        length_c = 4
        angle = math.acos((length_b**2 + length_c**2 - length_a**2) / (2 * length_b * length_c))
        return angle


    def _normalize_df(self, df: pd.DataFrame, normalization_parameters)->None:
        unadjusted_df = df.copy()
        translated_df = self._translate_df(df = unadjusted_df, translation_vector = normalization_parameters['translation_vector'])
        rotated_and_translated_df = self._rotate_df(df = translated_df, rotation_angle = normalization_parameters['rotation_angle'])
        final_df = self._convert_df_to_cm(df = rotated_and_translated_df, conversion_factor = normalization_parameters['conversion_factor'])
        return final_df
    
   
    def _translate_df(self, df: pd.DataFrame, translation_vector: np.array) -> pd.DataFrame:
        """
        Function that translates the raw dataframe to the null space.
        
        Parameter:
            translation_vector(np.Array): vector with offset of xy to XY
        Returns:
            translated_df(pandas.DataFrame): the dataframe translated to (0, 0)
        """
        for marker_id in self._get_all_unique_marker_ids(df = df):
            df.loc[:, [f'{marker_id}_x', f'{marker_id}_y']] += translation_vector
        return df

    
    def _rotate_df(self, df: pd.DataFrame, rotation_angle: float) -> pd.DataFrame:
        """
        Function, that rotates the dataframe in 2D.
        
        Parameter:
            rotation_angle: angle in radians of rotation of the xy to XY coordinate system around the z-axis
            df(pandas.DataFrame): the dataframe that will be rotated.
        Returns:
            rotated_df(pandas.DataFrame): the rotated dataframe
        """
        df_rotated = df.copy()
        cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
        for marker_id in self._get_all_unique_marker_ids(df = df):
            df_rotated[f'{marker_id}_x'] = df[f'{marker_id}_x'] * cos_theta - df[f'{marker_id}_y']  * sin_theta
            df_rotated[f'{marker_id}_y'] = df[f'{marker_id}_x'] * sin_theta + df[f'{marker_id}_y']  * cos_theta
        return df_rotated


    def _convert_df_to_cm(self, df: pd.DataFrame, conversion_factor: float) -> pd.DataFrame:
        """
        The coordinates are converted to cm.
        
        Parameters:
            conversion_factor(float): factor to convert the unspecified unit into cm.
            df(pandas.DataFrame): dataframe with unspecified unit
            
        Returns:
            df(pandas.DataFrame): dataframe with values in cm
        """
        for marker_id in self._get_all_unique_marker_ids(df = df):
            df.loc[:, [f'{marker_id}_x', f'{marker_id}_y']] *= conversion_factor
        return df


    def _create_bodypart_objects(self) -> Dict:
        bodyparts = {}
        for marker_id in self._get_all_unique_marker_ids(df = self.normalized_df):
            bodyparts[marker_id] = Bodypart2D(bodypart_id = marker_id, df = self.normalized_df, fps = self.fps)
        return bodyparts


    def _get_normalized_maze_corners(self, normalization_parameters: Dict) -> Dict:
        normalized_maze_corner_coordinates = {}
        corners = self._get_corner_coords_with_likelihoods(df = self.normalized_df)
        for corner_marker_id in corners.keys():
            normalized_maze_corner_coordinates[corner_marker_id] = corners[corner_marker_id]['coords']
        return normalized_maze_corner_coordinates
    
    
    def run_behavioral_analyses(self,
                                immobility_max_rolling_speed: float=2.0,
                                bodyparts_critical_for_freezing: List[str]=['Snout', 'CenterOfGravity'],
                                freezing_min_time: float=0.5,
                                bodyparts_for_direction_front_to_back: List[str]=['Snout', 'CenterOfGravity'],
                                gait_min_rolling_speed: float=4.0,
                                gait_min_duration: float=1.0,
                                gait_disruption_min_time: float=0.2,
                                merge_events_max_inbetween_time: float=0.15
                               ) -> None:
        sliding_window_size = int(round(self._get_max_odd_n_frames_for_time_interval(fps = self.fps, time_interval = 0.5) / 2, 0))
        for bodypart in self.bodyparts.values():
            bodypart.calculate_speed_and_identify_immobility(sliding_window_size = sliding_window_size, immobility_threshold = immobility_max_rolling_speed)
        self.behavior_df = pd.DataFrame(data = {'facing_towards_open_end': [False]*self.normalized_df.shape[0]})
        self._add_orientation_to_behavior_df(bodyparts_for_direction_front_to_back = bodyparts_for_direction_front_to_back)
        self._add_immobility_of_all_critical_bodyparts_to_behavior_df(bodyparts_critical_for_freezing = bodyparts_critical_for_freezing)
        freezing_events = self._get_all_immobility_dependent_events(min_interval_duration_in_s = freezing_min_time, event_type = 'freezing')
        self._add_event_bouts_to_behavior_df(events = freezing_events)
        potential_gait_disruption_events = self._get_all_immobility_dependent_events(min_interval_duration_in_s = gait_disruption_min_time, event_type = 'gait_disruption')
        gait_disruption_events = self._filter_events(events = potential_gait_disruption_events,
                                                     filter_column_name = 'gait'

        
        

        
        self.log['rolling_speed_sliding_window_size'] = sliding_window_size
        self.log['bodyparts_for_direction_front_to_back'] = bodyparts_for_direction_front_to_back
        self.log['immobility_max_rolling_speed'] = immobility_max_rolling_speed
        self.log['gait_min_rolling_speed'] = gait_min_rolling_speed
        self.log['gait_min_duration'] = gait_min_duration
        self.log['gait_disruption_min_time'] = gait_disruption_min_time
        self.log['freezing_min_time'] = freezing_min_time
        self.log['merge_events_max_inbetween_time'] = merge_events_max_inbetween_time
 

    def _add_immobility_of_all_critical_bodyparts_to_behavior_df(self, bodyparts_critical_for_freezing: List[str]) ->:
        # very similar to 
        # can they be combined? different operators for check, though
        valid_idxs_per_marker_id = []
        for bodypart_id in bodyparts_critical_for_freezing:
            tmp_df = self.bodyparts[bodypart_id].df.copy()
            valid_idxs_per_marker_id.append(tmp_df.loc[tmp_df['immobility'] == True].index.values)
        shared_valid_idxs_for_all_markers = valid_idxs_per_marker_id[0]
        if len(valid_idxs_per_marker_id) > 1:
            for i in range(1, len(valid_idxs_per_marker_id)):
                shared_valid_idxs_for_all_markers = np.intersect1d(shared_valid_idxs_for_all_markers, valid_idxs_per_marker_id[i])
        self.behavior_df.iloc[shared_valid_idxs_for_all_markers, 'immobility'] = True

    
    def _get_all_immobility_dependent_events(self, min_interval_duration_in_s: float, event_type: str) -> List[EventBout2D]:
        idxs_immobility_state_changes np.where(self.behavior_df['immobility'].diff().values == True)[0]
        all_bout_start_idxs = np.concatenate([np.array([0]), idxs_immobility_state_changes])
        all_bout_end_idxs = np.concatenate([idxs_immobility_state_changes, np.array([idxs_immobility_state_changes.shape[0]])])
        immobility_events_matching_duration_criterion = []
        event_id = 0
        for start_idx, end_idx in zip(all_bout_start_idxs, all_bout_end_idxs):
            if (end_idx - start_idx) >= self.fps*min_interval_duration_in_s:
                assert all(self.behavior_df.iloc[start_idx : end_idx, :]['immobility'].values), 'Not all values were True!'
                immobility_event = EventBout2D(event_id = event_id, start_index = start_idx, end_idx = end_ix, fps = self.fps, event_type = event_type)
                immobility_events_matching_duration_criterion.append(immobility_event)
                event_id += 1
        return immobility_events_matching_duration_criterion
    
    
    def _add_event_bouts_to_behavior_df(self, events: List[EventBout2D]) -> None:
        assert events[0].event_type not in list(self.behavior_df.columns), f'{events[0].event_type} was already a column in self.behavior_df!'
        self.behavior_df[events[0].event_type] = np.nan
        self.behavior_df[f'{events[0].event_type}_bout_id'] = np.nan
        for event_id, event_bout in enumerate(events):
            # do we need to add +1 to the end_idx?
            self.behavior_df.iloc[event_bout.start_idx : event_bout.end_idx, -2] = True
            self.behavior_df.iloc[event_bout.start_idx : event_bout.end_idx, -1] = event_id
        

    def _add_orientation_to_behavior_df(self, bodyparts_for_direction_front_to_back: List[str]) -> None:
        assert len(bodyparts_for_direction_front_to_back) ==2, '"bodyparts_for_direction_front_to_back" must be a list of exact 2 marker_ids!'
        front_marker_id = bodyparts_for_direction_front_to_back[0]
        back_marker_id = bodyparts_for_direction_front_to_back[1]
        self.behavior_df.loc[self.bodyparts[front_marker_id].df['x'] > self.bodyparts[back_marker_id].df['x'], 'facing_towards_open_end'] = True
    

   
    
    def _get_low_likelihood_interval_border_idxs(self, likelihood_series: pd.Series, max_interval_length: int, min_likelihood_threshold: float=0.5) -> List[Tuple[int, int]]:
        all_low_likelihood_idxs = np.where(likelihood_series.values < min_likelihood_threshold)[0]
        last_idxs_of_idx_intervals = np.where(np.diff(all_low_likelihood_idxs) > 1)[0]
        all_interval_end_idxs = np.concatenate([last_idxs_of_idx_intervals, np.array([all_low_likelihood_idxs.shape[0] - 1])])
        all_interval_start_idxs = np.concatenate([np.asarray([0]), last_idxs_of_idx_intervals + 1])
        interval_lengths = all_interval_end_idxs - all_interval_start_idxs
        idxs_of_intervals_matching_length_criterion = np.where(interval_lengths <= max_interval_length)[0]
        selected_interval_start_idxs = all_interval_start_idxs[idxs_of_intervals_matching_length_criterion]
        selected_interval_end_idxs = all_interval_end_idxs[idxs_of_intervals_matching_length_criterion]
        interval_border_idxs = []
        for start_idx, end_idx in zip(selected_interval_start_idxs, selected_interval_end_idxs):
            border_idxs = (all_low_likelihood_idxs[start_idx], all_low_likelihood_idxs[end_idx])
            interval_border_idxs.append(border_idxs)
        return interval_border_idxs
            
        
       
                
class Bodypart2D():
    """
    Class that contains information for one single Bodypart.
    
    Attributes:
        self.id(str): Deeplabcut label of the bodypart
    """
    
    @property
    def exclusion_criteria(self) -> Dict:
        return {'likelihood_threshold': 0.5,
                              'min_x': -5,
                              'max_x': 55,
                              'min_y': -3,
                              'max_y': 7}

    
    def __init__(self, bodypart_id: str, df: pd.DataFrame, fps: int)->None:
        """ 
        Constructor for class Bodypart. 
        
        Since the points in df_raw represent coordinates in the distorted dataframe, we use df_undistort for calculations.
        
        Parameters:
            bodypart_id(str): unique id of marker
        """
        self.id = bodypart_id
        sliced_df = self._slice_df(df = df)
        self.df = self._apply_exclusion_criteria(df = sliced_df, exclusion_criteria = self.exclusion_criteria)
        self.fps = fps
        self.framerate = 1/fps

        
    def _slice_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function, that extracts the coordinates of a single bodypart.
        
        Parameters:
            df(pandas.DataFrame): the full dataframe of the recording with all bodyparts
        """
        df_input = df.copy()
        data = {'x': df_input.loc[:, f'{self.id}_x'], 
                'y': df_input.loc[:, f'{self.id}_y'], 
                'likelihood': df_input.loc[:, f'{self.id}_likelihood']}
        return pd.DataFrame(data = data)
        
        
    def _apply_exclusion_criteria(self, df: pd.DataFrame, exclusion_criteria: Dict) -> None:
        df.loc[df['likelihood'] < exclusion_criteria['likelihood_threshold'], :] = np.nan
        for coord in ['x', 'y']:
            df.loc[df[coord] < exclusion_criteria[f'min_{coord}'], :] = np.nan
            df.loc[df[coord] > exclusion_criteria[f'max_{coord}'], :] = np.nan
        return df

        
    def calculate_speed_and_identify_immobility(self, sliding_window_size: int, immobility_threshold: float) -> None:
        self._add_speed_to_df()
        self._add_rolling_speed_to_df(sliding_window_size = sliding_window_size)
        self._add_immobility_to_df(immobility_threshold = immobility_threshold)
    
    
    def _add_speed_to_df(self)->None:
        self.df.loc[:, 'speed_cm_per_s'] = (self.df.loc[:, 'x'].diff()**2 + self.df.loc[:, 'y'].diff()**2)**0.5 / self.framerate              
        
    
    def _add_rolling_speed_to_df(self, sliding_window_size: int) -> None:
        min_periods = int(sliding_window_size * 0.66)
        self.df.loc[:, 'rolling_speed_cm_per_s'] = self.df.loc[:, 'speed_cm_per_s'].rolling(sliding_window_size, min_periods = min_periods, center = True).mean()

    
    def _add_immobility_to_df(self, immobility_threshold: float) -> None:
        self.df.loc[:, 'immobility'] = False
        self.df.loc[self.df['rolling_speed_cm_per_s'] < immobility_threshold, 'immobility'] = True     



class EventBout2D():
    """
    Class, that contains start_index, end_index, duration and position of an event.
    It doesn't differ to class EventBout for now, so for the future this classes could be merged.
    
    Attributes:
        self.start_index(int): index of event onset
        self.end_index(int): index of event ending
    """
    def __init__(self, start_index: int, end_index: int, fps: int, event_type: Optional[str]) -> None:
        """
        Constructor of class EventBout that sets the attributes start_ and end_index.
        
        Parameters: 
            start_index(int): index of event onset
            end_index(Optional[int]): index of event ending (if event is not only a single frame)
        """
        self.start_index = start_index
        if end_index != None:
            self.end_index = end_index
            self.duration = (self.end_index - self.start_index)/fps
        else:
            self.end_index = start_index
            self.duration = 0
        self._create_dict()
        self.id = None
        self.dict['start_index'] = start_index

    @property
    def freezing_threshold(self) -> float:
        """ Arbitrary chosen threshold in seconds to check for freezing."""
        return 1.

    def check_direction(self, facing_towards_open_end: pd.Series)->None:
        """ 
        Function, that checks the direction of the mouse at the start_index.
        
        Parameters:
            facing_towards_open_end(pandas.Series): Series with boolean values for each frame.
        """
        self.facing_towards_open_end = facing_towards_open_end.iloc[self.start_index]
        self.dict['facing_towards_open_end']=self.facing_towards_open_end

    def check_that_freezing_threshold_was_reached(self)->None:
        """
        Function, that calculates the duration of an event and checks, whether it exceeded the freezing_threshold.
        
        Parameters:
            fps(int): fps of the recording
        """
        self.freezing_threshold_reached = False
        if self.duration > self.freezing_threshold:
            self.freezing_threshold_reached = True
        self.dict['freezing_threshold_reached']=self.freezing_threshold_reached
        self.dict['duration_in_s'] = self.duration

    def get_position(self, centerofgravity: Bodypart2D)->None:
        """
        Function, that saves the position of the mouse at the start_index.
        
        Parameters:
            centerofgravity(Bodypart): object centerofgravity, its df x column is used to extract the mouse position
        """
        self.x_position=centerofgravity.df.loc[self.start_index:self.end_index, 'x'].median()
        self.dict['x_position']=self.x_position

    def _create_dict(self)->None:
        """
        Function that sets the attribut self.dict as Dictionary.
        """
        self.dict = {}
    
class EventSeries(ABC):
    @property
    def merge_threshold(self)->float:
        #in seconds
        return 0.2
    
    
    def __init__(self, range_end: int, events: List, event_type: str, fps: int, range_start: int = 0):
        self.events = self._merge_events(events = events, fps = fps)
        self.event_type = event_type
        
    def _merge_events(self, events: List, fps:int)->List:
        events_to_keep = []  
        for i in range(len(events)-1):
            try:
                events[i+1]
            except IndexError:
                break
            if ((events[i+1].start_index - events[i].end_index)/fps) < self.merge_threshold:
                j = i + 1
                try: 
                    events[j+1]
                    while ((events[j].start_index - events[i].end_index)/fps) < self.merge_threshold:
                        j += 1
                except IndexError: 
                    j -= 1
                events_to_keep.append(EventBout2D(start_index = events[i].start_index, end_index = events[j].end_index, fps=fps))
                for n in range(i, j):
                    try:
                        events.pop(n+1)
                    except IndexError:
                        break
            else:
                events_to_keep.append(events[i])
        return events_to_keep
        
    def run_basic_operations_on_events(self, facing_towards_open_end: pd.Series, centerofgravity: Bodypart2D):
        for i, event in enumerate(self.events):
            event.check_direction(facing_towards_open_end=facing_towards_open_end)
            event.check_that_freezing_threshold_was_reached()
            event.get_position(centerofgravity= centerofgravity)
            event.id = i
        
            
    def calculate_statistics(self):
        data = [{'duration': event.duration, 'x_position': event.x_position, 'id': event.id, 'facing_towards_open_end': event.facing_towards_open_end} for event in self.events]
        self.df = pd.DataFrame(data = data)
        self.mean_x_position = self.df['x_position'].mean()
        self.mean_duration = self.df['duration'].mean()
        self.total_duration = self.df['duration'].sum()
        self.total_count = len(self.events)
        self.mean_x_position_facing_open = self.df.loc[self.df['facing_towards_open_end'] == True, 'x_position'].mean()
        self.mean_duration_facing_open = self.df.loc[self.df['facing_towards_open_end'] == True, 'duration'].mean()
        self.total_count_facing_open = len([event for event in self.events if event.facing_towards_open_end])