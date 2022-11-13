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
                   marker_ids_to_compute_center_of_gravity: List[str]=['TailBase', 'Snout']
                   ) -> None:
        self.log = {'critical_markers': marker_ids_to_compute_coverage,
                    'coverage_threshold': coverage_threshold,
                    'max_seconds_to_interpolate': max_seconds_to_interpolate,
                    'likelihood_threshold': likelihood_threshold,
                    'center_of_gravity_based_on': marker_ids_to_compute_center_of_gravity}
        window_length = self._get_max_odd_n_frames_for_time_interval(fps = self.fps, time_interval = max_seconds_to_interpolate)
        all_marker_ids = self._get_all_unique_marker_ids(df = self.full_df_from_file)
        relevant_marker_ids = all_marker_ids
        for marker_id_to_exclude in ['LED5', 'MazeCornerClosedRight', 'MazeCornerClosedLeft', 'MazeCornerOpenRight', 'MazeCornerOpenLeft']:
            relevant_marker_ids.remove(marker_id_to_exclude)        
        smoothed_df = self._smooth_tracked_coords_and_likelihood(marker_ids = relevant_marker_ids, window_length = window_length, polyorder = 3)
        interpolated_df = self._interpolate_low_likelihood_intervals(df = smoothed_df, marker_ids = relevant_marker_ids, max_interval_length = window_length)
        interpolated_df_with_cog = self._add_new_marker_derived_existing_markers(df = interpolated_df,
                                                                                 existing_markers = marker_ids_to_compute_center_of_gravity,
                                                                                 new_marker_id = 'CenterOfGravity',
                                                                                 likelihood_threshold = likelihood_threshold)
        preprocessed_df = self._interpolate_low_likelihood_intervals(df = interpolated_df_with_cog,
                                                                     marker_ids = ['CenterOfGravity'],
                                                                     max_interval_length = window_length)
        
        self.log['coverage_critical_markers'] = self._compute_coverage(df = preprocessed_df,
                                                                         critical_marker_ids = marker_ids_to_compute_coverage,
                                                                         likelihood_threshold = likelihood_threshold)
        self.log['coverage_CenterOfGravity'] = self._compute_coverage(df = preprocessed_df,
                                                                      critical_marker_ids = ['CenterOfGravity'],
                                                                      likelihood_threshold = likelihood_threshold)
        if self.log['coverage_critical_markers'] >= coverage_threshold:
            self.preprocessed_df = preprocessed_df
            #self._instantiate_all_bodypart_objects()
            #self._normalize_coordinate_system()
            #self._run_basic_operations_on_bodyparts()
            #self._get_tracking_performance()
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
        
        

        


    def _create_all_bodyparts(self)->None:
        """
        Function, that creates a Dictionary with all Bodypart objects.
        
        The dictionary uses the label given from Deeplabcut tracking as key for the Bodypart objects.
        It sets the dictionary as self.bodyparts.
        """
        self.bodyparts = {}
        for key in self.full_df_from_file.keys():
            bodypart = key.split('_')[0]
            if bodypart not in self.bodyparts.keys():
                self.bodyparts[bodypart] = Bodypart2D(bodypart_id = bodypart, 
                                                      df = self.full_df_from_file, 
                                                      camera_parameters_for_undistortion=self.camera_parameters_for_undistortion)
                
    
    def _normalize_coordinate_system(self) -> None:
        corners = self._get_corner_coords_with_likelihoods()
        translation_vector = self._get_translation_vector(coords_closed_left = corners['MazeCornerClosedLeft']['coords'])
        best_result = self._evaluate_maze_shape_using_open_corners(corners_and_likelihoods = corners, tolerance = 0.25)
        if best_result['valid']:
            conversion_factor = self._get_conversion_factor_px_to_cm(coords_point_a = corners[f'MazeCornerClosed{best_result["side_id"]}']['coords'],
                                                                     coords_point_b = corners[f'MazeCornerOpen{best_result["side_id"]}']['coords'],
                                                                     distance_in_cm = 50)
            rotation_angle = self._get_rotation_angle_with_open_corner(corners = corners,
                                                                       side_id = best_result['side_id'],
                                                                       translation_vector = translation_vector,
                                                                       conversion_factor = conversion_factor)
        else:
            conversion_factor = self._get_conversion_factor_px_to_cm(coords_point_a = corners['MazeCornerClosedLeft']['coords'],
                                                                     coords_point_b = corners['MazeCornerClosedRight']['coords'],
                                                                     distance_in_cm = 4)
            rotation_angle = self._get_rotation_angle_with_closed_corners_only(corners = corners,
                                                                               translation_vector = translation_vector,
                                                                               conversion_factor = conversion_factor)
        for bodypart in self.bodyparts.values():
            bodypart.normalize_df(translation_vector = translation_vector, rotation_angle = rotation_angle, conversion_factor = conversion_factor)          
        # kept from previous version:
        coverage_threshold = 0.9
        likelihood_threshold = 0.6
        self.log = {}
        self.log['likelihood_threshold'] = likelihood_threshold
        self.log['rotation_angle'] = rotation_angle
        self.log['conversion_factor'] = conversion_factor
        normalized_corner_coords = []
        for corner_marker_id in corners.keys():
            normalized_coords = self._normalize_coords(coords = corners[corner_marker_id]['coords'],
                                                       translation_vector = translation_vector,
                                                       conversion_factor = conversion_factor,
                                                       rotation_angle = rotation_angle)
            normalized_corner_coords.append(normalized_coords)
        self.log['plotting_marker'] = normalized_corner_coords
        self.log['closed_right'] = self._normalize_coords(coords = corners['MazeCornerClosedRight']['coords'],
                                                       translation_vector = translation_vector,
                                                       conversion_factor = conversion_factor,
                                                       rotation_angle = rotation_angle)
        self.log['closed_left'] = self._normalize_coords(coords = corners['MazeCornerClosedLeft']['coords'],
                                                       translation_vector = translation_vector,
                                                       conversion_factor = conversion_factor,
                                                       rotation_angle = rotation_angle)
        self.log['number_frames'] = self.bodyparts['Snout'].df.shape[0]

        
    def _get_corner_coords_with_likelihoods(self) -> Dict:
        corner_coords_with_likelihood = {}
        for corner_marker_id in ['MazeCornerClosedRight', 'MazeCornerClosedLeft', 'MazeCornerOpenRight', 'MazeCornerOpenLeft']:
            xy_coords, min_likelihood = self._get_most_reliable_marker_position_with_likelihood(df = self.bodyparts[corner_marker_id].df_undistort)
            corner_coords_with_likelihood[corner_marker_id] = {'coords': xy_coords, 'min_likelihood': min_likelihood}
        return corner_coords_with_likelihood


    def _get_most_reliable_marker_position_with_likelihood(self, df: pd.DataFrame, percentile: float=99.95) -> Tuple[np.array, float]:
        likelihood_threshold = np.nanpercentile(df['likelihood'].values, percentile)
        df_most_reliable_frames = df.loc[df['likelihood'] >= likelihood_threshold].copy()
        most_reliable_x, most_reliable_y = df_most_reliable_frames['x'].median(), df_most_reliable_frames['y'].median()
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
        print('using open corner')
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
        print('using closed corners only')
        translated_closed_left = corners['MazeCornerClosedLeft']['coords'] + translation_vector
        translated_closed_right = corners['MazeCornerClosedRight']['coords'] + translation_vector
        target_rotated_closed_right = np.asarray([0, 4 / conversion_factor])
        
        length_a = self._get_distance_between_two_points(translated_closed_right, target_rotated_closed_right) * conversion_factor
        length_b = self._get_distance_between_two_points(translated_closed_left, translated_closed_right) * conversion_factor
        length_c = 4
        angle = math.acos((length_b**2 + length_c**2 - length_a**2) / (2 * length_b * length_c))
        return angle


    def _normalize_coords(self, coords: np.ndarray, translation_vector: np.ndarray, conversion_factor: float, rotation_angle: float) -> np.ndarray:
        translated_coords = coords + translation_vector
        converted_coords = translated_coords * conversion_factor
        rotated_coords = self._rotate_coords(coords = converted_coords, rotation_angle = rotation_angle)
        return rotated_coords
        
        
    def _rotate_coords(self, coords: np.ndarray, rotation_angle: float) -> np.ndarray:
        cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
        rotated_x = coords[0] * cos_theta - coords[1] * sin_theta
        rotated_y = coords[0] * sin_theta + coords[1] * cos_theta
        return np.asarray([rotated_x, rotated_y])


       
                
class Bodypart2D():
    """
    Class that contains information for one single Bodypart.
    
    Attributes:
        self.id(str): Deeplabcut label of the bodypart
    """
    def __init__(self, bodypart_id: str, df: pd.DataFrame, camera_parameters_for_undistortion: Dict)->None:
        """ 
        Constructor for class Bodypart. 
        
        Since the points in df_raw represent coordinates in the distorted dataframe, we use df_undistort for calculations.
        
        Parameters:
            bodypart_id(str): unique id of marker
            camera_parameters_for_undistortion(Dict): storage of intrinsic camera parameters K and D as well as the size of the recorded video
        """
        self.id = bodypart_id
        self._get_sliced_df(df = df)
        self._undistort_points(camera_parameters_for_undistortion)
        
        
    def _get_sliced_df(self, df: pd.DataFrame)->None:
        """
        Function, that extracts the coordinates of a single bodypart.
        
        Parameters:
            df(pandas.DataFrame): the full dataframe of the recording with all bodyparts
        """
        self.df_raw = pd.DataFrame(data={'x': df.loc[:, self.id + '_x'], 'y': df.loc[:, self.id + '_y'], 'likelihood': df.loc[:, self.id + '_likelihood']})
    
        
    def normalize_df(self, translation_vector: np.array, rotation_angle: float, conversion_factor: float)->None:
        """
        Given the parameters, this function aligns the xyz-coordinate system with the null space.
        
        After translation to zero, rotation around the given angles and axes is performed and the units are converted into cm.
        The normalized dataframe is set as attribut self.df.
        
        Parameter:
            translation_vector(np.Array): vector with offset of xyz to XYZ in each dimension
            rotation_matrix(scipy.spatial.transform.Rotation): Rotation matrix obtained from Euler angles
            conversion_factor(float): factor to convert the unspecified unit into cm.
        """
        translated_df = self._translate_df(translation_vector=translation_vector)
        rotated_df = self._rotate_df(rotation_angle=rotation_angle, df=translated_df)
        self.df = self._convert_df_to_cm(conversion_factor=conversion_factor, df=rotated_df)
        self._exclude_frames()
        self._interpolate_low_likelihood_frames()
    
   
    def _translate_df(self, translation_vector: np.array)->pd.DataFrame:
        """
        Function that translates the raw dataframe to the null space.
        
        Parameter:
            translation_vector(np.Array): vector with offset of xy to XY
        Returns:
            translated_df(pandas.DataFrame): the dataframe translated to (0, 0)
        """
        translated_df = self.df_undistort.loc[:, ('x', 'y')] + translation_vector
        return translated_df
    
    def _rotate_df(self, rotation_angle: float, df: pd.DataFrame)->pd.DataFrame:
        """
        Function, that rotates the dataframe in 2D.
        
        Besides calculating the coordinates, the likelihood is added to the Dataframe.
        
        Parameter:
            rotation angle: angle of rotation of the xy to XY coordinate system around the z-axis
            df(pandas.DataFrame): the dataframe that will be rotated.
        Returns:
            rotated_df(pandas.DataFrame): the rotated dataframe
        """
        angle = rotation_angle
        cos_theta, sin_theta = math.cos(angle), math.sin(angle)
        rotated_df = pd.DataFrame()
        rotated_df['x'], rotated_df['y'] = df['x'] * cos_theta - df['y'] * sin_theta, df['x'] * sin_theta + df['y'] * cos_theta
        rotated_df['likelihood']=self.df_raw['likelihood']
        return rotated_df
        
    def _convert_df_to_cm(self, conversion_factor: float, df: pd.DataFrame)->pd.DataFrame:
        """
        The coordinates are converted to cm.
        
        Parameters:
            conversion_factor(float): factor to convert the unspecified unit into cm.
            df(pandas.DataFrame): dataframe with unspecified unit
            
        Returns:
            df(pandas.DataFrame): dataframe with values in cm
        """
        df.loc[:, ('x', 'y')]*=conversion_factor
        return df
        
    def run_basic_operations(self, fps: int)->None:
        """
        Function that calculates Speed and Immobility.
        """
        self._get_speed(fps = fps)
        self._get_rolling_speed()
        self._get_immobility()
        
    def _exclude_frames(self) -> None:
        if self.id == 'centerofgravity':
            self.df.loc[self.df['likelihood']<self.dlc_likelihood_threshold**2, ('x', 'y')] = np.NaN
        else:
            self.df.loc[self.df['likelihood']<self.dlc_likelihood_threshold, ('x', 'y')] = np.NaN
        self.df.loc[(self.df['x'] < -5) | (self.df['x'] > 55) | (self.df['y'] < -1) | (self.df['y'] > 6), ('x', 'y')] = np.NaN
            
    def _interpolate_low_likelihood_frames(self) -> None:
        try:
            self.df = self.df.interpolate(limit = 10, method = 'slinear', order=1)
        except ValueError:
            pass
        
    def check_tracking_stability(self, start_end_index: Optional[Tuple]=(0, None))->float:
        """
        Function, that calculates the percentage of frames, in which the marker was detected with high likelihood.
        
        Parameters:
            start_end_index: range in which the percentage of detected labels above the likelihood threshold should be calculated. If no values are passed, the percentage over the total session is returned.
        
        Returns:
            marker_detected_per_total_frames(float)
        """
        marker_detected_per_total_frames = self.df.loc[start_end_index[0]:start_end_index[1], :].loc[self.df['likelihood']>self.dlc_likelihood_threshold, :].shape[0]/self.df.loc[start_end_index[0]:start_end_index[1], :].shape[0]
        return marker_detected_per_total_frames
    
    def _get_speed(self, fps: int)->None:
        """
        Function, that calculates the speed of the bodypart, based on the framerate.
        
        After creating an empty column with np.NaN values, the speed is calculated 
        as the squareroot of the squared difference between two frames in -x and -y dimension divided by the duration of a frame.
        
        Parameters:
            fps(int): fps of the recording
        """
        self.df.loc[:, 'speed_cm_per_s'] = np.NaN
        self.df.loc[:, 'speed_cm_per_s'] = (np.sqrt(self.df.loc[:, 'x'].diff()**2 + self.df.loc[:, 'y'].diff()**2)) / (1/fps)        
    
    
    def _get_rolling_speed(self)->None:
        """
        Function, that applies a sliding window of the size 5 on the speed.
        """
        self.df.loc[:, 'rolling_speed_cm_per_s'] = np.NaN
        self.df.loc[:, 'rolling_speed_cm_per_s'] = self.df.loc[:, 'speed_cm_per_s'].rolling(5, min_periods=3, center=True).mean()

    @property
    def immobility_threshold(self) -> float:
        """ Arbitrary chosen threshold in cm per s for defining immobility."""
        return 4.
    
    @property
    def dlc_likelihood_threshold(self)->float:
        """ Threshold for likelihood of DLC labels. Values above are considered as good trackings. """
        return 0.6
    
    def _get_immobility(self)->None:
        """
        Function, that checks frame by frame, whether the rolling_speed of the bodypart is below the immobility threshold.
        """
        self.df.loc[:, 'immobility'] = False
        self.df.loc[self.df['rolling_speed_cm_per_s'] < self.immobility_threshold, 'immobility'] = True     
        
    def _detect_steps(self)->None:
        """
        Function, that detects steps as peaks in the speed based on scipy find_peaks.
        """
        speed = self.df["speed_cm_per_s"].copy()
        peaks = find_peaks(speed, prominence=50)
        steps_per_paw = self._create_steps(steps=peaks[0])
        return steps_per_paw
            
        
    def _create_steps(self, steps: List)->List['Step']:
        """
        Function, that creates Step objects for every speed peak inside of a gait event.
        
        Parameters:
            List with start_indices for steps.
            
        Returns:
            List with Step elements.
        """
        return [Step(paw = self.id, start_index = step_index) for step_index in steps]
   
    def _undistort_points(self, camera_parameters_for_undistortion: Dict)->None:
        """
        Function that undistort the coordinates of the tracked points based on the camera intrinsics.
        
        The undistorted coordinates are stored as self.df_undistort attribute.
        understanding the maths behind it: https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html
        
        Parameters:
            camera_parameters_for_undistortion(Dict): storage for intrinsic camera parameters K and D and the size of the recorded video
        """
        points = self.df_raw[['x', 'y']].copy().values
        new_K, _ = cv2.getOptimalNewCameraMatrix(camera_parameters_for_undistortion['K'], camera_parameters_for_undistortion['D'], camera_parameters_for_undistortion['size'], 1, camera_parameters_for_undistortion['size'])
        points_undistorted = cv2.undistortPoints(points, camera_parameters_for_undistortion['K'], camera_parameters_for_undistortion['D'], None, new_K)
        points_undistorted = np.squeeze(points_undistorted)
        self.df_undistort = pd.DataFrame()
        self.df_undistort[['x', 'y']] = points_undistorted
        self.df_undistort['likelihood'] = self.df_raw['likelihood']

    

class EventBout2D():
    """
    Class, that contains start_index, end_index, duration and position of an event.
    It doesn't differ to class EventBout for now, so for the future this classes could be merged.
    
    Attributes:
        self.start_index(int): index of event onset
        self.end_index(int): index of event ending
    """
    def __init__(self, start_index: int, end_index: Optional[int]=None, fps: Optional[int]=None)->None:
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