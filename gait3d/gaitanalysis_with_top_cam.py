from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path, PosixPath
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter


def process_all_dlc_tracking_h5s_with_default_settings(input_directory_with_dlc_tracking_files: Path,
                                                       week_id: int,
                                                       output_directory_to_save_results: Path) -> None:
    filepaths_dlc_unfiltered_h5_trackings = []
    for filepath in input_directory_with_dlc_tracking_files.iterdir():
        if filepath.name.endswith('.h5'):
            if 'filtered' not in filepath.name:
                filepaths_dlc_unfiltered_h5_trackings.append(filepath)
    for filepath in tqdm(filepaths_dlc_unfiltered_h5_trackings):
        recording = RecordingTop(filepath = filepath, week_id = week_id)
        if recording.df_successfully_loaded:
            recording.preprocess()
            if recording.logs['coverage_critical_markers'] >= recording.logs['coverage_threshold']: 
                recording.run_behavioral_analyses()
                recording.export_results(out_dir_path = output_directory_to_save_results)
                recording.inspect_processing()



class EventBout2D():
    """
    Class, that contains start_index, end_index, duration and position of an event.
    It doesn't differ to class EventBout for now, so for the future this classes could be merged.
    
    Attributes:
        self.start_index(int): index of event onset
        self.end_index(int): index of event ending
    """
    
    def __init__(self, event_id: int, start_idx: int, end_idx: int, fps: int, event_type: str) -> None:
        """
        Constructor of class EventBout that sets the attributes start_ and end_index.
        
        Parameters: 
            start_index(int): index of event onset
            end_index(Optional[int]): index of event ending (if event is not only a single frame)
        """
        self.id = event_id
        self.event_type = event_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.duration = ((self.end_idx + 1) - self.start_idx)/fps


        
class RecordingTop():
    """
    Class for Analysing 2D-Position Data of mice in the OpeningTrack.
    
    Attributes:
        full_df_from_file(pandas.DataFrame): the Dataframe containing all bodyparts with x, y-coordinates and likelihood as returned by DLC
        fps(int): fps of the recording
        metadata(Dict): dictionary containing information read from the filename, such as animal_id, recording_date and Opening Track paradigm
    """
       
    @property
    def valid_paradigms(self) -> List[str]:
        return ['OTR', 'OTT', 'OTE']
    
    @property
    def valid_mouse_lines(self) -> List[str]:
        return ['194', '195', '196', '206', '209']
    
    @property
    def valid_recording_weeks(self) -> List[int]:
        return [1, 4, 8, 12, 14]
    
    @property
    def marker_ids_to_exclude_for_smoothing_and_interpolation(self) -> List[str]:
        return ['LED5', 'MazeCornerClosedRight', 'MazeCornerClosedLeft', 'MazeCornerOpenRight', 'MazeCornerOpenLeft']

    
    def __init__(self, filepath: Path, week_id: int)->None:
        """
        Constructor for the Recording2D class.
        
        This function calls functions to get the Dataframe from the csv, that is given as filepath argument and extracts and checks metadata from the filename.
        
        Parameters:
            filepath(pathlib.Path): the filepath to the h5 or csv containing DLC data
            week_id(int): the experimental week in which the recording was performed (1, 4, 8, 12, or 14)
        """
        assert type(filepath) == PosixPath, '"filepath" has to be a pathlib.Path object'
        assert week_id in self.valid_recording_weeks, f'"week_id" = {week_id} is not listed in "valid_recording_weeks": {self.valid_recording_weeks}'
        self.filepath = filepath
        self.week_id = week_id
        self.df_successfully_loaded, self.full_df_from_file = self._attempt_loading_df_from_file(filepath = filepath)
        if self.df_successfully_loaded:
            self.fps = self._get_correct_fps()
            self.framerate = 1/self.fps
            self.metadata = self._retrieve_metadata(filename = filepath.name)
        
        
    def _attempt_loading_df_from_file(self, filepath: Path) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Reads the Dataframe from the h5-file and drops irrelevant columns and rows.
        
        Parameters:
            filepath(pathlib.Path): the path linked to the h5.file
        Returns:
            pandas.DataFrame: the Dataframe containing all bodyparts with x, y-coordinates and likelihood as returned by DLC
        """
        assert filepath.name.endswith('.csv') or filepath.name.endswith('.h5'), 'The filepath you specified is not referring to a .csv or a .h5 file!'
        try:
            if filepath.name.endswith('.csv'):
                df = pd.read_csv(filepath, low_memory = False)
                df = df.drop('scorer', axis=1)
                df.columns = df.iloc[0, :]+ '_' + df.iloc[1, :]
                df = df.drop([0, 1], axis=0)
                df = df.reset_index()
                df = df.drop('index', axis=1)
                df = df.astype(float)
            else:
                df = pd.read_hdf(filepath)
                target_column_names = []
                for marker_id, data_id in zip(df.columns.get_level_values(1), df.columns.get_level_values(2)):
                    target_column_names.append(f'{marker_id}_{data_id}') 
                df.columns = target_column_names
                df = df.astype(float)
            successfully_loaded = True
        except:
            successfully_loaded = False
            df = None
        return successfully_loaded, df
    

    def _get_correct_fps(self) -> int:
        if self.full_df_from_file.shape[0] > 25_000:
            fps = 80
        else:
            fps = 30
        return fps


    def _retrieve_metadata(self, filename: str)->Dict:
        """
        Function, that slices the Filename to get the encoded metadata.
        
        Relying on file naming like this: 196_F7-27_220826_OTT_Bottom_synchronizedDLC_resnet152_OT_BottomCam_finalSep20shuffle1_550000filtered.h5
        
        Parameters:
            filepath(pathlib.Path): the path linked to the h5.file
        Returns:
            Dict: containing date of recording, animal_id and OT paradigm
        """
        filename_slices = filename.split('_')
        animal_line, animal_id, recording_date, paradigm, cam_id = filename_slices[0], filename_slices[1], filename_slices[2], filename_slices[3][0:3], 'Top'
        self._check_metadata(metadata = (animal_line, animal_id, recording_date, paradigm, cam_id))
        return {'recording_date': self.recording_date, 'animal': f'{self.mouse_line}_{self.mouse_id}', 'paradigm': self.paradigm, 'cam': self.cam_id}
    
    
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
        initial_logs_to_add = {'critical_markers_to_compute_coverage': marker_ids_to_compute_coverage,
                               'coverage_threshold': coverage_threshold,
                               'max_seconds_to_interpolate': max_seconds_to_interpolate,
                               'min_likelihood_threshold': likelihood_threshold,
                               'center_of_gravity_based_on': marker_ids_to_compute_center_of_gravity, 
                               'relative_error_tolerance_corner_detection': relative_maze_normalization_error_tolerance}
        window_length = self._get_max_odd_n_frames_for_time_interval(fps = self.fps, time_interval = max_seconds_to_interpolate)
        marker_ids_to_preprocess = self._get_preprocessing_relevant_marker_ids(df = self.full_df_from_file)
        smoothed_df = self._smooth_tracked_coords_and_likelihood(marker_ids = marker_ids_to_preprocess, window_length = window_length, polyorder = 3)
        interpolated_df = self._interpolate_low_likelihood_intervals(df = smoothed_df, marker_ids = marker_ids_to_preprocess, max_interval_length = window_length)
        interpolated_df_with_cog = self._add_new_marker_derived_from_existing_markers(df = interpolated_df,
                                                                                      existing_markers = marker_ids_to_compute_center_of_gravity,
                                                                                      new_marker_id = 'CenterOfGravity',
                                                                                      likelihood_threshold = likelihood_threshold)
        preprocessed_df = self._interpolate_low_likelihood_intervals(df = interpolated_df_with_cog,
                                                                     marker_ids = ['CenterOfGravity'],
                                                                     max_interval_length = window_length)
        coverage_critical_markers = self._compute_coverage(df = preprocessed_df,
                                                           critical_marker_ids = marker_ids_to_compute_coverage,
                                                           likelihood_threshold = likelihood_threshold)
        initial_logs_to_add['coverage_critical_markers'] = coverage_critical_markers
        self._add_to_logs(logs_to_add = initial_logs_to_add)
        if coverage_critical_markers >= coverage_threshold:
            normalization_params = self._get_parameters_to_normalize_maze_coordinates(df = preprocessed_df,
                                                                                      relative_error_tolerance = relative_maze_normalization_error_tolerance)
            self.normalized_df = self._normalize_df(df = preprocessed_df, normalization_parameters = normalization_params)
            self.bodyparts = self._create_bodypart_objects()
            normalized_maze_corner_coords = self._get_normalized_maze_corners(normalization_parameters = normalization_params)
            coverage_center_of_gravity = self._compute_coverage(df = preprocessed_df,
                                                                critical_marker_ids = ['CenterOfGravity'],
                                                                likelihood_threshold = likelihood_threshold)
            additional_logs_to_add = {'coverage_CenterOfGravity': coverage_center_of_gravity}
            for key, value in normalization_params.items():
                additional_logs_to_add[key] = value
            for key, value in normalized_maze_corner_coords.items():
                additional_logs_to_add[f'normalized_{key}_coords'] = value
            self._add_to_logs(logs_to_add = additional_logs_to_add)


    def _add_to_logs(self, logs_to_add: Dict) -> None:
        if hasattr(self, 'logs') == False:
            self.logs = {}
        for key, value in logs_to_add.items():
            assert key not in self.logs.keys(), f'{key} is already in self.logs.keys - adding it would result in overwriting the previous entry; please ensure unique naming'
            self.logs[key] = value           
            
            
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
        short_low_likelihood_interval_border_idxs = self._get_interval_border_idxs(all_matching_idxs = all_low_likelihood_idxs,
                                                                                   max_interval_duration = max_interval_length*self.framerate)
        return short_low_likelihood_interval_border_idxs
       
    
    def _get_interval_border_idxs(self,
                                  all_matching_idxs: np.ndarray, 
                                  min_interval_duration: Optional[float]=None, 
                                  max_interval_duration: Optional[float]=None,
                                 ) -> List[Tuple[int, int]]:
        interval_border_idxs = []
        if all_matching_idxs.shape[0] >= 1:
            step_idxs = np.where(np.diff(all_matching_idxs) > 1)[0]
            step_end_idxs = np.concatenate([step_idxs, np.array([all_matching_idxs.shape[0] - 1])])
            step_start_idxs = np.concatenate([np.array([0]), step_idxs + 1])
            interval_start_idxs = all_matching_idxs[step_start_idxs]
            interval_end_idxs = all_matching_idxs[step_end_idxs]
            for start_idx, end_idx in zip(interval_start_idxs, interval_end_idxs):
                interval_frame_count = (end_idx+1) - start_idx
                interval_duration = interval_frame_count * self.framerate          
                if (min_interval_duration != None) and (max_interval_duration != None):
                    append_interval = min_interval_duration <= interval_duration <= max_interval_duration 
                elif min_interval_duration != None:
                    append_interval = min_interval_duration <= interval_duration
                elif max_interval_duration != None:
                    append_interval = interval_duration <= max_interval_duration
                else:
                    append_interval = True
                if append_interval:
                    interval_border_idxs.append((start_idx, end_idx))
        return interval_border_idxs    
    

    def _add_new_marker_derived_from_existing_markers(self, df: pd.DataFrame, existing_markers: List[str], new_marker_id: str, likelihood_threshold: float = 0.5)->None:
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
            logs_to_add = {'maze_normalization_based_on': f'MazeCornerClosed{side_id}_and_MazeCornerOpen{side_id}'}
            conversion_factor = self._get_conversion_factor_px_to_cm(coords_point_a = corners[f'MazeCornerClosed{side_id}']['coords'],
                                                                     coords_point_b = corners[f'MazeCornerOpen{side_id}']['coords'],
                                                                     distance_in_cm = 50)
            rotation_angle = self._get_rotation_angle_with_open_corner(corners = corners,
                                                                       side_id = best_result['side_id'],
                                                                       translation_vector = translation_vector,
                                                                       conversion_factor = conversion_factor)
        else:
            logs_to_add = {'maze_normalization_based_on': f'MazeCornerClosedRight_and_MazeCornerClosedLeft'}
            conversion_factor = self._get_conversion_factor_px_to_cm(coords_point_a = corners['MazeCornerClosedLeft']['coords'],
                                                                     coords_point_b = corners['MazeCornerClosedRight']['coords'],
                                                                     distance_in_cm = 4)
            rotation_angle = self._get_rotation_angle_with_closed_corners_only(corners = corners,
                                                                               translation_vector = translation_vector,
                                                                               conversion_factor = conversion_factor)
        self._add_to_logs(logs_to_add = logs_to_add)
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
                                bodyparts_critical_for_freezing: List[str]=['Snout', 'CenterOfGravity'],
                                bodyparts_for_direction_front_to_back: List[str]=['Snout', 'CenterOfGravity'],
                                immobility_max_rolling_speed: float=2.0,
                                immobility_min_duration: float=0.1,
                                freezing_min_duration: float=0.5,
                                gait_min_rolling_speed: float=3.0,
                                gait_min_duration: float=0.5,
                                gait_disruption_max_time_to_immobility: float=0.15,
                                merge_events_max_inbetween_time: float=0.15,
                                bodyparts_to_include_in_behavior_df = ['CenterOfGravity', 'Snout', 'TailBase'],
                               ) -> None:
        # parameter selection for immobility speed threshold and freezing min duration based on:
        # https://www.sciencedirect.com/science/article/pii/S0960982216306182
        sliding_window_size = int(round(self._get_max_odd_n_frames_for_time_interval(fps = self.fps, time_interval = 0.5) / 2, 0))
        logs_to_add = {'bodyparts_checked_to_infer_immobility': bodyparts_critical_for_freezing,
                       'bodypart_used_to_identify_front': bodyparts_for_direction_front_to_back[0],
                       'bodypart_used_to_identify_back': bodyparts_for_direction_front_to_back[1],
                       'immobility_max_rolling_speed' : immobility_max_rolling_speed,
                       'immobility_min_duration': immobility_min_duration,
                       'freezing_min_duration': freezing_min_duration, 
                       'gait_min_rolling_speed': gait_min_rolling_speed, 
                       'gait_min_duration': gait_min_duration,
                       'gait_disruption_max_time_to_immobility': gait_disruption_max_time_to_immobility,
                       #'merge_events_max_inbetween_time': merge_events_max_inbetween_time,
                       'sliding_window_size_to_compute_speed': sliding_window_size}
        self._add_to_logs(logs_to_add = logs_to_add)
        for bodypart in self.bodyparts.values():
            bodypart.calculate_speed_and_identify_immobility(sliding_window_size = sliding_window_size, immobility_threshold = immobility_max_rolling_speed)
        self.behavior_df = self._create_behavior_df(bodyparts_to_include = bodyparts_to_include_in_behavior_df)
        self._add_orientation_to_behavior_df(bodyparts_for_direction_front_to_back = bodyparts_for_direction_front_to_back)
        self._add_immobility_based_on_several_bodyparts_to_behavior_df(bodyparts_critical_for_freezing = bodyparts_critical_for_freezing)
        immobility_events = self._get_immobility_related_events(min_interval_duration = immobility_min_duration, event_type = 'immobility_bout')
        self._add_event_bouts_to_behavior_df(event_type = 'immobility_bout', events = immobility_events)
        freezing_events = self._get_immobility_related_events(min_interval_duration = freezing_min_duration, event_type = 'freezing_bout')
        self._add_event_bouts_to_behavior_df(event_type = 'freezing_bout', events = freezing_events)
        gait_events = self._get_gait_events(gait_min_rolling_speed = gait_min_rolling_speed, gait_min_duration = gait_min_duration)
        self._add_event_bouts_to_behavior_df(event_type = 'gait_bout', events = gait_events)
        gait_disruption_events = self._get_gait_disruption_events(gait_events = gait_events, 
                                                                  gait_disruption_max_time_to_immobility = gait_disruption_max_time_to_immobility)
        self._add_event_bouts_to_behavior_df(event_type = 'gait_disruption_bout', events = gait_disruption_events)      

        
    def _create_behavior_df(self, bodyparts_to_include: List[str]) -> pd.DataFrame:
        column_names = self._get_column_names(df = self.normalized_df, column_identifiers = ['x', 'y', 'likelihood'], marker_ids = bodyparts_to_include)
        return self.normalized_df[column_names].copy()
        
        
    def _add_orientation_to_behavior_df(self, bodyparts_for_direction_front_to_back: List[str]) -> None:
        assert len(bodyparts_for_direction_front_to_back) ==2, '"bodyparts_for_direction_front_to_back" must be a list of exact 2 marker_ids!'
        front_marker_id = bodyparts_for_direction_front_to_back[0]
        back_marker_id = bodyparts_for_direction_front_to_back[1]
        self.behavior_df.loc[self.bodyparts[front_marker_id].df['x'] > self.bodyparts[back_marker_id].df['x'], 'facing_towards_open_end'] = True
        
        
    def _add_immobility_based_on_several_bodyparts_to_behavior_df(self, bodyparts_critical_for_freezing: List[str]) -> None:
        # very similar to 
        # can they be combined? different operators for check, though
        valid_idxs_per_marker_id = []
        for bodypart_id in bodyparts_critical_for_freezing:
            tmp_df = self.bodyparts[bodypart_id].df.copy()
            valid_idxs_per_marker_id.append(tmp_df.loc[tmp_df['immobility'] == True].index.values)
        shared_valid_idxs_for_all_markers = valid_idxs_per_marker_id[0]
        if len(valid_idxs_per_marker_id) > 1:
            for next_set_of_valid_idxs in valid_idxs_per_marker_id[1:]:
                shared_valid_idxs_for_all_markers = np.intersect1d(shared_valid_idxs_for_all_markers, next_set_of_valid_idxs)
        self.behavior_df.loc[shared_valid_idxs_for_all_markers, 'immobility'] = True        
        
        
    def _get_immobility_related_events(self, min_interval_duration: float, event_type: str) -> List[EventBout2D]:
        all_immobility_idxs = np.where(self.behavior_df['immobility'].values == True)[0]
        immobility_interval_border_idxs = self._get_interval_border_idxs(all_matching_idxs = all_immobility_idxs, min_interval_duration = min_interval_duration)
        immobility_related_events = self._create_event_objects(interval_border_idxs = immobility_interval_border_idxs, event_type = event_type)
        return immobility_related_events        
        
        
    def _create_event_objects(self, interval_border_idxs: List[Tuple[int, int]], event_type: str) -> List[EventBout2D]:
        events = []
        event_id = 0
        for start_idx, end_idx in interval_border_idxs:
            single_event = EventBout2D(event_id = event_id, start_idx = start_idx, end_idx = end_idx, fps = self.fps, event_type = event_type)
            events.append(single_event)
            event_id += 1
        return events         


    def _add_event_bouts_to_behavior_df(self, event_type: str, events: List[EventBout2D]) -> None:
        assert event_type not in list(self.behavior_df.columns), f'{event_type} was already a column in self.behavior_df!'
        self.behavior_df[event_type] = np.nan
        self.behavior_df[f'{event_type}_id'] = np.nan
        self.behavior_df[f'{event_type}_duration'] = np.nan
        if len(events) > 0:
            for event_bout in events:
                assert event_bout.event_type == event_type, f'Event types didn´t match! Expected {event_type} but found {event_bout.event_type}.'
                self.behavior_df.iloc[event_bout.start_idx : event_bout.end_idx + 1, -3] = True
                self.behavior_df.iloc[event_bout.start_idx : event_bout.end_idx + 1, -2] = event_bout.id
                self.behavior_df.iloc[event_bout.start_idx : event_bout.end_idx + 1, -1] = event_bout.duration

        
    def _get_gait_events(self, gait_min_rolling_speed: float, gait_min_duration: float) -> List[EventBout2D]:
        idxs_with_sufficient_speed = np.where(self.bodyparts['CenterOfGravity'].df['rolling_speed_cm_per_s'].values >= gait_min_rolling_speed)[0]
        gait_interval_border_idxs = self._get_interval_border_idxs(all_matching_idxs = idxs_with_sufficient_speed, min_interval_duration = gait_min_duration)
        gait_events = self._create_event_objects(interval_border_idxs = gait_interval_border_idxs, event_type = 'gait_bout')
        return gait_events
        
    
    def _get_gait_disruption_events(self, gait_events: List[EventBout2D], gait_disruption_max_time_to_immobility: float) -> List[EventBout2D]:
        n_frames_max_distance = int(gait_disruption_max_time_to_immobility * self.fps)
        gait_disruption_interval_border_idxs = []
        for gait_bout in gait_events:
            end_idx = gait_bout.end_idx
            unique_immobility_bout_values = self.behavior_df.loc[end_idx : end_idx + n_frames_max_distance + 1, 'immobility_bout'].unique()
            if True in unique_immobility_bout_values:
                closest_immobility_bout_id = self.behavior_df.loc[end_idx : end_idx + n_frames_max_distance + 1, 'immobility_bout_id'].dropna().unique().min()
                immobility_interval_border_idxs = self._get_interval_border_idxs_from_event_type_and_id(event_type = 'immobility_bout', event_id = closest_immobility_bout_id)
                gait_disruption_interval_border_idxs.append(immobility_interval_border_idxs)
        gait_disruption_events = self._create_event_objects(interval_border_idxs = gait_disruption_interval_border_idxs, event_type = 'gait_disruption_bout')
        return gait_disruption_events
                
                
    def _get_interval_border_idxs_from_event_type_and_id(self, event_type: str, event_id: int) -> Tuple[int, int]:
        interval_idxs = self.behavior_df.loc[self.behavior_df[f'{event_type}_id'] == event_id].index.values
        return interval_idxs[0], interval_idxs[-1]
    
    
    def export_results(self, out_dir_path: Path) -> None:
        dfs_to_export = {'immobility_bouts': self._export_immobility_related_bouts(df = self.behavior_df, event_type = 'immobility_bout'),
                         'freezing_bouts': self._export_immobility_related_bouts(df = self.behavior_df, event_type = 'freezing_bout'), 
                         'gait_disruption_bouts': self._export_immobility_related_bouts(df = self.behavior_df, event_type = 'gait_disruption_bout'), 
                         'gait_bouts': self._export_gait_related_bouts(df = self.behavior_df, event_type = 'gait_bout')}
        dfs_to_export['session_overview'] = self._create_session_overview_df(dfs_to_export_with_individual_bout_dfs = dfs_to_export)
        dfs_to_export['parameter_settings'] = self._create_parameter_settings_df()
        self.base_output_filepath = out_dir_path.joinpath(f'{self.metadata["animal"]}_{self.metadata["paradigm"]}_week-{self.week_id}')
        self._write_xlsx_file_to_disk(dfs_to_export = dfs_to_export)   

                                           
    def _export_immobility_related_bouts(self, df: pd.DataFrame, event_type: str) -> pd.DataFrame:
        results_per_event = {'bout_id': [],
                            'duration': [],
                            'CenterOfGravity_x_at_bout_start': [],
                            'towards_open_at_bout_start': [],
                            'distance_covered_cm': [], 
                            'start_time': [],
                            'end_time': []}
        results_per_event['bout_id'] = self._get_all_bout_ids(df = df, event_type = event_type)
        if len(results_per_event['bout_id']) >= 1:
            results_per_event['duration'] = self._get_bout_duration_per_bout_id(df = df, event_type = event_type, event_ids = results_per_event['bout_id'])
            x_positions_center_of_gravity_at_interval_borders = self._get_column_values_at_event_borders(df = df,
                                                                                                    event_type = event_type,
                                                                                                    event_ids = results_per_event['bout_id'],
                                                                                                    column_name = 'CenterOfGravity_x')
            results_per_event['CenterOfGravity_x_at_bout_start'] = x_positions_center_of_gravity_at_interval_borders[:, 0]
            direction_towards_open_at_interval_borders = self._get_column_values_at_event_borders(df = df,
                                                                                            event_type = event_type,
                                                                                            event_ids = results_per_event['bout_id'],
                                                                                            column_name = 'facing_towards_open_end')
            results_per_event['towards_open_at_bout_start'] = direction_towards_open_at_interval_borders[:, 0]
            results_per_event['distance_covered_cm'] = self._get_distance_covered_per_event(df = df, 
                                                                                       event_type = event_type,
                                                                                       event_ids = results_per_event['bout_id'],
                                                                                       marker_id = 'CenterOfGravity')
            bout_start_and_end_idxs = self._get_interval_start_and_end_idxs_per_event(df = df, event_type = event_type, event_ids = results_per_event['bout_id'])
            results_per_event['start_time'] = bout_start_and_end_idxs[:, 0]
            results_per_event['end_time'] = bout_start_and_end_idxs[:, 1]
        return pd.DataFrame(data = results_per_event)
    
       

    def _export_gait_related_bouts(self, df: pd.DataFrame, event_type: str) -> pd.DataFrame:
        results_per_event = {'bout_id': [],
                            'duration': [],
                            'CenterOfGravity_x_at_bout_end': [],
                            'towards_open_at_bout_end': [],
                            #'mean_rolling_speed_cm_per_s': [],
                            'distance_covered_cm': [], 
                            'start_time': [],
                            'end_time': []}
        results_per_event['bout_id'] = self._get_all_bout_ids(df = df, event_type = event_type)
        if len(results_per_event['bout_id']) >= 1:
            results_per_event['duration'] = self._get_bout_duration_per_bout_id(df = df, event_type = event_type, event_ids = results_per_event['bout_id'])
            x_positions_center_of_gravity_at_interval_borders = self._get_column_values_at_event_borders(df = df,
                                                                                                    event_type = event_type,
                                                                                                    event_ids = results_per_event['bout_id'],
                                                                                                    column_name = 'CenterOfGravity_x')
            results_per_event['CenterOfGravity_x_at_bout_end'] = x_positions_center_of_gravity_at_interval_borders[:, 1]
            direction_towards_open_at_interval_borders = self._get_column_values_at_event_borders(df = df,
                                                                                            event_type = event_type,
                                                                                            event_ids = results_per_event['bout_id'],
                                                                                            column_name = 'facing_towards_open_end')
            results_per_event['towards_open_at_bout_end'] = direction_towards_open_at_interval_borders[:, 1]
            """
            results_per_event['mean_rolling_speed_cm_per_s'] = self._get_mean_column_value_per_event(df = df,
                                                                                                event_type = event_type,
                                                                                                event_ids = results_per_event['bout_id'],
                                                                                                column_name = 'CenterOfGravity_rolling_speed_cm_per_s')
            """
            results_per_event['distance_covered_cm'] = self._get_distance_covered_per_event(df = df, 
                                                                                       event_type = event_type,
                                                                                       event_ids = results_per_event['bout_id'],
                                                                                       marker_id = 'CenterOfGravity')
            bout_start_and_end_idxs = self._get_interval_start_and_end_idxs_per_event(df = df, event_type = event_type, event_ids = results_per_event['bout_id'])
            results_per_event['start_time'] = bout_start_and_end_idxs[:, 0]
            results_per_event['end_time'] = bout_start_and_end_idxs[:, 1]
        return pd.DataFrame(data = results_per_event)

    
    def _get_distance_covered_per_event(self, df: pd.DataFrame, event_type: str, event_ids: List[float], marker_id: str) -> List[float]:
        distances_per_event = []
        for event_id in event_ids:
            df_tmp = df.loc[df[f'{event_type}_id'] == event_id].copy()
            distances_per_event.append(((df_tmp[f'{marker_id}_x'].diff()**2 + df_tmp[f'{marker_id}_y'].diff()**2)**0.5).cumsum().iloc[-1])
        return distances_per_event


    def _get_mean_column_value_per_event(self, df: pd.DataFrame, event_type: str, event_ids: List[float], column_name: str) -> List[float]:
        mean_values = []
        for event_id in event_ids:
            mean_values.append(df.loc[df[f'{event_type}_id'] == event_id, column_name].mean())
        return mean_values


    def _get_all_bout_ids(self, df: pd.DataFrame, event_type: str) -> np.ndarray:
        return df[f'{event_type}_id'].dropna().unique()


    def _get_bout_duration_per_bout_id(self, df: pd.DataFrame, event_type: str, event_ids: List[float]) -> List[float]:
        durations = []
        for event_id in event_ids:
            durations.append(df.loc[df[f'{event_type}_id'] == event_id, f'{event_type}_duration'].iloc[0])
        return durations


    def _get_column_values_at_event_borders(self, df: pd.DataFrame, event_type: str, event_ids: List[float], column_name: str) -> np.ndarray:
        values_at_interval_borders = []
        for event_id in event_ids:
            start_value = df.loc[df[f'{event_type}_id'] == event_id, column_name].iloc[0]
            end_value = df.loc[df[f'{event_type}_id'] == event_id, column_name].iloc[-1]
            values_at_interval_borders.append((start_value, end_value))
        return np.asarray(values_at_interval_borders)


    def _get_interval_start_and_end_idxs_per_event(self, df: pd.DataFrame, event_type: str, event_ids: List[float]) -> np.ndarray:
        interval_border_idxs = []
        for event_id in event_ids:
            start_time, end_time = df.loc[df[f'{event_type}_id'] == event_id].index.values[[0, -1]]*self.framerate
            interval_border_idxs.append((start_time, end_time))
        return np.asarray(interval_border_idxs)


    def _create_session_overview_df(self, dfs_to_export_with_individual_bout_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        session_overview = {'bout_type': [],
                            'total_bouts_count': [],
                            'total_duration': [],
                            'total_distance_covered': [],
                            'mean_duration': [],
                            'mean_distance_covered': [],
                            'mean_CenterOfGravity_x': []}
        for tab_name, df in dfs_to_export_with_individual_bout_dfs.items():
            bout_ids_split_depending_on_direction = self._get_bout_id_splits_depending_on_direction(df = df)
            for split_id, relevant_bout_ids in bout_ids_split_depending_on_direction.items():
                session_overview = self._add_results_to_session_overview(session_overview = session_overview, 
                                                                         df = df, 
                                                                         event_type = tab_name, 
                                                                         event_prefix = split_id, 
                                                                         bout_ids = relevant_bout_ids)
        return pd.DataFrame(data = session_overview)


    def _get_bout_id_splits_depending_on_direction(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        towards_open_column_name = self._get_column_name_from_substring(all_columns = list(df.columns), substring = 'towards_open')
        bout_ids_split_by_direction = {'all': list(df['bout_id'].unique()),
                                       'towards_open': list(df.loc[df[towards_open_column_name] == True, 'bout_id'].unique()),
                                       'towards_closed': list(df.loc[df[towards_open_column_name] != True, 'bout_id'].unique())}
        return bout_ids_split_by_direction                                                                                   

    
    def _get_column_name_from_substring(self, all_columns: List[str], substring: str) -> str:
        matching_column_names = [column_name for column_name in all_columns if substring in column_name]
        assert len(matching_column_names) == 1, \
                f'There should be exactly one match for {substring} - however, {len(matching_column_names)} were found: [{matching_column_names}].'
        return matching_column_names[0]
    

    def _add_results_to_session_overview(self, session_overview: Dict, df: pd.DataFrame, event_type: str, event_prefix: str, bout_ids: List[float]) -> Dict:
        session_overview['bout_type'].append(f'{event_prefix}_{event_type}')
        if len(bout_ids) > 0:
            session_overview['total_bouts_count'].append(len(bout_ids))
            session_overview['total_duration'].append(df.loc[df['bout_id'].isin(bout_ids), 'duration'].cumsum().iloc[-1])
            session_overview['total_distance_covered'].append(df.loc[df['bout_id'].isin(bout_ids), 'distance_covered_cm'].cumsum().iloc[-1])
            session_overview['mean_duration'].append(df.loc[df['bout_id'].isin(bout_ids), 'duration'].mean())
            session_overview['mean_distance_covered'].append(df.loc[df['bout_id'].isin(bout_ids), 'distance_covered_cm'].mean())
            center_of_gravity_x_column_name = self._get_column_name_from_substring(all_columns = list(df.columns), substring = 'CenterOfGravity_x')
            session_overview['mean_CenterOfGravity_x'].append(df.loc[df['bout_id'].isin(bout_ids), center_of_gravity_x_column_name].mean())
        else:
            session_overview['total_bouts_count'].append(0)
            session_overview['total_duration'].append(0)
            session_overview['total_distance_covered'].append(0)
            session_overview['mean_duration'].append(np.nan)
            session_overview['mean_distance_covered'].append(np.nan)
            session_overview['mean_CenterOfGravity_x'].append(np.nan)            
        return session_overview
    
    
    def _create_parameter_settings_df(self) -> pd.DataFrame:
        logged_settings = {'parameter': [], 'specified_value': []}
        for parameter, value in self.logs.items():
            logged_settings['parameter'].append(parameter)
            logged_settings['specified_value'].append(value)
        return pd.DataFrame(data = logged_settings)
    
    
    def _write_xlsx_file_to_disk(self, dfs_to_export: Dict[str, pd.DataFrame]) -> None:
        writer = pd.ExcelWriter(f'{self.base_output_filepath}.xlsx', engine='xlsxwriter')
        for tab_name, df in dfs_to_export.items():
            df.to_excel(writer, sheet_name = tab_name)
        writer.save()


    def inspect_processing(self,
                           marker_ids_to_inspect: List[str]=['CenterOfGravity'],
                           verbose: bool=False,
                           show_plot: bool=False,
                           save_plot: bool=True,
                           show_legend: bool=False
                          ) -> None:
        if verbose:
            print(f'Inspection of file: {self.filepath.name}:')
            for marker_id in marker_ids_to_inspect:
                coverage = self._compute_coverage(df = self.bodyparts[marker_id].df, critical_marker_ids = [marker_id])
                print(f'... coverage of "{marker_id}" was at: {round(coverage*100, 2)} %')
        if show_plot or save_plot:
            self._plot_selected_marker_ids_on_normalized_maze(marker_ids = marker_ids_to_inspect,
                                                              show = show_plot, 
                                                              save = save_plot, 
                                                              legend = show_legend)

        
    def _plot_selected_marker_ids_on_normalized_maze(self, marker_ids: List[str], show: bool=True, save: bool=True, legend: bool=False) -> None:
        fig = plt.figure(figsize=(10, 5), facecolor='white')
        ax = fig.add_subplot(111)
        for corner_marker_id in ['MazeCornerClosedRight', 'MazeCornerClosedLeft', 'MazeCornerOpenRight', 'MazeCornerOpenLeft']:
            x, y = self.logs[f'normalized_{corner_marker_id}_coords']
            plt.scatter(x, y, label = corner_marker_id)
        for marker_id in marker_ids:
            plt.scatter(self.bodyparts[marker_id].df['x'], self.bodyparts[marker_id].df['y'], alpha = 0.1, label = marker_id)
        plt.plot([0, 0, 50, 50, 0], [0, 4, 4, 0, 0], c = 'black')
        ax.set_aspect('equal')
        if legend:
            plt.legend()
        if save:
            assert hasattr(self, 'base_output_filepath'), 'You must run ".export_results()" first if you´d like to save the plot. Alternatively, just opt to show the plot.' 
            plt.savefig(f'{self.base_output_filepath}.png', dpi = 300)
        if show:
            plt.show()
        else:
            plt.close()
                
                
                
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