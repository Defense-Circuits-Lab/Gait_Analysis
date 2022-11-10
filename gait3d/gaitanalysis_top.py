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
        
class RecordingTop(ABC):
    """
    Class for Analysing 2D-Position Data of mice in the OpeningTrack.
    
    Attributes:
        full_df_from_file(pandas.DataFrame): the Dataframe containing all bodyparts with x, y-coordinates and likelihood as returned by DLC
        recorded_framerate(int): fps of the recording
        metadata(Dict): dictionary containing information read from the filename, such as animal_id, recording_date and Opening Track paradigm
    """
    def __init__(self, filepath: Path, recorded_framerate: int)->None:
        """
        Constructor for the Recording2D class.
        
        This function calls functions to get the Dataframe from the csv, that is given as filepath argument and to read metadata from the filename.
        
        Parameters:
            filepath(pathlib.Path): the filepath to the h5 containing DLC data
            recorded_framerate(int): fps of the recording
        """
        self.filepath = filepath
        self.full_df_from_file = self._get_df_from_file(filepath = filepath)
        self.recorded_framerate = recorded_framerate
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
        
        Relying on file naming like this: relying on this file naming: 196_F7-27_220826_OTT_Bottom_synchronizedDLC_resnet152_OT_BottomCam_finalSep20shuffle1_550000filtered.h5
        
        Parameters:
            filepath(pathlib.Path): the path linked to the h5.file
        Returns:
            Dict: containing date of recording, animal_id and OT paradigm
        """
        filepath_slices = filepath.split('_')
        animal_line, animal_id, recording_date, paradigm, cam_id = filepath_slices[0], filepath_slices[1], filepath_slices[2], filepath_slices[3][0:3], 'Top'
        self._check_metadata(metadata = (animal_line, animal_id, recording_date, paradigm, cam_id))
        return {'recording_date': self.recording_date, 'animal': self.mouse_line + '_' + self.mouse_id, 'paradigm': self.paradigm, 'cam': self.cam_id}
    
    
    def _read_metadata(self, project_config_filepath: Path, recording_config_filepath: Path, video_filepath: Path)->None:
        with open(project_config_filepath, "r") as ymlfile:
            project_config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        
        with open(recording_config_filepath,'r') as ymlfile2:
            recording_config = yaml.load(ymlfile2, Loader=yaml.SafeLoader)

        for key in ['target_fps', 'valid_cam_IDs', 'paradigms', 'animal_lines', 'intrinsic_calibration_dir', 'led_extraction_type', 'led_extraction_path']:
            try:
                project_config[key]
            except KeyError:
                raise KeyError(f'Missing metadata information in the project_config_file {project_config_filepath} for {key}.')
            
                
        self.target_fps = project_config['target_fps']
        self.valid_cam_ids = project_config['valid_cam_IDs']
        self.valid_paradigms = project_config['paradigms']
        self.valid_mouse_lines = project_config['animal_lines']
        self.intrinsic_calibrations_directory = Path(project_config['intrinsic_calibration_dir'])
        
        self._extract_filepath_metadata(filepath_name = video_filepath.name)
            
        for key in ['led_pattern', self.cam_id]:
            try:
                recording_config[key]
            except KeyError:
                raise KeyError(f'Missing information for {key} in the config_file {recording_config_filepath}!')
        
        self.led_pattern = recording_config['led_pattern']
        if self.recording_date != recording_config['recording_date']:
            raise ValueError (f'The date of the recording_config_file {recording_config_filepath} and the provided video {self.video_filepath} do not match! Did you pass the right config-file and check the filename carefully?')
        metadata_dict = recording_config[self.cam_id]
    
        for key in ['fps', 'offset_row_idx', 'offset_col_idx', 'flip_h', 'flip_v', 'fisheye']:
            try:
                metadata_dict[key]
            except KeyError:
                raise KeyError(f'Missing metadata information in the recording_config_file {recording_config_filepath} for {self.cam_id} for {key}.')  
                
        self.fps = metadata_dict['fps']
        self.offset_row_idx = metadata_dict['offset_row_idx']
        self.offset_col_idx = metadata_dict['offset_col_idx']
        self.flip_h = metadata_dict['flip_h']
        self.flip_v = metadata_dict['flip_v']
        self.fisheye = metadata_dict['fisheye']

        self.processing_type = project_config['processing_type'][self.cam_id]
        self.calibration_evaluation_type = project_config['calibration_evaluation_type'][self.cam_id]
        self.processing_path = Path(project_config['processing_path'][self.cam_id])
        self.calibration_evaluation_path = Path(project_config['calibration_evaluation_path'][self.cam_id])
        self.led_extraction_type = project_config['led_extraction_type'][self.cam_id]
        self.led_extraction_path = project_config['led_extraction_path'][self.cam_id]
       
    @property
    def valid_paradigms(self)->List[str]:
        return ['OTR', 'OTT', 'OTE']
    
    @property
    def valid_mouse_lines(self)->List[str]:
        return ['194', '195', '196', '206', '209']
        
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
        
    def run(self, intrinsic_camera_calibration_filepath: Path)->None:
        """
        Function to create Bodypart2D objects for all the tracked markers.
        
        A function is called, that first calculates the centerofgravity and then creates Bodypart objects for all the markers.
        The points are first undistorted, based on the intrinsic camera calibration, which is adjusted based on the cropping. Afterwards, the coordinate system is normalized via translation, rotation and conversion to unit cms.
        Basic parameters for the bodyparts are already calculated, such as, speed and immobility.
        Currently no frames are excluded.
        
        Parameters: #the latter two could be read from the .config file
            intrinsic_camera_calibration_filepath(Path): pickle file containing intrinsic camera parameters
            xy_offset(Tuple): cropping offsets of the recorded video
            video_filepath: path to the recorded video
        """
        self.log = {}
        self._calculate_center_of_gravity()
        K, D = self._load_intrinsic_camera_calibration(intrinsic_camera_calibration_filepath = intrinsic_camera_calibration_filepath)
        size = (640, 480)
        self.camera_parameters_for_undistortion = {'K': K, 'D': D, 'size': size}
        self._create_all_bodyparts()
        self._normalize_coordinate_system()
        self._run_basic_operations_on_bodyparts()
        self._get_tracking_performance()

    def _calculate_center_of_gravity(self)->None:
        """
        Function, that calculates the centerofgravity.
        
        The center_of_gravity is calculated using the bodyparts Snout and TailBase. The likelihood is calculated as the multiplied likelihood of Snout and TailBase.
        It adds centerofgravity to self.full_df_from_file.
        """
        self._calculate_new_bodypart(['Snout', 'TailBase'], 'centerofgravity')
        
    def _calculate_new_bodypart(self, bodyparts: List[str], label: str)->None:
        for coordinate in ['x', 'y']:
            self.full_df_from_file[f'{label}_{coordinate}'] = (sum([self.full_df_from_file[f'{bp}_{coordinate}'] for bp in bodyparts]))/len(bodyparts)
        self.full_df_from_file[f'{label}_likelihood'] = np.prod([self.full_df_from_file[f'{bp}_likelihood'] for bp in bodyparts], axis = 0) 
        
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
                self.bodyparts[bodypart] = Bodypart2D(bodypart_id = bodypart, df = self.full_df_from_file, camera_parameters_for_undistortion=self.camera_parameters_for_undistortion)
                
    def _normalize_coordinate_system(self)->None:
        """
        This Function normalizes the coordinate system.
        
        The mazecorners of the best frame are used to calculate necessary parameters for the following functions, 
        such as the conversion factor from the intrinsic unit to cms, 
        the translation vector from the real-world-coordinate system to the null-space,
        and the angle between x-axis and X-axis for 2D rotation.
        With this parameters it calls the normalization for each single bodypart.
        """
        self.sanity = False
        likelihood_threshold = 0.99
        coverage_threshold = 0.95
        self.log['crashed'] = False
        while self.sanity == False:
            mazecorners = self._fix_coordinates_of_maze_corners(likelihood_threshold = likelihood_threshold)
            conversion_factor = self._get_conversion_factor_px_to_cm(reference_points = mazecorners)
            translation_vector = self._get_translation_vector(reference_points = mazecorners)
            rotation_angle = self._get_rotation_angle_alt(reference_points = mazecorners)
            angle = self._get_rotation_angle(reference_points = mazecorners)
            
            for bodypart in self.bodyparts.values():
                bodypart.normalize_df(translation_vector = translation_vector, rotation_angle = rotation_angle, conversion_factor = conversion_factor)
            if self._check_sanity(coverage_threshold = coverage_threshold):
                break
            likelihood_threshold += 0.0005
            if likelihood_threshold > 1:
                mazecorners = self._fix_coordinates_of_maze_corners(likelihood_threshold = 0.9999)
                conversion_factor = self._get_conversion_factor_px_to_cm_alt(reference_points = mazecorners)
                rotation_angle = self._get_rotation_angle_alt(reference_points = mazecorners)

                for bodypart in self.bodyparts.values():
                    bodypart.normalize_df(translation_vector = translation_vector, rotation_angle = rotation_angle, conversion_factor = conversion_factor)
                self._check_sanity(coverage_threshold = coverage_threshold)
                if self.sanity == False:
                    self.log['crashed'] = True
                    raise OverflowError ('The .csv could not be normalized!')
        self.log['likelihood_threshold'] = likelihood_threshold
        self.log['rotation_angle'] = rotation_angle
        self.log['conversion_factor'] = conversion_factor
        self.log['plotting_marker'] = self._add_plotting_marker_to_log(likelihood_threshold = likelihood_threshold)
        self.log['number_frames'] = self.bodyparts['Snout'].df.shape[0]
        
    def _add_plotting_marker_to_log(self, likelihood_threshold: float)->None:
        mazecorneropenright = self._get_most_reliable_marker_position(df = self.bodyparts['MazeCornerOpenRight'].df, likelihood_threshold = likelihood_threshold)
        mazecornerclosedright = self._get_most_reliable_marker_position(df = self.bodyparts['MazeCornerClosedRight'].df, likelihood_threshold = likelihood_threshold)
        mazecorneropenleft = self._get_most_reliable_marker_position(df = self.bodyparts['MazeCornerOpenLeft'].df, likelihood_threshold = likelihood_threshold)
        mazecornerclosedleft = self._get_most_reliable_marker_position(df = self.bodyparts['MazeCornerClosedLeft'].df, likelihood_threshold = likelihood_threshold)
        return [mazecornerclosedleft, mazecornerclosedright, mazecorneropenright, mazecorneropenleft]
            
    def _get_conversion_factor_px_to_cm_alt(self, reference_points: Tuple[np.array, np.array])->None:
        """
        Function to get the conversion factor of the unspecified unit to cm.
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
            
        Returns:
            conversion_factor(float): factor to convert the unspecified unit into cm.
        """
        conversion_factor = 4/np.sqrt(sum((reference_points[2]-reference_points[0])**2))
        return conversion_factor
    
    def _get_rotation_angle_alt(self, reference_points: Tuple[np.array, np.array])->None:
        """
        Function, that calculates the angle between the x-axis and the X-axis, rotated around the z-axis.
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
            
        Returns:
            float: angle in radians
        """
        closed_left_translated = reference_points[2]-reference_points[0]
        a, b = closed_left_translated[0], closed_left_translated[1]
        closed_left_translated = np.array([a, b, 0])
        open_side = np.cross(closed_left_translated, np.array([0, 0, 1]))[0:2]
        angle = np.dot(np.array([50, 0]), open_side)/(np.linalg.norm(np.array([50, 0]))*np.linalg.norm(open_side))
        return math.acos(angle)
            
    def _check_sanity(self, coverage_threshold: float=0.95)->None:
        a, snout = self._bodypart_on_maze(bodypart = 'Snout', coverage_threshold = coverage_threshold)
        b, tailbase = self._bodypart_on_maze(bodypart = 'TailBase', coverage_threshold = coverage_threshold)
        if all([a, b]):
            self.log['snout_on_maze'] = snout
            self.log['tailbase_on_maze'] = tailbase
            self.sanity = True
            return True
        else:
            return False

    def _bodypart_on_maze(self, bodypart: str, coverage_threshold: float=0.9)->bool:
        x = (self.bodyparts[bodypart].df.loc[(self.bodyparts[bodypart].df['x'] > -5) & (self.bodyparts[bodypart].df['x'] < 55), :].shape[0]/self.bodyparts[bodypart].df.shape[0])
        y = (self.bodyparts[bodypart].df.loc[(self.bodyparts[bodypart].df['y'] > -1) & (self.bodyparts[bodypart].df['y'] < 6), :].shape[0]/self.bodyparts[bodypart].df.shape[0])
        if all([x > coverage_threshold, y > coverage_threshold]):
            return True, (x+y)/2
        else:
            return False, 0
        
    def _get_conversion_factor_px_to_cm(self, reference_points: Tuple[np.array, np.array])->float:
        """
        Function to get the conversion factor of the unspecified unit to cm.
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
            
        Returns:
            conversion_factor(float): factor to convert the unspecified unit into cm.
        """
        conversion_factor = 50/np.sqrt(sum((reference_points[1]-reference_points[0])**2))
        return conversion_factor
    
    def _get_translation_vector(self, reference_points:  Tuple[np.array, np.array])->float:
        """
        Function that calculates the offset of the right closed mazecorner to (0, 0).
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
            
        Returns:
            translation_vector(np.array): vector with offset in each dimension
        """
        translation_vector = -reference_points[0]
        return translation_vector
    
    def _get_rotation_angle(self, reference_points: Tuple[np.array, np.array])->float:
        """
        Function, that calculates the angle between the x-axis and the X-axis, rotated around the z-axis.
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
            
        Returns:
            float: angle in radians
        """
        closed_right_translated, open_right_translated = np.array([0, 0]), (reference_points[1]-reference_points[0])
        
        length_b = math.sqrt(open_right_translated[0]**2 + open_right_translated[1]**2)
        length_c = 50
        length_a = math.sqrt((open_right_translated[0]-50)**2 + (open_right_translated[1])**2)
        angle = (length_b**2 + length_c**2 - length_a**2) / (2 * length_b * length_c)  
        return math.acos(angle)
    
    def _load_intrinsic_camera_calibration(self, intrinsic_camera_calibration_filepath: Path) -> Tuple[np.array, np.array]:
        """
        This function opens the camera calibration from the pickle file and adjusts it based on the cropping parameters.
        
        Parameters:
            intrinsic_camera_calibration_filepath(Path): pickle file containing intrinsic camera parameters
            x_offset(int): cropping offset x of the recorded video
            y_offset(int): cropping offset y of the recorded video
        Returns:
            Tuple: the camera matrix K and the distortion coefficient D as np.array
        """
        with open(intrinsic_camera_calibration_filepath, 'rb') as io:
            intrinsic_calibration = pickle.load(io)
        return intrinsic_calibration['K'], intrinsic_calibration['D']

    def _fix_coordinates_of_maze_corners(self, likelihood_threshold: float)->Tuple[np.array, np.array, np.array]:
        """
        Function that creates the reference_points from the mazecorners.
        
        After finding the best matching frame it combines the -x, -y, coordinate of this frame for all corners into a np.array.
        
        Returns:
            reference_points(Tuple): the best tracked maze corners as numpy.Array
        """        
        mazecorneropenright = self._get_most_reliable_marker_position(df = self.bodyparts['MazeCornerOpenRight'].df_undistort, likelihood_threshold = likelihood_threshold)
        mazecornerclosedright = self._get_most_reliable_marker_position(df = self.bodyparts['MazeCornerClosedRight'].df_undistort, likelihood_threshold = likelihood_threshold)
        mazecorneropenleft = self._get_most_reliable_marker_position(df = self.bodyparts['MazeCornerOpenLeft'].df_undistort, likelihood_threshold = likelihood_threshold)
        mazecornerclosedleft = self._get_most_reliable_marker_position(df = self.bodyparts['MazeCornerClosedLeft'].df_undistort, likelihood_threshold = likelihood_threshold)
        return mazecornerclosedleft, mazecorneropenleft, mazecornerclosedright
        
    def _get_most_reliable_marker_position(self, df: pd.DataFrame, likelihood_threshold: float=0.99) -> np.array:
        while True:
            df_reliable_frames = df.loc[df['likelihood'] >= likelihood_threshold, ['x', 'y']].copy()
            if df_reliable_frames.shape[0] > 0:
                break
            likelihood_threshold -= 0.0005
        return np.array([df_reliable_frames['x'].median(), df_reliable_frames['y'].median()])
        
    def _run_basic_operations_on_bodyparts(self)->None:
        """ Basic parameters for the bodyparts are calculated, such as, speed and immobility. """
        for bodypart in self.bodyparts.values():
            bodypart.run_basic_operations(recorded_framerate = self.recorded_framerate)
            
    def _get_tracking_performance(self)->None:
        """
        Function, that calculates the percentage of frames, in which a marker was detected.
        
        It sets self.tracking_performance as pandas.DataFrame with columns for all markers and the percentage value over the whole session as first row.
        """
        tracking_dict = {bodypart.id: bodypart.check_tracking_stability() for bodypart in self.bodyparts.values()}
        #calculate standard derivation for fixed markers
        #for bodypart in set(['MazeCornerOpenLeft', 'MazeCornerOpenRight', 'MazeCornerClosedLeft', 'MazeCornerClosedRight', 'LED5']): 
            #tracking_dict['standard_derivation']=np.stddev([])
        self.tracking_performance = pd.DataFrame([tracking_dict], index=['over_total_session', 'over_gait_events'])
        self.log['snout_tracking'] = self.tracking_performance.loc['over_total_session', 'Snout']
        self.log['tailbase_tracking'] = self.tracking_performance.loc['over_total_session', 'TailBase']
    
    def get_freezing_bouts(self)->None:
        """
        Function for the detection of freezing bouts.
        
        After calculation of important parameters such as direction, turns and immobility of the most relevant bodyparts,
        the freezing bouts are collected.
        """
        self._get_direction()
        self._get_turns()
        self._check_immobility_of_all_freezing_bodyparts()        
        # create class for each parameter and abstract parent class?
        self._get_immobility_bouts()
        # think of better definition of freezing/immobility
        self._run_operations_on_immobility_bouts()
        self._collect_freezing_bouts()
        
    def _get_direction(self)->None:
        """
        Checks frame by frame, whether the Snout is closer to the open end of the maze than the Ears.
        
        This is set as a parameter (in the future: Parameter object) self.facing_towards_open_end with Boolean values.
        """
        self.facing_towards_open_end = self._initialize_new_parameter(dtype=bool)
        self.facing_towards_open_end.loc[(self.bodyparts['Snout'].df.loc[:, 'x']>self.bodyparts['TailBase'].df.loc[:, 'x'])] = True
        
    def _get_turns(self)->None:
        """
        Function that checks for turning events.
        
        Based on self.facing_towards_open_end, the indices of events, where a mouse turns are extracted and the attributes
        self.turns_to_closed and self.turns_to_open are created as a list of EventBout2Ds.
        """
        turn_indices = self.facing_towards_open_end.where(self.facing_towards_open_end.diff()==True).dropna().index
        self.turns_to_closed=[EventBout2D(start_index = start_index) for start_index in turn_indices if self.facing_towards_open_end[start_index-1]==True]
        self.turns_to_open=[EventBout2D(start_index = start_index) for start_index in turn_indices if self.facing_towards_open_end[start_index-1]==False]
        for turning_bout in self.turns_to_closed:
            turning_bout.get_position(centerofgravity=self.bodyparts["centerofgravity"])
        for turning_bout in self.turns_to_open:
            turning_bout.get_position(centerofgravity=self.bodyparts["centerofgravity"])

    def _initialize_new_parameter(self, dtype: type)->pd.Series:
        """   
        Creates a Series object for initializing a parameter.
        
        This Function will be replaced by the object Parameter in the future.
        
        Parameters:
            dtype(type): the default type of the new created parameter.
        Returns:
            pd.Series of an array in shape of n_frames with default values set to 0 (if dtype bool->False)
        """
        return pd.Series(np.zeros_like(np.arange(self.full_df_from_file.shape[0]), dtype = dtype))
                    
    def _check_immobility_of_all_freezing_bodyparts(self)->None:    
        """
        Function, that checks frame by frame, whether the relevant bodyparts for freezing are immobile.
        
        The information is stored as attribute self.all_freezing_bodyparts_immobile.
        """
        self.all_freezing_bodyparts_immobile = self._initialize_new_parameter(dtype=bool)
        self.all_freezing_bodyparts_immobile.loc[(self.bodyparts['Snout'].df.loc[:, 'immobility']) & (self.bodyparts['TailBase'].df.loc[:, 'immobility'])] = True
        
    def _get_immobility_bouts(self)->None:
        """
        Function, that creates immobility EventBouts.
        
        The Function detects start and end of an immobility episode and creates EventBouts for every episode.
        Sets a List of EventBouts as attribute self.immobility_bouts.
        """
        changes_from_immobility_to_mobility = self.all_freezing_bodyparts_immobile.where(self.all_freezing_bodyparts_immobile.diff()==True).dropna()
        start_indices_of_immobility_bouts = changes_from_immobility_to_mobility[::2]
        end_indices_of_immobility_bouts = changes_from_immobility_to_mobility[1::2]
        immobility_bouts = []
        for i in range(len(start_indices_of_immobility_bouts)-1):
            start_index, end_index = start_indices_of_immobility_bouts.index[i], end_indices_of_immobility_bouts.index[i]
            immobility_bout = EventBout2D(start_index = start_index, end_index = end_index, recorded_framerate = self.recorded_framerate)
            if immobility_bout.duration > 0.2:
                immobility_bouts.append(immobility_bout)
        self.immobility_bouts = EventSeries(range_end = self.bodyparts['Snout'].df.shape[0], events = immobility_bouts, event_type = 'immobility', recorded_framerate = self.recorded_framerate, range_start = 0)
              
    def _run_operations_on_immobility_bouts(self)->None:
        """
        Basic operations are run on the Immobility Bouts.
        
        A pandas.DataFrame for immobility bouts is created and set as self.immobility_bout_df.
        """
        self.immobility_bouts.run_basic_operations_on_events(facing_towards_open_end = self.facing_towards_open_end, centerofgravity = self.bodyparts['centerofgravity'])
        self.immobility_bout_df = pd.DataFrame([immobility_bout.dict for immobility_bout in self.immobility_bouts.events])
            
    def _collect_freezing_bouts(self)->None:
        """
        The Immobility bouts, where the freezing threshold was exceeded are collected as freezing bouts.
        
        A pandas.DataFrame for freezing bouts is created and set as self.freezing_bout_df.
        """
        freezing_bouts = []
        for immobility_bout in self.immobility_bouts.events:
            if immobility_bout.freezing_threshold_reached:
                freezing_bouts.append(immobility_bout)
        self.freezing_bouts = EventSeries(range_end = self.bodyparts['Snout'].df.shape[0], events = freezing_bouts, event_type = 'freezing', recorded_framerate = self.recorded_framerate, range_start = 0)
        self.freezing_bouts.run_basic_operations_on_events(facing_towards_open_end = self.facing_towards_open_end, centerofgravity = self.bodyparts['centerofgravity'])
        self.freezing_bout_df = pd.DataFrame([freezing_bout.dict for freezing_bout in self.freezing_bouts.events])

    def run_gait_analysis(self)->None:
        """
        Function, that runs functions, necessary for gait analysis.
        
        Angles between bodyparts of interest are calculated as Angle objects.
        A peak detection algorithm on paw_speed is used to detect steps.
        EventBouts are created.
        """
        self.gait_events = self._define_gait_events()
        self.gait_events.run_basic_operations_on_events(facing_towards_open_end = self.facing_towards_open_end, centerofgravity = self.bodyparts['centerofgravity'])
        self._get_gait_event_bouts(gait_events = self.gait_events)
    
    def _define_gait_events(self)->List:
        locomotion_threshold_reached = self.bodyparts['centerofgravity'].df.loc[self.bodyparts['centerofgravity'].df['rolling_speed_cm_per_s']>10, :].index.values
        events = [EventBout2D(start_index = index) for index in locomotion_threshold_reached]
        return EventSeries(range_end = self.bodyparts['centerofgravity'].df.shape[0], events = events, event_type = 'gait', recorded_framerate = self.recorded_framerate)
    
    def _get_gait_event_bouts(self, gait_events: List)->None:
        """
        Function that collects eventbouts of a specific type after a gait_event together.
        
        It sets the attribute self.turns_to_closed_after_gait for turns to the closed side after a gait event, 
        self.turns_to_open_after_gait for turns to the open side after a gait event,
        self.gait_disruption_bouts for immobility bouts after a gait event and
        self.freezing_of_gait_events for freezing bouts after a gait event.
        The lists are cleared from None values afterwards.
        
        Parameters:
            gait_events(List): nested list with sublists that represent single gait events containing single step indices
        """
        self.turns_to_closed_after_gait = [turn for sublist in [self._bout_after_gait_event(gait_event = gait_event, event = self.turns_to_closed) for gait_event in gait_events.events] for turn in sublist if type(turn)==EventBout2D]
        self.turns_to_open_after_gait = [turn for sublist in [self._bout_after_gait_event(gait_event = gait_event, event = self.turns_to_open) for gait_event in gait_events.events] for turn in sublist if type(turn)==EventBout2D]
        gait_disruption_bouts = [disruption for sublist in [self._bout_after_gait_event(gait_event = gait_event, event = self.immobility_bouts.events) for gait_event in gait_events.events] for disruption in sublist if type(disruption)==EventBout2D]
        self.gait_disruption_bouts = EventSeries(range_end = self.bodyparts['Snout'].df.shape[0], events = gait_disruption_bouts, event_type = 'gait_disruption', recorded_framerate = self.recorded_framerate, range_start = 0)
        self.gait_disruption_bouts.run_basic_operations_on_events(facing_towards_open_end = self.facing_towards_open_end, centerofgravity = self.bodyparts['centerofgravity'])
        freezing_of_gait_events = [freezing for sublist in [self._bout_after_gait_event(gait_event = gait_event, event = self.freezing_bouts.events) for gait_event in gait_events.events] for freezing in sublist if type(freezing)==EventBout2D]
        self.freezing_of_gait_events = EventSeries(range_end = self.bodyparts['Snout'].df.shape[0], events = freezing_of_gait_events, event_type = 'freezing_of_gait', recorded_framerate = self.recorded_framerate, range_start = 0)
        self.freezing_of_gait_events.run_basic_operations_on_events(facing_towards_open_end = self.facing_towards_open_end, centerofgravity = self.bodyparts['centerofgravity'])
            
    def _bout_after_gait_event(self, gait_event: List, event: List)->List:
        """
        Function that checks the second after a gait_event for an specified event.
        
        Parameters:
            frame_index(int): last index of a gait_event
            event(List): List with Event Bouts to check for
        Returns:
            Union: the Eventbout is returned if found, otherwise None
        """
        bouts = []
        for bout in event:
            if bout.start_index in range(gait_event.start_index, gait_event.end_index + 1*self.recorded_framerate):
                bouts.append(bout)
        return bouts
    
    def export_as_session(self, save: bool=False)->None:
        self.log['missing_event_types'] = []
        self.session_df = pd.DataFrame(columns = ['mean_value', 'total_count', 'total_duration', 'mean_x_position', 'mean_duration', 'total_count_facing_open', 'mean_duration_facing_open', 'mean_x_position_facing_open'])
        avg_speed = self.bodyparts['centerofgravity'].df['speed_cm_per_s'].mean()
        self.session_df.loc['average_speed_cm_per_s', 'mean_value'] = avg_speed
        for event_type in [self.immobility_bouts, self.gait_disruption_bouts, self.freezing_bouts, self.freezing_of_gait_events, self.gait_events]:
            try:
                event_type.calculate_statistics()
                total_count = event_type.total_count
                mean_duration = event_type.mean_duration
                mean_x_position = event_type.mean_x_position
                total_duration = event_type.total_duration
                if event_type.event_type != 'Gait':
                    total_count_facing_open = event_type.total_count_facing_open
                    mean_duration_facing_open = event_type.mean_duration_facing_open
                    mean_x_position_facing_open = event_type.mean_x_position_facing_open
                    self.session_df.loc[event_type.event_type, ['total_count', 'total_duration','mean_x_position', 'mean_duration', 'total_count_facing_open', 'mean_duration_facing_open', 'mean_x_position_facing_open']] = total_count, total_duration, mean_x_position, mean_duration, total_count_facing_open, mean_duration_facing_open, mean_x_position_facing_open
                else:
                    self.session_df.loc[event_type.event_type, ['total_count', 'total_duration','mean_x_position', 'mean_duration']] = total_count, total_duration, mean_x_position, mean_duration
            except KeyError:
                self.log['missing_event_types'].append(event_type.event_type)
        if save:
            self.session_df.to_csv(f'{self.metadata["animal"]}_{self.metadata["recording_date"]}_{self.metadata["paradigm"]}.csv')
        self._create_report()
        
    def _create_report(self)->None:
        reporting = False
        print(f'\n\nReport for {self.metadata["animal"]}_{self.metadata["recording_date"]}_{self.metadata["paradigm"]}: ')
        if len (self.log['missing_event_types']) > 0:
            reporting = True
            print(f'\nMissing events: {self.log["missing_event_types"]}')
        if self.log['snout_tracking'] < 0.9:
            reporting = True
            print(f'\nSnout tracking: {self.log["snout_tracking"]}')
        if self.log['tailbase_tracking'] < 0.9:
            reporting = True
            print(f'\nTailBase tracking: {self.log["tailbase_tracking"]}')
        if self.log['snout_on_maze'] < 0.95 or self.log['tailbase_on_maze'] < 0.95:
            reporting = True
            print(f'\nRotation or normalization didnt work out well. \nUsed {self.log["rotation_angle"]} and {self.log["conversion_factor"]}.\nCheck out the distribution plot!')
        if self.log['number_frames'] < 60*self.recorded_framerate:
            reporting = True
            print(f'\nDetected only {self.log["number_frames"]} frames! \nThats less than one minute in framerate {self.recorded_framerate}!')
        if not reporting:
            print('\nNothing to report!')

            
                
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
        
    def run_basic_operations(self, recorded_framerate: int)->None:
        """
        Function that calculates Speed and Immobility.
        """
        self._get_speed(recorded_framerate = recorded_framerate)
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
    
    def _get_speed(self, recorded_framerate: int)->None:
        """
        Function, that calculates the speed of the bodypart, based on the framerate.
        
        After creating an empty column with np.NaN values, the speed is calculated 
        as the squareroot of the squared difference between two frames in -x and -y dimension divided by the duration of a frame.
        
        Parameters:
            recorded_framerate(int): fps of the recording
        """
        self.df.loc[:, 'speed_cm_per_s'] = np.NaN
        self.df.loc[:, 'speed_cm_per_s'] = (np.sqrt(self.df.loc[:, 'x'].diff()**2 + self.df.loc[:, 'y'].diff()**2)) / (1/recorded_framerate)        
    
    
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
    def __init__(self, start_index: int, end_index: Optional[int]=None, recorded_framerate: Optional[int]=None)->None:
        """
        Constructor of class EventBout that sets the attributes start_ and end_index.
        
        Parameters: 
            start_index(int): index of event onset
            end_index(Optional[int]): index of event ending (if event is not only a single frame)
        """
        self.start_index = start_index
        if end_index != None:
            self.end_index = end_index
            self.duration = (self.end_index - self.start_index)/recorded_framerate
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
            recorded_framerate(int): fps of the recording
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
    
    
    def __init__(self, range_end: int, events: List, event_type: str, recorded_framerate: int, range_start: int = 0):
        self.events = self._merge_events(events = events, recorded_framerate = recorded_framerate)
        self.event_type = event_type
        
    def _merge_events(self, events: List, recorded_framerate:int)->List:
        events_to_keep = []  
        for i in range(len(events)-1):
            try:
                events[i+1]
            except IndexError:
                break
            if ((events[i+1].start_index - events[i].end_index)/recorded_framerate) < self.merge_threshold:
                j = i + 1
                try: 
                    events[j+1]
                    while ((events[j].start_index - events[i].end_index)/recorded_framerate) < self.merge_threshold:
                        j += 1
                except IndexError: 
                    j -= 1
                events_to_keep.append(EventBout2D(start_index = events[i].start_index, end_index = events[j].end_index, recorded_framerate=recorded_framerate))
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