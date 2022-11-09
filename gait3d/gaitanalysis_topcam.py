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

        

class Recording2D(ABC):
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
            df = pd.read_csv(filepath)
            df = df.drop('scorer', axis=1)
            df.columns = df.iloc[0, :]+ '_' + df.iloc[1, :]
            df = df.drop([0, 1], axis=0)
            df = df.reset_index()
            df = df.drop('index', axis=1)
            df = df.astype(float)
        elif filepath.name.endswith('.h5'):
            df = pd.read_hdf(filepath)
            df = df.drop('scorer', axis=1)
            df.columns = df.iloc[0, :]+ '_' + df.iloc[1, :]
            df = df.drop([0, 1], axis=0)
            df = df.reset_index()
            df = df.drop('index', axis=1)
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
        animal_line, animal_id, recording_date, paradigm, cam_id = filepath_slices[0], filepath_slices[1], filepath_slices[2], filepath_slices[3], filepath_slices[4]
        return {'recording_date': recording_date, 'animal': animal_line + '_' + animal_id, 'paradigm': paradigm, 'cam': cam_id}
        
        
    def run(self, intrinsic_camera_calibration_filepath: Path, xy_offset: Tuple[int, int], video_filepath: Path)->None:
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
        self._calculate_center_of_gravity()
        self._create_freezing_body_regions()
        K, D = self._load_intrinsic_camera_calibration(intrinsic_camera_calibration_filepath = intrinsic_camera_calibration_filepath, x_offset=xy_offset[0], y_offset=xy_offset[1])
        image = iio.imread(video_filepath, index = 0)
        size = image.shape[1], image.shape[0]
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
            
    def _create_freezing_body_regions(self)->None:
        self._calculate_new_bodypart(['Snout', 'ForePawRight', 'ForePawLeft'], 'Front')
        self._calculate_new_bodypart(['TailBase', 'HindPawRight', 'HindPawLeft'], 'Back')
        
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
        mazecorners = self._fix_coordinates_of_maze_corners()
        conversion_factor = self._get_conversion_factor_px_to_cm(reference_points = mazecorners)
        translation_vector = self._get_translation_vector(reference_points = mazecorners)
        rotation_angle = self._get_rotation_angle(reference_points = mazecorners)
        
        for bodypart in self.bodyparts.values():
            bodypart.normalize_df(translation_vector = translation_vector, rotation_angle = rotation_angle, conversion_factor = conversion_factor)
    
    def _find_best_mazecorners_for_normalization(self)->int:
        """
        Function to find the frame, where the mazecorners are tracked best.
        
        It calculates the frame with the maximal mean likelihood of the four mazecorners.
        
        Returns:
            best_matching_frame(int): the index of the frame
        """
        frame_likelihood = [(self.bodyparts['MazeCornerOpenRight'].df_raw['likelihood'][frame] + 
                            self.bodyparts['MazeCornerOpenLeft'].df_raw['likelihood'][frame]+
                            self.bodyparts['MazeCornerClosedRight'].df_raw['likelihood'][frame]+
                            self.bodyparts['MazeCornerClosedLeft'].df_raw['likelihood'][frame])/4
                        for frame in range(self.full_df_from_file.shape[0])]
        best_matching_frame = frame_likelihood.index(max(frame_likelihood))
        return best_matching_frame
    
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
    
    def _load_intrinsic_camera_calibration(self, intrinsic_camera_calibration_filepath: Path, x_offset: int, y_offset: int) -> Tuple[np.array, np.array]:
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
        adjusted_K = intrinsic_calibration['K'].copy()
        adjusted_K[0][2] = adjusted_K[0][2] - x_offset
        adjusted_K[1][2] = adjusted_K[1][2] - y_offset
        return adjusted_K, intrinsic_calibration['D']

    def _fix_coordinates_of_maze_corners(self)->Tuple[np.array, np.array, np.array]:
        """
        Function that creates the reference_points from the mazecorners.
        
        After finding the best matching frame it combines the -x, -y, coordinate of this frame for all corners into a np.array.
        
        Returns:
            reference_points(Tuple): the best tracked maze corners as numpy.Array
        """
        frame = self._find_best_mazecorners_for_normalization()
        
        mazecorneropenright = np.array([self.bodyparts['MazeCornerOpenRight'].df_undistort.loc[frame, 'x'], self.bodyparts['MazeCornerOpenRight'].df_undistort.loc[frame, 'y']])
        mazecornerclosedright = np.array([self.bodyparts['MazeCornerClosedRight'].df_undistort.loc[frame, 'x'], self.bodyparts['MazeCornerClosedRight'].df_undistort.loc[frame, 'y']])
        mazecorneropenleft = np.array([self.bodyparts['MazeCornerOpenLeft'].df_undistort.loc[frame, 'x'], self.bodyparts['MazeCornerOpenLeft'].df_undistort.loc[frame, 'y']])
        mazecornerclosedleft = np.array([self.bodyparts['MazeCornerClosedLeft'].df_undistort.loc[frame, 'x'], self.bodyparts['MazeCornerClosedLeft'].df_undistort.loc[frame, 'y']])
        return mazecornerclosedright, mazecorneropenright, mazecorneropenleft
                
        
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
        self.facing_towards_open_end.loc[(self.bodyparts['Front'].df.loc[:, 'x']>self.bodyparts['Back'].df.loc[:, 'x']) &
                                    (self.bodyparts['Front'].df.loc[:, 'x']>self.bodyparts['Back'].df.loc[:, 'x'])] = True
        
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
        self.all_freezing_bodyparts_immobile.loc[(self.bodyparts['Front'].df.loc[:, 'immobility']) & (self.bodyparts['Back'].df.loc[:, 'immobility'])] = True
        
        
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
        for i in range(len(start_indices_of_immobility_bouts)):
            start_index, end_index = start_indices_of_immobility_bouts.index[i], end_indices_of_immobility_bouts.index[i]
            immobility_bout = EventBout2D(start_index = start_index, end_index = end_index, recorded_framerate = self.recorded_framerate)
            if immobility_bout.duration > 0.2:
                immobility_bouts.append(immobility_bout)
        self.immobility_bouts = EventSeries(range_end = self.bodyparts['Snout'].df.shape[0], events = immobility_bouts, recorded_framerate = self.recorded_framerate, range_start = 0)
            
            
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
        self.freezing_bouts = EventSeries(range_end = self.bodyparts['Snout'].df.shape[0], events = freezing_bouts, recorded_framerate = self.recorded_framerate, range_start = 0)
        self.freezing_bouts.run_basic_operations_on_events(facing_towards_open_end = self.facing_towards_open_end, centerofgravity = self.bodyparts['centerofgravity'])
        self.freezing_bout_df = pd.DataFrame([freezing_bout.dict for freezing_bout in self.freezing_bouts.events])


    def run_gait_analysis(self)->None:
        """
        Function, that runs functions, necessary for gait analysis.
        
        Angles between bodyparts of interest are calculated as Angle objects.
        A peak detection algorithm on paw_speed is used to detect steps.
        EventBouts are created.
        """
        self._calculate_angles()
        steps = self._detect_steps()
        gait_periodes = self._define_gait_periodes(steps=steps)
        self._get_gait_event_bouts(gait_events=gait_periodes)
        self._add_angles_to_steps(gait_events=gait_periodes)
        self._get_tracking_stability_in_gait_periodes(gait_periodes=gait_periodes)
        self._calculate_parameters_for_gait_analysis()
        self._create_PSTHs()
        
        
    def _detect_steps(self)->List:
        """
        Function that runs step detection in the individual paw Bodypart objects and sorts the steps by start_index.
        
        Returns:
            steps(List): Step objects for speed peaks in all paws
        """
        steps_per_paw = [self.bodyparts[paw]._detect_steps() for paw in['HindPawRight', 'HindPawLeft', 'ForePawRight', 'ForePawLeft']]
        steps = [item for sublist in steps_per_paw for item in sublist]
        steps_sorted = sorted(steps, key=lambda x: getattr(x, 'start_index'))
        return steps_sorted
    
    def _define_gait_periodes(self, steps: List)->List:
        """
        Function, that recognizes periodes of gait.
        
        If more than x steps are done in less than y seconds, those steps are considered a gait_event.

        Parameters:
            steps(List): list of steps with Step objects
        Returns:
            gait_events(List): nested list with sublists that represent single gait events containing single step indices
        """
        
        x = 3
        y = 3
        
        gait_events = []
        for i in range(len(steps)):
            if i == 0:
                gait_event = []
            gait_event.append(steps[i])
            if i == len(steps)-1:
                if len(gait_event)>x:
                    gait_events.append(gait_event)
            elif (steps[i+1].start_index - steps[i].start_index) > (y * self.recorded_framerate):
                # y s, where the mouse didn't do a step
                # create new gait event
                if len(gait_event)>x:
                    gait_events.append(gait_event)
                gait_event = []
        # gait_event as EventSeries including/not including bouts?
        return gait_events
    
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
        self.turns_to_closed_after_gait = [turn for sublist in [self._bout_after_gait_event(gait_event = gait_event, event = self.turns_to_closed) for gait_event in gait_events] for turn in sublist if type(turn)==EventBout2D]
        self.turns_to_open_after_gait = [turn for sublist in [self._bout_after_gait_event(gait_event = gait_event, event = self.turns_to_open) for gait_event in gait_events] for turn in sublist if type(turn)==EventBout2D]
        gait_disruption_bouts = [disruption for sublist in [self._bout_after_gait_event(gait_event = gait_event, event = self.immobility_bouts.events) for gait_event in gait_events] for disruption in sublist if type(disruption)==EventBout2D]
        self.gait_disruption_bouts = EventSeries(range_end = self.bodyparts['Snout'].df.shape[0], events = gait_disruption_bouts, recorded_framerate = self.recorded_framerate, range_start = 0)
        self.gait_disruption_bouts.run_basic_operations_on_events(facing_towards_open_end = self.facing_towards_open_end, centerofgravity = self.bodyparts['centerofgravity'])
        freezing_of_gait_events = [freezing for sublist in [self._bout_after_gait_event(gait_event = gait_event, event = self.freezing_bouts.events) for gait_event in gait_events] for freezing in sublist if type(freezing)==EventBout2D]
        self.freezing_of_gait_events = EventSeries(range_end = self.bodyparts['Snout'].df.shape[0], events = freezing_of_gait_events, recorded_framerate = self.recorded_framerate, range_start = 0)
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
            if bout.start_index in range(gait_event[0].start_index, gait_event[-1].start_index + self.recorded_framerate):
                bouts.append(bout)
        return bouts
        
        
    def _calculate_angles(self)->None:
        """
        creates angles objects that are interesting for gait analysis
        """
        self.angle_hindkneeleft = Angle2D(bodypart_a = self.bodyparts['HindKneeLeft'], bodypart_b = self.bodyparts['BackAnkleLeft'], object_to_calculate_angle=self.bodyparts['HipLeft'])
        self.angle_backankleleft = Angle2D(bodypart_a = self.bodyparts['BackAnkleLeft'], bodypart_b = self.bodyparts['HindPawLeft'], object_to_calculate_angle=self.bodyparts['HindKneeLeft'])
        self.angle_hipleft = Angle2D(bodypart_a = self.bodyparts['HipLeft'], bodypart_b = self.bodyparts['HindKneeLeft'], object_to_calculate_angle=self.bodyparts['IliacCrestLeft'])
        
        self.angle_hindkneeright = Angle2D(bodypart_a = self.bodyparts['HindKneeRight'], bodypart_b = self.bodyparts['BackAngleRight'], object_to_calculate_angle=self.bodyparts['HipRight'])
        self.angle_backankleright = Angle2D(bodypart_a = self.bodyparts['BackAngleRight'], bodypart_b = self.bodyparts['HindPawRight'], object_to_calculate_angle=self.bodyparts['HindKneeRight'])
        self.angle_hipright = Angle2D(bodypart_a = self.bodyparts['HipRight'], bodypart_b = self.bodyparts['HindKneeRight'], object_to_calculate_angle=self.bodyparts['IliacCrestRight'])
        
        self.angle_wristleft = Angle2D(bodypart_a = self.bodyparts['WristLeft'], bodypart_b = self.bodyparts['ForePawLeft'], object_to_calculate_angle=self.bodyparts['ElbowLeft'])
        self.angle_elbowleft = Angle2D(bodypart_a = self.bodyparts['ElbowLeft'], bodypart_b = self.bodyparts['ShoulderLeft'], object_to_calculate_angle=self.bodyparts['WristLeft'])
        
        self.angle_wristright = Angle2D(bodypart_a = self.bodyparts['WristRight'], bodypart_b = self.bodyparts['ForePawRight'], object_to_calculate_angle=self.bodyparts['ElbowRight'])
        self.angle_elbowright = Angle2D(bodypart_a = self.bodyparts['ElbowRight'], bodypart_b = self.bodyparts['ShoulderRight'], object_to_calculate_angle=self.bodyparts['WristRight'])

    
    def _add_angles_to_steps(self, gait_events: List)->None:
        pass
    
    def _calculate_parameters_for_gait_analysis(self)->None:
        self.hind_stance_right = Stance2D(paw=self.bodyparts['HindPawRight'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['TailBase'])
        self.hind_stance_left = Stance2D(paw=self.bodyparts['HindPawLeft'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['TailBase'])
        self.fore_stance_right = Stance2D(paw=self.bodyparts['ForePawRight'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['Snout'])
        self.fore_stance_left = Stance2D(paw=self.bodyparts['ForePawLeft'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['Snout'])
        
        self.hind_stance = self.hind_stance_right.parameter_array + self.hind_stance_left.parameter_array
        self.fore_stance = self.fore_stance_right.parameter_array + self.fore_stance_left.parameter_array
        
        #Step Length
        #Stride Length
        
        #Gait Symmetry
        
    
    def _create_PSTHs(self)->None:
        pass
    
    def _get_tracking_stability_in_gait_periodes(self, gait_periodes: List)->None:
        """
        Function, that calculates the percentage, in which a marker was detected during all gait periodes.
        
        This should reflect the quality of marker detection better than the whole session percentage.
        
        Parameters:
            gait_periods(List): nested list with sublists that represent single gait events containing single step indices
        """
        tracking_dict = dict.fromkeys(self.bodyparts.keys())
        for bodypart in self.bodyparts.values():
            tracking_dict[bodypart.id]=[]
            for gait_event in gait_periodes:
                tracking_dict[bodypart.id].append(bodypart.check_tracking_stability(start_end_index=(gait_event[0].start_index, gait_event[-1].start_index)))
            tracking_dict[bodypart.id]=mean(tracking_dict[bodypart.id])
        self.tracking_performance.loc['over_gait_events', :] = tracking_dict
        
        

        
class Parameter2D():
    def __init__(self):
        pass

        
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
        # to reduce computation time outcomment the two lines below
        #self._exclude_frames()
        #self._interpolate_low_likelihood_frames()
    
    
    def _identify_duplicates(self)->None:
        pass
    
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
        angle = - rotation_angle #counterclockwise
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
        elif self.id in set(['Back', 'Front']):
            self.df.loc[self.df['likelihood']<self.dlc_likelihood_threshold**3, ('x', 'y')] = np.NaN
        else:
            self.df.loc[self.df['likelihood']<self.dlc_likelihood_threshold, ('x', 'y')] = np.NaN
            
    def _interpolate_low_likelihood_frames(self) -> None:
        #Data smoothening:
        m = np.arange(0, self.df.shape[0])
        x = np.nan_to_num(self.df['x'], copy=True)
        spline = interpolate.UnivariateSpline(m, x, s=1)
        self.df['x'] = spline(m)
        y = np.nan_to_num(self.df['y'], copy=True)
        spline = interpolate.UnivariateSpline(m, y, s=1)
        self.df['y'] = spline(m)

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
        return 1.
    
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
        self._create_dict()
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
        self.x_position=centerofgravity.df.loc[self.start_index, 'x']
        self.dict['x_position']=self.x_position

    def _create_dict(self)->None:
        """
        Function that sets the attribut self.dict as Dictionary.
        """
        self.dict = {}
        
        
class Angle2D():
    """
    Class that creates an object for an Angle over a Recording between instances.
    """
    def __init__(self, bodypart_a: 'Bodypart2D', bodypart_b: 'Bodypart2D', object_to_calculate_angle: 'Bodypart2D')->None:
        """
        Constructor for class Angle.
        
        The class contains functions to calculate the angle of 3 bodyparts to each other at the first given bodypart (bodypart_a).
        
        Parameters:
            bodypart_a(Bodypart2D): bodypart, at which the angle is calculated.
            bodypart_b(Bodypart2D): second bodypart, necessary for defining a line
            bodypart_c(Bodypart2D): third bodypart, necessary for calculating the angle of the line between it and bodypart_a and _b
        
        """
        self.bodypart_a = bodypart_a
        self.bodypart_b = bodypart_b
        self.bodypart_c = object_to_calculate_angle
        self.parameter_array = self._calculate_angle_between_three_bodyparts()
        
    def _calculate_angle_between_three_bodyparts(self)->np.array:
        """
        Calculates angle at bodypart_a.
        
        After calculating the length of the sides of a triangle ABC, the angles are calculated using law of cosines.
        
        Returns:
            angle(np.array): angle over the whole recording stored in an numpy.Array
        """
        length_a = self._get_length_in_2d_space(self.bodypart_b, self.bodypart_c)
        length_b = self._get_length_in_2d_space(self.bodypart_a, self.bodypart_c)
        length_c = self._get_length_in_2d_space(self.bodypart_a, self.bodypart_b)
        return self._get_angle_from_law_of_cosines(length_a, length_b, length_c)
    
    def _get_length_in_2d_space(self, object_a: 'Bodypart2D', object_b: 'Bodypart2D') -> np.array:
        """
        Calculates the length between two objects in 2D. 
        
        Parameters:
            object_a(Bodypart)
            object_b(Bodypart)
            
        Returns:
            length(np.array): Length between two objects over the whole recording stored as an numpy.Array.
        """
        length = np.sqrt((object_a.df['x']-object_b.df['x'])**2 + 
                         (object_a.df['y']-object_b.df['y'])**2)
            
        return length
    
    def _get_angle_from_law_of_cosines(self, length_a: np.array, length_b: np.array, length_c: np.array)->np.array:
        """
        Function that calculates the angle at corner A of a triangle ABC.
        
        After calculating cos(a) using the law of cosines, the inverted cosinus is calculated and it is converted from radians in degrees.
        https://en.wikipedia.org/wiki/Law_of_cosines
        
        Returns: 
            np.array: Angle at corner A in degrees.
        """
        cos_angle = (length_b**2 + length_c**2 - length_a**2) / (2 * length_b * length_c)
        return np.degrees(np.arccos(cos_angle))
        
                                           
class Stance2D():
    """
    Class for calculation of the given paw to the bodyaxis as defined by object_a and object_b.
    
    Attributes:
        self.paw (Bodypart): Bodypart representation of the paw
        self.object_a: object, which defines the Bodyaxis
        self.object_b: object, which defines the Bodyaxis
        self.parameter_array: array, that contains the calculated distance (Stance) for every frame
    """
    def __init__(self, paw: 'Bodypart2D', object_a: 'Bodypart2D', object_b: 'Bodypart2D')->None:
        """
        Constructor for class Stance2D. It calls the functions to calculate the stance already.
        
        The Stance is stored in self.parameter_array
        
        Parameters:
            self.paw (Bodypart): Bodypart representation of the paw
            self.object_a: object, which defines the Bodyaxis
            self.object_b: object, which defines the Bodyaxis
        """
        self.paw = paw
        self.object_a = object_a
        self.object_b = object_b
        s = self._point_on_line_orthogonal_to_paw()
        self.parameter_array = self._calculate_distance(s=s)
        
    def _calculate_distance(self, s: Tuple[int, int])->float:
        """
        Function to calculate the distance between a point s and the paw.

        Returns:
            length(float): distance between s and the paw.
        """
        length = np.sqrt((self.paw.df['x']- s[0])**2 + 
                             (self.paw.df['y']-s[1])**2)
        return length
    
    
    def _point_on_line_orthogonal_to_paw(self)->Tuple[int, int]:
        """
        Function, that finds the point on the bodyaxis with the shortest distance to the given paw.
        
        First, the slope of the bodyaxis and the orthogonal on the bodyaxis is calculated as m1, m2. 
        Next, the intersection between the two lines and the y_axis is calculated as t1, t2.
        Last step, the coordinates of the intersection point sx and sy are calculated.
        
        Returns:
            Tuple: coordinates of the point on the bodyaxis with the shortest distance to the given paw.
        """
        #calculates the distance between the line given by a and b (intersection = s) and a point c
        m1 = (self.object_a.df['y'] - self.object_b.df['y']) / (self.object_a.df['x'] - self.object_b.df['x'])
        m2 = 1/(-m1)
        t1 = self.object_a.df['y'] - m1 * self.object_a.df['x']
        t2 = self.paw.df['y'] - m2 * self.paw.df['x']   
        sx = (t2 - t1)/(m1 - m2)
        sy = m2 * sx + t2
        return (sx, sy)
    
    
class Step():
    def __init__(self, paw: str, start_index: int)->None:
        self.start_index=start_index
        self.paw=paw
                
    def _calculate_end_index(self)->None:
        pass
    
    
class EventSeries(ABC):
    @property
    def merge_threshold(self)->float:
        #in seconds
        return 0.2
    
    
    def __init__(self, range_end: int, events: List, recorded_framerate: int, range_start: int = 0):
        self.events = self._merge_events(events = events, recorded_framerate = recorded_framerate)
        
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
                    while ((events[j].start_index - events[i].end_index)/recorded_framerate) < self.merge_threshold:
                        j += 1
                except IndexError: 
                    j -= 1
                events_to_keep.append(EventBout2D(start_index = events[i].start_index, end_index = events[j].end_index, recorded_framerate=recorded_framerate))
                for n in range(i, j):
                    events.pop(n+1)
            else:
                events_to_keep.append(events[i])
        return events_to_keep
        
    def run_basic_operations_on_events(self, facing_towards_open_end: pd.Series, centerofgravity: Bodypart2D):
        for event in self.events:
            event.check_direction(facing_towards_open_end=facing_towards_open_end)
            event.check_that_freezing_threshold_was_reached()
            event.get_position(centerofgravity= centerofgravity)