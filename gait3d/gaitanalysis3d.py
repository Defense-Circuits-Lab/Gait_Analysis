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


class Recording3D(ABC):
    """
    Class for Analysing 3D-Position Data of mice in the OpeningTrack.
    
    Attributes:
        full_df_from_csv(pandas.DataFrame): the Dataframe containing all bodyparts with x, y, z-coordinates and reprojection error as returned by Anipose
        recorded_framerate(int): fps of the recording
        metadata(Dict): dictionary containing information read from the filename, such as animal_id, recording_date and Opening Track paradigm
    """
    
    def __init__(self, filepath: Path, recorded_framerate: int)->None:
        """
        Constructor for the Recording3D class.
        
        This function calls functions to get the Dataframe from the csv, that is given as filepath argument and to read metadata from the filename.
        
        Parameters:
            filepath(pathlib.Path): the filepath to the csv containing Anipose data
            recorded_framerate(int): fps of the recording
        """
        self.full_df_from_csv = self._get_df_from_csv(filepath = filepath)
        self.recorded_framerate = recorded_framerate
        self.metadata = self._retrieve_metadata(filepath = filepath)
        
        
    def _get_df_from_csv(self, filepath: Path)->pd.DataFrame:
        """
        Reads the Dataframe from the csv-file.
        
        Parameters:
            filepath(pathlib.Path): the path linked to the csv.file
        Returns:
            pandas.DataFrame: containing all bodyparts with x, y, z-coordinates and reprojection error as returned by Anipose
        """
        if not filepath.endswith('.csv'):
            raise ValueError('The Path you specified is not linking to a .csv-file!')
        return pd.read_csv(filepath)
    
    
    def _retrieve_metadata(self, filepath: Path)->Dict:
        """
        Function, that slices the Filename to get the encoded metadata.
        
        Relying on file naming like this: 196_F7-27_220826_OTT.csv
        
        Parameters:
            filepath(pathlib.Path): the path linked to the csv.file
        Returns:
            Dict: containing date of recording, animal_id and OT paradigm
        """
        filepath_slices = filepath.split('_')
        animal_line, animal_id, recording_date, paradigm = filepath_slices[0], filepath_slices[1], filepath_slices[2], filepath_slices[3]
        return {'recording_date': recording_date, 'animal': animal_line + '_' + animal_id, 'paradigm': paradigm}
        
        
    def run(self)->None:
        """
        Function to create Bodypart objects for all the tracked markers.
        
        A function is called, that first calculates the centerofgravity and then creates Bodypart objects for all the markers.
        The coordinate system is then normalized via translation, rotation and conversion to unit cms.
        Basic parameters for the bodyparts are already calculated, such as, speed and immobility.
        Currently no frames are excluded.
        """
        self._calculate_center_of_gravity()
        #todo: calculate error?
        self._create_all_bodyparts()
        self._normalize_coordinate_system()
        #add error to find best matching frame?
        self._run_basic_operations_on_bodyparts()
        self._get_tracking_stability()
        

    def _calculate_center_of_gravity(self)->None:
        """
        Function, that calculates the centerofgravity.
        
        The center_of_gravity is calculated using the bodyparts Snout and TailBase.
        It adds centerofgravity to self.full_df_from_csv.
        """
        for coordinate in ['x', 'y', 'z']:
            self.full_df_from_csv[f'centerofgravity_{coordinate}'] = (self.full_df_from_csv[f'Snout_{coordinate}'] + self.full_df_from_csv[f'TailBase_{coordinate}'])/2
        self.full_df_from_csv['centerofgravity_error'] = 0
            
        
    def _create_all_bodyparts(self)->None:
        """
        Function, that creates a Dictionary with all Bodypart objects.
        
        The dictionary uses the label given from Deeplabcut tracking as key for the Bodypart objects.
        It sets the dictionary as self.bodyparts.
        """
        self.bodyparts = {}
        for key in self.full_df_from_csv.keys():
            bodypart = key.split('_')[0]
            if bodypart not in self.bodyparts.keys() and bodypart not in set (['M', 'center', 'fnum']):
                self.bodyparts[bodypart] = Bodypart(bodypart_id = bodypart, df = self.full_df_from_csv)
                
    
    
    def _normalize_coordinate_system(self)->None:
        """
        This Function normalizes the coordinate system.
        
        The mazecorners of the best frame are used to calculate necessary parameters for the following functions, 
        such as the conversion factor from the intrinsic unit to cms, 
        the translation vector from the real-world-coordinate system to the null-space,
        and the rotation_matrix for 3D rotation, based on calculation of Euler-angles.
        With this parameters it calls the normalization for each single bodypart.
        """
        mazecorners = self._fix_coordinates_of_maze_corners()
        conversion_factor = self._get_conversion_factor_to_cm(reference_points = mazecorners)
        translation_vector = self._get_translation_vector(reference_points = mazecorners)
        rotation_matrix = self._get_rotation_matrix(reference_points = mazecorners)
        
        for bodypart in self.bodyparts.values():
            bodypart.normalize_df(translation_vector = translation_vector, rotation_matrix = rotation_matrix, conversion_factor = conversion_factor)
    
    def _find_best_mazecorners_for_normalization(self)->int:
        """
        Function to find the frame, where the mazecorners are tracked best.
        
        It creates Angle objects for the four corners and calculates the length of the long maze sides.
        This parameters are used to create a weighted residual mean of squares function to find the best matching frame.
        https://en.wikipedia.org/wiki/Residual_sum_of_squares
        
        Returns:
            best_matching_frame(int): the index of the frame
        """
        angle_open_right = Angle(bodypart_a = self.bodyparts['MazeCornerOpenRight'], bodypart_b = self.bodyparts['MazeCornerClosedRight'], object_to_calculate_angle = self.bodyparts['MazeCornerOpenLeft'])
        angle_open_left = Angle(bodypart_a = self.bodyparts['MazeCornerOpenLeft'], bodypart_b = self.bodyparts['MazeCornerClosedLeft'], object_to_calculate_angle = self.bodyparts['MazeCornerOpenRight'])
        angle_closed_right = Angle(bodypart_a = self.bodyparts['MazeCornerClosedRight'], bodypart_b = self.bodyparts['MazeCornerOpenRight'], object_to_calculate_angle = self.bodyparts['MazeCornerClosedLeft'])    
        angle_closed_left = Angle(bodypart_a = self.bodyparts['MazeCornerClosedLeft'], bodypart_b = self.bodyparts['MazeCornerOpenLeft'], object_to_calculate_angle = self.bodyparts['MazeCornerClosedRight'])
        
        length_right_side = np.sqrt(np.sum(self.bodyparts['MazeCornerOpenRight'].df_raw.loc[:, ('x', 'y', 'z')].values - self.bodyparts['MazeCornerClosedRight'].df_raw.loc[:, ('x', 'y', 'z')].values, axis=1)**2)
        length_left_side = np.sqrt(np.sum(self.bodyparts['MazeCornerOpenLeft'].df_raw.loc[:, ('x', 'y', 'z')].values - self.bodyparts['MazeCornerClosedLeft'].df_raw.loc[:, ('x', 'y', 'z')].values, axis=1)**2)
        
        w_angle = 1
        w_sides = 4
        #w_error = 4, use reprojection error in formula?
        frame_errors = []
        n = 4*w_angle + w_sides
        for frame in range(self.full_df_from_csv.shape[0]):
            weighted_residual_mean_of_squares = math.sqrt((w_angle*(angle_open_right.parameter_array[frame]-90)**2
                                                               +w_angle*(angle_open_left.parameter_array[frame]-90)**2
                                                               +w_angle*(angle_closed_right.parameter_array[frame]-90)**2
                                                               +w_angle*(angle_closed_left.parameter_array[frame]-90)**2
                                                               +w_sides*(length_right_side[frame]-length_left_side[frame])**2)/n)
            frame_errors.append(weighted_residual_mean_of_squares)
            # alle Fehler normieren? z.B. ((angle-90)/90)**2, ((length_right_side-length_left_side)/length_right_side)**2
        """
        weighted_residual_mean_of_squares = np.sqrt((w_angle*(angle_open_right.parameter_array-90)**2
                                                               +w_angle*(angle_open_left.parameter_array-90)**2
                                                               +w_angle*(angle_closed_right.parameter_array-90)**2
                                                               +w_angle*(angle_closed_left.parameter_array-90)**2
                                                               +w_sides*(length_right_side[frame]-length_left_side)**2)/n)
        """
        best_matching_frame = frame_errors.index(min(frame_errors))
        #print(angle_closed_left.parameter_array[best_matching_frame], angle_closed_right.parameter_array[best_matching_frame], angle_open_left.parameter_array[best_matching_frame], angle_open_right.parameter_array[best_matching_frame])
        return best_matching_frame
    
    def _get_conversion_factor_to_cm(self, reference_points: Tuple[np.array, np.array, np.array])->float:
        """
        Function to get the conversion factor of the unspecified unit to cm.
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
            
        Returns:
            conversion_factor(float): factor to convert the unspecified unit into cm.
        """
        conversion_factor = (50/np.sqrt(sum((reference_points[1]-reference_points[0])**2)))
        return conversion_factor
    
    def _get_translation_vector(self, reference_points:  Tuple[np.array, np.array, np.array])->np.array:
        """
        Function that calculates the offset of the real-world-coordinate system to the null space.
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
            
        Returns:
            translation_vector(np.array): vector with offset in each dimension
        """
        translation_vector = -reference_points[0]
        return translation_vector
    
    def _get_rotation_matrix(self, reference_points: Tuple[np.array, np.array, np.array])->Rotation:
        """
        Function that calculates the Rotation matrix.
        
        It calculates the Euler angles between the real-world-coordinate system (xyz) to the null space (XYZ) first,
        and inserts them into the rotation matrix.
        https://en.wikipedia.org/wiki/Euler_angles
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
        
        Returns:
            scipy.spatial.transform.Rotation: Rotation matrix from the three calculated angles
        """
        # The axes of the real-world-coordinate system are calculated
        x_axis, y_axis, z_axis = self._get_axes(reference_points=reference_points)
        # The angle between the x_axis and the intersection of the z_axis and the Z_axis are calculated.
        x_angle = self._calculate_angle(vector1=x_axis, vector2 = np.cross(z_axis, np.array([0, 0, 1])))
        # The xyz-system is rotated by that angle.
        reference_points_z = self._apply_single_rotation(angle = x_angle, reference_points=reference_points, axis = 'z')
        
        # The axes of the rotated coordinate system are calculated
        x_axis2, y_axis2, z_axis2 = self._get_axes(reference_points=reference_points_z)
        # The angle between the rotated z_axis and the Z_axis are calculated.
        z_angle = self._calculate_angle(vector1=z_axis2, vector2 = np.array([0, 0, 1]))
        # The xyz-system is rotated by that angle.
        reference_points_zx = self._apply_single_rotation(angle = z_angle, reference_points=reference_points_z, axis = 'x')

        # The axes of the rotated coordinate system are calculated
        x_axis3, y_axis3, z_axis3 = self._get_axes(reference_points=reference_points_zx)
        # The angle between the intersection of the z_axis and the Z_axis and the x_axis are calculated.
        x_angle2 = self._calculate_angle(vector1=np.cross(z_axis, np.array([0, 0, 1])), vector2 = np.array([1, 0, 0]))
        
        return Rotation.from_euler('zxz', [x_angle, z_angle, x_angle2], degrees=True)
   
    def _calculate_angle(self, vector1:np.array, vector2: np.array)->float:
        """
        Function to calculate the angle between two vectors.
        
        Parameters:
            vector1(np.array): first vector
            vector2(np.array): second vector
            
        Returns:
            float: the calculated angle in degrees
        """
        angle = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
        return math.degrees(math.acos(angle))
    
    def _apply_single_rotation(self, angle: float, reference_points: Tuple[np.array, np.array, np.array], axis: str)->Tuple[np.array, np.array, np.array]:
        """
        Function that rotates the reference points around a given axis by a specified angle.
        
        It calculates the rotation matrix and applies it to the reference points.
        
        Parameters:
            angle(float): rotation angle 
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
            axis: specifies the axis on which the rotation is performed
            
        Returns:
            Tuple: rotated reference_points
        
        """
        r = Rotation.from_euler(axis, [-angle], degrees=True)
        reference_points_rotated = []
        for point in reference_points:
             reference_points_rotated.append(r.apply(point, inverse=True)[0])
        return reference_points_rotated
    
    def _fix_coordinates_of_maze_corners(self)->Tuple[np.array, np.array, np.array]:
        """
        Function that creates the reference_points from the mazecorners.
        
        After finding the best matching frame it combines the -x, -y, -z coordinate of this frame for all corners into a np.array.
        
        Returns:
            reference_points(Tuple): the best tracked maze corners as numpy.Array
        """
        frame = self._find_best_mazecorners_for_normalization()
        
        mazecorneropenright = np.array([self.bodyparts['MazeCornerOpenRight'].df_raw.loc[frame, 'x'], self.bodyparts['MazeCornerOpenRight'].df_raw.loc[frame, 'y'], self.bodyparts['MazeCornerOpenRight'].df_raw.loc[frame, 'z']])
        mazecornerclosedright = np.array([self.bodyparts['MazeCornerClosedRight'].df_raw.loc[frame, 'x'], self.bodyparts['MazeCornerClosedRight'].df_raw.loc[frame, 'y'], self.bodyparts['MazeCornerClosedRight'].df_raw.loc[frame, 'z']])
        mazecorneropenleft = np.array([self.bodyparts['MazeCornerOpenLeft'].df_raw.loc[frame, 'x'], self.bodyparts['MazeCornerOpenLeft'].df_raw.loc[frame, 'y'], self.bodyparts['MazeCornerOpenLeft'].df_raw.loc[frame, 'z']])
        mazecornerclosedleft = np.array([self.bodyparts['MazeCornerClosedLeft'].df_raw.loc[frame, 'x'], self.bodyparts['MazeCornerClosedLeft'].df_raw.loc[frame, 'y'], self.bodyparts['MazeCornerClosedLeft'].df_raw.loc[frame, 'z']])
        return mazecornerclosedright, mazecorneropenright, mazecornerclosedleft

    def _get_axes(self, reference_points: Tuple[np.array, np.array, np.array])->Tuple[np.array, np.array, np.array]:
        """
        Calculates the axis-vectors from given reference points.
        
        Parameters:
            reference_points (Tuple): containing the vecotrs as np.arrays of the best tracked mazecorners
        Returns:
            Tuple: the calculated axes
        """
        x_axis = reference_points[0]-reference_points[1]
        y_axis = reference_points[2]-reference_points[0]
        # the cross product returns an vector, that is orthogonal on both of the factorized vectors, which is the z-axis in our case.
        z_axis = np.cross(x_axis, y_axis)
        # Could point in the negative direction as well!
        return x_axis, y_axis, z_axis
                
        
        
    def _run_basic_operations_on_bodyparts(self)->None:
        """ Basic parameters for the bodyparts are calculated, such as, speed and immobility. """
        for bodypart in self.bodyparts.values():
            bodypart.run_basic_operations(recorded_framerate = self.recorded_framerate)            
            
    def _get_tracking_stability(self)->None:
        """
        Function, that calculates the percentage of frames, in which a marker was detected.
        
        It sets self.tracking_stability as pandas.DataFrame with columns for all markers and the percentage value as first row.
        """
        tracking_dict = {bodypart.id: bodypart.check_tracking_stability() for bodypart in self.bodyparts.values()}
        #calculate standard derivation for fixed markers
        #for bodypart in set(['MazeCornerOpenLeft', 'MazeCornerOpenRight', 'MazeCornerClosedLeft', 'MazeCornerClosedRight', 'LED5']): 
            #tracking_dict['standard_derivation']=np.stddev([])
        self.tracking_stability = pd.DataFrame([tracking_dict], index=['over_total_session', 'over_gait_events'])
        
    
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
        self.facing_towards_open_end.loc[(self.bodyparts['Snout'].df.loc[:, 'x']>self.bodyparts['EarLeft'].df.loc[:, 'x']) &
                                    (self.bodyparts['Snout'].df.loc[:, 'x']>self.bodyparts['EarRight'].df.loc[:, 'x'])] = True
        
        
    def _get_turns(self)->None:
        """
        Function that checks for turning events.
        
        Based on self.facing_towards_open_end, the indices of events, where a mouse turns are extracted and the attributes
        self.turns_to_closed and self.turns_to_open are created as a list of EventBouts.
        """
        turn_indices = self.facing_towards_open_end.where(self.facing_towards_open_end.diff()==True).dropna().index
        self.turns_to_closed=[EventBout(start_index = start_index) for start_index in turn_indices if self.facing_towards_open_end[start_index-1]==True]
        self.turns_to_open=[EventBout(start_index = start_index) for start_index in turn_indices if self.facing_towards_open_end[start_index-1]==False]
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
        return pd.Series(np.zeros_like(np.arange(self.full_df_from_csv.shape[0]), dtype = dtype))
    
                    
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
        #since the first frame is allways immobile, the first change in all freezing bodyparts can be seen as the start of a immobility episode
        #this function finds all immobility_bouts, checks for directionality and whether the freezing threshold was reached
        changes_from_immobility_to_mobility = self.all_freezing_bodyparts_immobile.where(self.all_freezing_bodyparts_immobile.diff()==True).dropna()
        start_indices_of_immobility_bouts = changes_from_immobility_to_mobility[::2]
        end_indices_of_immobility_bouts = changes_from_immobility_to_mobility[1::2]
        self.immobility_bouts = []
        for i in range(len(start_indices_of_immobility_bouts)):
            start_index, end_index = start_indices_of_immobility_bouts.index[i], end_indices_of_immobility_bouts.index[i]
            self.immobility_bouts.append(EventBout(start_index = start_index, end_index = end_index))
            
    def _run_operations_on_immobility_bouts(self)->None:
        """
        Basic operations are run on the Immobility Bouts.
        
        A pandas.DataFrame for immobility bouts is created and set as self.immobility_bout_df.
        """
        for immobility_bout in self.immobility_bouts:
            immobility_bout.check_direction(facing_towards_open_end=self.facing_towards_open_end)
            immobility_bout.check_that_freezing_threshold_was_reached(recorded_framerate=self.recorded_framerate)
            immobility_bout.get_position(centerofgravity=self.bodyparts["centerofgravity"])
        self.immobility_bout_df = pd.DataFrame([immobility_bout.dict for immobility_bout in self.immobility_bouts])
            
            
    def _collect_freezing_bouts(self)->None:
        """
        The Immobility bouts, where the freezing threshold was exceeded are collected as freezing bouts.
        
        A pandas.DataFrame for freezing bouts is created and set as self.freezing_bout_df.
        """
        self.freezing_bouts = []
        for immobility_bout in self.immobility_bouts:
            if immobility_bout.freezing_threshold_reached:
                self.freezing_bouts.append(immobility_bout)
        self.freezing_bout_df = pd.DataFrame([freezing_bout.dict for freezing_bout in self.freezing_bouts])
         
    def run_gait_analysis(self)->None:
        """
        Function, that runs functions, necessary for gait analysis.
        
        Angles between bodyparts of interest are calculated as Angle objects.
        A peak detection algorithm on paw_speed is used to detect steps.
        EventBouts are created.
        """
        steps = self._detect_steps()
        gait_periodes = self._define_gait_periodes(steps=steps)
        self._get_gait_event_bouts(gait_events=gait_periodes)
        self._get_tracking_stability_in_gait_periodes(gait_periodes=gait_periodes)
        self._calculate_parameters_for_gait_analysis()
        self._calculate_angles()
        self._add_parameters_to_steps(gait_events=gait_periodes)        
        
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
        
        If more than 3 steps are done in less than 3 seconds, those steps are considered a gait_event.

        Parameters:
            steps(List): list of steps with Step objects
        Returns:
            gait_events(List): nested list with sublists that represent single gait events containing single step indices
        """
        gait_events = []
        for i in range(len(steps)):
            if i == 0:
                gait_event = []
            gait_event.append(steps[i])
            if i == len(steps)-1:
                if len(gait_event)>3:
                    gait_events.append(gait_event)
            elif (steps[i+1].start_index - steps[i].start_index) > 3*self.recorded_framerate:
                # 3s, where the mouse didn't do a step
                # create new gait event
                if len(gait_event)>3:
                    gait_events.append(gait_event)
                gait_event = []
            # create EventBouts for Gait?
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
        self.turns_to_closed_after_gait = [turn for turn in [self._bout_after_gait_event(gait_event = gait_event, event = self.turns_to_closed) for gait_event in gait_events] if type(turn)==EventBout]
        self.turns_to_open_after_gait = [turn for turn in [self._bout_after_gait_event(gait_event = gait_event, event = self.turns_to_open) for gait_event in gait_events] if type(turn)==EventBout]
        self.gait_disruption_bouts = [disruption for disruption in [self._bout_after_gait_event(gait_event = gait_event, event = self.immobility_bouts) for gait_event in gait_events] if type(disruption)==EventBout]
        self.freezing_of_gait_events = [freezing for freezing in [self._bout_after_gait_event(gait_event = gait_event, event = self.freezing_bouts) for gait_event in gait_events] if type(freezing)==EventBout]

            
    def _bout_after_gait_event(self, gait_event: List, event: List)->Union:
        """
        Function that checks the in and after a gait_event for an specified event.
        
        Parameters:
            frame_index(int): last index of a gait_event
            event(List): List with Event Bouts to check for
        Returns:
            Union: the Eventbout is returned if found, otherwise None
        """
        for bout in event:
            if bout.start_index in range(gait_event[0].start_index, gait_event[-1].start_index+self.recorded_framerate):
                return bout
                break
            else:
                pass
            
        
    def _calculate_angles(self)->None:
        """
        creates angles objects that are interesting for gait analysis
        """
        self.angle_hindkneeleft = Angle(bodypart_a = self.bodyparts['HindKneeLeft'], bodypart_b = self.bodyparts['BackAnkleLeft'], object_to_calculate_angle=self.bodyparts['HipLeft'])
        self.angle_backankleleft = Angle(bodypart_a = self.bodyparts['BackAnkleLeft'], bodypart_b = self.bodyparts['HindPawLeft'], object_to_calculate_angle=self.bodyparts['HindKneeLeft'])
        self.angle_hipleft = Angle(bodypart_a = self.bodyparts['HipLeft'], bodypart_b = self.bodyparts['HindKneeLeft'], object_to_calculate_angle=self.bodyparts['IliacCrestLeft'])
        
        self.angle_hindkneeright = Angle(bodypart_a = self.bodyparts['HindKneeRight'], bodypart_b = self.bodyparts['BackAngleRight'], object_to_calculate_angle=self.bodyparts['HipRight'])
        self.angle_backankleright = Angle(bodypart_a = self.bodyparts['BackAngleRight'], bodypart_b = self.bodyparts['HindPawRight'], object_to_calculate_angle=self.bodyparts['HindKneeRight'])
        self.angle_hipright = Angle(bodypart_a = self.bodyparts['HipRight'], bodypart_b = self.bodyparts['HindKneeRight'], object_to_calculate_angle=self.bodyparts['IliacCrestRight'])
        
        self.angle_wristleft = Angle(bodypart_a = self.bodyparts['WristLeft'], bodypart_b = self.bodyparts['ForePawLeft'], object_to_calculate_angle=self.bodyparts['ElbowLeft'])
        self.angle_elbowleft = Angle(bodypart_a = self.bodyparts['ElbowLeft'], bodypart_b = self.bodyparts['ShoulderLeft'], object_to_calculate_angle=self.bodyparts['WristLeft'])
        
        self.angle_wristright = Angle(bodypart_a = self.bodyparts['WristRight'], bodypart_b = self.bodyparts['ForePawRight'], object_to_calculate_angle=self.bodyparts['ElbowRight'])
        self.angle_elbowright = Angle(bodypart_a = self.bodyparts['ElbowRight'], bodypart_b = self.bodyparts['ShoulderRight'], object_to_calculate_angle=self.bodyparts['WristRight'])
    
    def _add_parameters_to_steps(self, gait_events: List)->None:
        gait_event = gait_events[2]
        for parameter in [self.angle_hindkneeleft.parameter_array, self.angle_hindkneeright.parameter_array, self.angle_backankleleft.parameter_array, self.angle_backankleright.parameter_array, self.angle_hipleft.parameter_array, self.angle_hipright.parameter_array, self.angle_wristleft.parameter_array, self.angle_wristright.parameter_array, self.angle_elbowleft.parameter_array, self.angle_elbowright.parameter_array, self.hind_stance, self.fore_stance]:
            plt.close()
            fig = plt.figure()
            plt.plot(parameter[gait_event[0].start_index:gait_event[-1].start_index])
            plt.show()
        pass
    
    def _calculate_parameters_for_gait_analysis(self)->None:        
        self.hind_stance_right = Stance(paw=self.bodyparts['HindPawRight'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['TailBase'])
        self.hind_stance_left = Stance(paw=self.bodyparts['HindPawLeft'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['TailBase'])
        self.fore_stance_right = Stance(paw=self.bodyparts['ForePawRight'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['Snout'])
        self.fore_stance_left = Stance(paw=self.bodyparts['ForePawLeft'], object_a=self.bodyparts['centerofgravity'], object_b=self.bodyparts['Snout'])
        
        self.hind_stance = self.hind_stance_right.parameter_array + self.hind_stance_left.parameter_array
        self.fore_stance = self.fore_stance_right.parameter_array + self.fore_stance_left.parameter_array
        
        """
        Step Length
        Stride Length
                
        Gait Symmetry
        
        Angle Paw-BackAnkle/Wrist - Bodyaxis(Snout/TailBase/CoG)
        """
        pass
    
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
        self.tracking_stability.loc['over_gait_events', :] = tracking_dict

                
class Step():
    def __init__(self, paw: str, start_index: int)->None:
        self.start_index=start_index
        self.paw=paw
                
    def _calculate_end_index(self)->None:
        pass
    
                                               
class Stance():
    """
    Class for calculation of the given paw to the bodyaxis as defined by object_a and object_b.
    
    Attributes:
        self.paw (Bodypart): Bodypart representation of the paw
        self.object_a: object, which defines the Bodyaxis
        self.object_b: object, which defines the Bodyaxis
        self.parameter_array: array, that contains the calculated distance (Stance) for every frame
    """
    def __init__(self, paw: 'Bodypart', object_a: 'Bodypart', object_b: 'Bodypart')->None:
        """
        Constructor for class Stance. It calls the function to calculate the stance already.
        
        The Stance is stored in self.parameter_array
        
        Parameters:
            self.paw (Bodypart): Bodypart representation of the paw
            self.object_a: object, which defines the Bodyaxis
            self.object_b: object, which defines the Bodyaxis
        """
        self.paw = paw
        self.object_a = object_a
        self.object_b = object_b
        self.parameter_array = self._shortest_distance_between_point_and_line()
    
    def _shortest_distance_between_point_and_line(self)->float:
        """
        Function to calculate the 
        
        First, the vector between the paw and one point on the bodyaxis is creeated as m0m1. Next, the direction vector of the line is defined.
        Last step: the normalized cross product between m0m1 and the bodyaxis vector divided by the normalized bodyaxis vector results in the shortest distance between the paw and the bodyaxis. https://onlinemschool.com/math/library/analytic_geometry/p_line/
        
        Returns:
            distance(float): shortest distance between the given paw and the bodyaxis.
        """
        m0m1 = np.array([self.paw.df['x'] - self.object_a.df['x'], self.paw.df['y'] - self.object_a.df['y'], self.paw.df['z'] - self.object_a.df['z']]).reshape(self.paw.df.shape[0], 3)
        line = np.array([self.object_a.df['x']-self.object_b.df['x'], self.object_a.df['y']-self.object_b.df['y'], self.object_a.df['z']-self.object_b.df['z']]).reshape(self.paw.df.shape[0], 3)
        distance = norm(np.cross(m0m1, line), axis=1, check_finite = False) / norm(line, axis=1, check_finite = False)
        return distance
    
        
class Bodypart():
    """
    Class that contains information for one single Bodypart.
    
    Attributes:
        self.id(str): Deeplabcut label of the bodypart
    """
    def __init__(self, bodypart_id: str, df: pd.DataFrame)->None:
        """
        Constructor for class Bodypart
        """
        self.id = bodypart_id
        self._get_sliced_df(df = df)
        
        
    def _get_sliced_df(self, df: pd.DataFrame)->None:
        """
        Function, that extracts the coordinates of a single bodypart.
        
        Parameters:
            df(pandas.DataFrame): the full dataframe of the recording with all bodyparts
        """
        self.df_raw = pd.DataFrame(data={'x': df.loc[:, self.id + '_x'], 'y': df.loc[:, self.id + '_y'], 'z': df.loc[:, self.id + '_z'], 'error': df.loc[:, self.id + '_error']})
    
        
    def normalize_df(self, translation_vector: np.array, rotation_matrix: Rotation, conversion_factor: float)->None:
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
        rotated_df = self._rotate_df(rotation_matrix=rotation_matrix, df=translated_df)
        self.df = self._convert_df_to_cm(conversion_factor=conversion_factor, df=rotated_df)
    
    
    def _identify_duplicates(self)->None:
        # not necessary, since we don't have duplicates in the DLC Tracking
        pass
    
    def _translate_df(self, translation_vector: np.array)->pd.DataFrame:
        """
        Function that translates the raw dataframe to the null space.
        
        Parameter:
            translation_vector(np.Array): vector with offset of xyz to XYZ in each dimension
        Returns:
            translated_df(pandas.DataFrame): the dataframe translated to (0, 0, 0)
        """
        translated_df = self.df_raw.loc[:, ('x', 'y', 'z')] + translation_vector
        return translated_df
    
    def _rotate_df(self, rotation_matrix: Rotation, df: pd.DataFrame)->pd.DataFrame:
        """
        Function, that applies the rotation matrix to the dataframe.
        
        Besides calculating the coordinates, the error is added to the Dataframe.
        
        Parameter:
            rotation_matrix(scipy.spatial.transform.Rotation): Rotation matrix obtained from Euler angles
            df(pandas.DataFrame): the dataframe that will be rotated.
        Returns:
            rotated_df(pandas.DataFrame): the rotated dataframe
        """
        rotated_array = rotation_matrix.apply(df.loc[:, ('x', 'y', 'z')].values, inverse=True)
        rotated_df = pd.DataFrame(rotated_array, columns=['x', 'y', 'z'])
        rotated_df.loc[:, ('x', 'z')] *= -1
        rotated_df['error']=self.df_raw['error']
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
        df.loc[:, ('x', 'y', 'z')]*=conversion_factor
        return df
        
    def run_basic_operations(self, recorded_framerate: int)->None:
        """
        Function that calculates Speed and Immobility.
        """
        self._get_speed(recorded_framerate = recorded_framerate)
        self._get_rolling_speed()
        self._get_immobility()
        
    def _exclude_frames(self)->None:
        # check for reprojection error (checking for outliers should already be done before triangulation)
        pass
    
    def check_tracking_stability(self, start_end_index: Optional[Tuple]=(0, None))->float:
        """
        Function, that calculates the percentage of frames, in which the marker was detected.
        
        Parameters:
            start_end_index: range in which the percentage of detected labels should be calculated. If no values are passed, the percentage over the total session is returned.
        
        Returns:
            marker_detected_per_total_frames(float)
        """
        marker_detected_per_total_frames = self.df.loc[start_end_index[0]:start_end_index[1], 'x'].count()/self.df.loc[start_end_index[0]:start_end_index[1], :].shape[0]
        return marker_detected_per_total_frames
    
    def _get_speed(self, recorded_framerate: int)->None:
        """
        Function, that calculates the speed of the bodypart, based on the framerate.
        
        After creating an empty column with np.NaN values, the speed is calculated 
        as the squareroot of the squared difference between two frames in -x, -y and -z dimension divided by the duration of a frame.
        
        Parameters:
            recorded_framerate(int): fps of the recording
        """
        self.df.loc[:, 'speed_cm_per_s'] = np.NaN
        self.df.loc[:, 'speed_cm_per_s'] = (np.sqrt(self.df.loc[:, 'x'].diff()**2 + self.df.loc[:, 'y'].diff()**2 + self.df.loc[:, 'z'].diff()**2)) / (1/recorded_framerate)        
    
    
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
    
    def _get_immobility(self)->None:
        """
        Function, that checks frame by frame, whether the rolling_speed of the bodypart is below the immobility threshold.
        """
        self.df.loc[:, 'immobility'] = False
        self.df.loc[self.df['rolling_speed_cm_per_s'] < self.immobility_threshold, 'immobility'] = True     

    def _detect_steps(self)->None:
        """
        Detection of speed peaks based on scipy find_peaks.
        """
        speed = self.df["speed_cm_per_s"].copy()
        #Data smoothening:
        #x = np.arange(0, len(speed))
        #speed = np.nan_to_num(speed, copy=True)
        #spline = interpolate.UnivariateSpline(x, speed, s=1)
        #speed = spline(x)

        peaks = find_peaks(speed, prominence=50)
        steps_per_paw = self._create_steps(steps=peaks[0])
        return steps_per_paw
            
    def _create_steps(self, steps: List)->List['Step']:#as Class Step is not defined yet
        """
        Function, that creates Step objects for every speed peak inside of a gait event.
        
        Parameters:
            List with start_indices for steps.
            
        Returns:
            List with Step elements.
        """
        return [Step(paw = self.id, start_index = step_index) for step_index in steps]
        
                
   

                    
class EventBout():
    """
    Class, that contains start_index, end_index, duration and position of an event.
    
    Attributes:
        self.start_index(int): index of event onset
        self.end_index(int): index of event ending
    """
    def __init__(self, start_index: int, end_index: Optional[int]=0)->None:
        """
        Constructor of class EventBout that sets the attributes start_ and end_index.
        
        Parameters: 
            start_index(int): index of event onset
            end_index(Optional[int]): index of event ending (if event is not only a single frame)
        """
        self.start_index = start_index
        if end_index!=0:
            self.end_index = end_index
        else:
            self.end_index = start_index
        self._create_dict()

    @property
    def freezing_threshold(self) -> float:
        """ Arbitrary chosen threshold in seconds to check for freezing."""
        return 2.

    def check_direction(self, facing_towards_open_end: pd.Series)->None:
        """ 
        Function, that checks the direction of the mouse at the start_index.
        
        Parameters:
            facing_towards_open_end(pandas.Series): Series with boolean values for each frame.
        """
        self.facing_towards_open_end = facing_towards_open_end.iloc[self.start_index]
        self.dict['facing_towards_open_end']=self.facing_towards_open_end

    def check_that_freezing_threshold_was_reached(self, recorded_framerate: int)->None:
        """
        Function, that calculates the duration of an event and checks, whether it exceeded the freezing_threshold.
        
        Parameters:
            recorded_framerate(int): fps of the recording
        """
        self.duration = (self.end_index - self.start_index)/recorded_framerate
        self.freezing_threshold_reached = False
        if self.duration > self.freezing_threshold:
            self.freezing_threshold_reached = True
        self.dict['freezing_threshold_reached']=self.freezing_threshold_reached


    def get_position(self, centerofgravity: Bodypart)->None:
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
        
        
        
class Angle():
    """
    Class that creates an object for an Angle over a Recording between instances.
    """
    def __init__(self, bodypart_a: Bodypart, bodypart_b: Bodypart, object_to_calculate_angle: Union[np.array, Bodypart])->None:
        """
        Constructor for class Angle.
        
        Depending on the input type of object_to_calculate_angle this class contains functions to calculate the angle of 3 bodyparts to each other at the first given bodypart (bodypart_a)
        if object_to_calculate_angle is type Bodyparr or the angle of 2 bodyparts on a line to a plane if object_to_calculate_angle is a np.array with the plane in coordinate_form.
        
        Parameters:
            bodypart_a(Bodypart): bodypart, at which the angle is calculated.
            bodypart_b(Bodypart): second bodypart, necessary for defining a line
            object_to_calculate_angle(Union): plane as np.array or bodypart necessary for calculating the angle of the line between it and bodypart_a and _b
        
        """
        self.bodypart_a = bodypart_a
        self.bodypart_b = bodypart_b
        if type(object_to_calculate_angle)==Bodypart:
            self.bodypart_c = object_to_calculate_angle
            self.parameter_array = self._calculate_angle_between_three_bodyparts()
        elif type(object_to_calculate_angle)==np.array:
            self.plane = object_to_calculate_angle
            self._calculate_angle_between_bodypart_and_plane()
        
    def _calculate_angle_between_three_bodyparts(self)->np.array:
        """
        Calculates angle at bodypart_a.
        
        After calculating the length of the sides of a triangle ABC, the angles are calculated using law of cosines.
        
        Returns:
            angle(np.array): angle over the whole recording stored in an numpy.Array
        """
        length_a = self._get_length_in_3d_space(self.bodypart_b, self.bodypart_c)
        length_b = self._get_length_in_3d_space(self.bodypart_a, self.bodypart_c)
        length_c = self._get_length_in_3d_space(self.bodypart_a, self.bodypart_b)
        return self._get_angle_from_law_of_cosines(length_a, length_b, length_c)
    
    def _get_length_in_3d_space(self, object_a: Bodypart, object_b: Bodypart) -> np.array:
        """
        Calculates the length between two objects in 3D. 
        
        Parameters:
            object_a(Bodypart)
            object_b(Bodypart)
            
        Returns:
            length(np.array): Length between two objects over the whole recording stored as an numpy.Array.
        """
        if hasattr(object_a, 'df'):       
            length = np.sqrt((object_a.df['x']-object_b.df['x'])**2 + 
                             (object_a.df['y']-object_b.df['y'])**2 + 
                             (object_a.df['z']-object_b.df['z'])**2)
        else:
            length = np.sqrt((object_a.df_raw['x']-object_b.df_raw['x'])**2 + 
                             (object_a.df_raw['y']-object_b.df_raw['y'])**2 + 
                             (object_a.df_raw['z']-object_b.df_raw['z'])**2)
        #theoretisch ist es nicht nötig, den normalisierten df zu nutzen, da der Winkel ja relativ bestimmt wird 
        #und sich daher im df zum raw_df nicht unterscheiden dürfte, dann müsste man allerdings für alle Rechnungen df benutzen, 
        #damit auch die MazeCorners zum rotieren diese Klasse callen können
            
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

    def angle_between_two_lines(self, ax, ay, bx, by, cx, cy, dx, dy):
        #calculates the angle at the intersection of two linear equations, each given by two points (a, b / c, d)
        #following the rule cos(angle)= (m1-m2)/(1+m1*m2)
        m1 = (ay - by) / (ax - bx)
        m2 = (cy - dy) / (cx - dx)
        tan = (m1 - m2) / (1 + m1 * m2)
        angle = np.degrees(np.arctan(tan))
        return angle
    