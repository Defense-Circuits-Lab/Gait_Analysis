from typing import List, Tuple, Dict, Union, Optional
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import imageio as iio

import aniposelib as ap_lib

# ToDo:
# Should the SingleCamDataForAnipose & the CalibrationForAnipose3DTracking only be 
#   subclasses of a more general parent that could then also be used as base class
#   to process the actual experimental recordings? Here only triangulation would be
#   needed & calibration should be loaded and error estimation based on test position
#   markers would not be neccessary!


class IntrinsicCameraCalibrator(ABC):
    #https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    
    @abstractmethod 
    def _run_cam_type_specific_calibration(self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]) -> Tuple:
        # wrapper to camera type specific calibration function
        # all remaining data is stored in attributes of the object
        pass


    def __init__(self, filepath_calibration_video: Path, max_frame_count: int) -> None:
        import cv2 
        self.video_filepath = filepath_calibration_video
        self.max_frame_count = max_frame_count
        self.video_reader = iio.get_reader(filepath_calibration_video)


    @property
    def checkerboard_rows_and_columns(self) -> Tuple[int, int]:
        return (5, 5)

    @property
    def d(self) -> np.ndarray:
        return np.zeros((4, 1))

    @property
    def imsize(self) -> Tuple[int, int]:
        # Does it have to be the shape of the grayscale image?
        frame = np.asarray(self.video_reader.get_data(0))
        frame_in_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame_in_gray_scale.shape[::-1]

    @property
    def k(self) -> np.ndarray:
        return np.zeros((3, 3))

    @property
    def objp(self) -> np.ndarray:
        objp = np.zeros((1, self.checkerboard_rows_and_columns[0]*self.checkerboard_rows_and_columns[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:self.checkerboard_rows_and_columns[0], 0:self.checkerboard_rows_and_columns[1]].T.reshape(-1, 2)
        return objp

    @property
    def subpixel_criteria(self) -> Tuple:
        return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)


    def run(self) -> Dict:
        selected_frame_idxs = self._get_indices_of_selected_frames()
        detected_checkerboard_corners_per_image = self._detect_checkerboard_corners(frame_idxs = selected_frame_idxs)
        if len(detected_checkerboard_corners_per_image) != self.max_frame_count:
            detected_checkerboard_corners_per_image = self._attempt_to_match_max_frame_count(corners_per_image = detected_checkerboard_corners_per_image,
                                                                                             already_selected_frame_idxs = selected_frame_idxs)
        object_points = self._compute_object_points(n_detected_boards = len(detected_checkerboard_corners_per_image))
        retval, K, D, rvec, tvec = self._run_camera_type_specific_calibration(objpoints = object_points, imgpoints = detected_checkerboard_corners_per_image)
        calibration_results = self._construct_calibration_results(K = K, D = D, rvec = rvec, tvec = tvec)
        return calibration_results


    def save(self) -> None:
        video_filename = self.video_filepath.name
        filename = f'{video_filename[:video_filename.rfind(".")]}_intrinsic_calibration_results.p'
        with open(self.video_filepath.parent.joinpath(filename), 'wb') as io:
            pickle.dump(self.calibration_results, io)


    def _attempt_to_match_max_frame_count(self, corners_per_image: List[np.ndarray], already_selected_frame_idxs: List[int]) -> List[np.ndarray]:
        if len(corners_per_image) < self.max_frame_count:
            corners_per_image = self._attempt_to_reach_max_frame_count(corners_per_image = corners_per_image, 
                                                                       already_selected_frame_idxs = already_selected_frame_idxs)
        elif len(corners_per_image) > self.max_frame_count:
            corners_per_image = self._limit_to_max_frame_count(all_detected_corners = corners_per_image)
        return corners_per_image


    def _attempt_to_reach_max_frame_count(self, corners_per_image: List[np.ndarray], already_selected_frame_idxs: List[int]) -> List[np.ndarray]:
        # ToDo
        # limit time?
        total_frame_count = self.video_reader.count_frames()
        for idx in range(total_frame_count):
            if len(corners_per_image) < self.max_frame_count:
                if idx not in already_selected_frame_idxs:
                    checkerboard_detected, predicted_corners = self._run_checkerboard_corner_detection(idx = idx)
                    if checkerboard_detected:
                        corners_per_image.append(predicted_corners)
            else:
                break
        return corners_per_image


    def _construct_calibration_results(self, K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> Dict:
        # ToDo: 
        # confirm type hints
        # Potentially add more parameters that might be required for adjusting this intrinsic
        #    calibration to cropping or flipping of the actual recordings (both calibration and experiment)
        calibration_results = {"K": K, "D": D, "rvec": rvec, "tvec": tvec, "size": self.imsize}
        setattr(self, 'calibration_results', calibration_results)
        return calibration_results


    def _compute_object_points(self, n_detected_boards: int) -> List[np.ndarray]:
        object_points = list(self.objp)*n_detected_boards
        return object_points


    def _detect_checkerboard_corners(self, frame_idxs: List[int]) -> List[np.ndarray]:
        detected_checkerboard_corners_per_image = []
        for idx in frame_idxs:
            checkerboard_detected, predicted_corners = self._run_checkerboard_corner_detection(idx = idx)
            if checkerboard_detected:
                detected_checkerboard_corners_per_image.append(predicted_corners)
        return detected_checkerboard_corners_per_image


    def _determine_sampling_rate(self) -> int:
        total_frame_count = self.video_reader.count_frames()
        if total_frame_count >= 5*self.max_frame_count:
            sampling_rate = total_frame_count // (5*self.max_frame_count)
        else:
            sampling_rate = 1
        return sampling_rate


    def _get_indices_of_selected_frames(self) -> List[int]:
        sampling_rate = self._determine_sampling_rate()
        frame_idxs = self._get_sampling_frame_indices(sampling_rate = sampling_rate)
        return list(frame_idxs)


    def _get_sampling_frame_indices(self, sampling_rate: int) -> np.ndarray:
        total_frame_count = self.video_reader.count_frames()
        n_frames_to_select = total_frame_count // sampling_rate
        return np.linspace(0, total_frame_count, n_frames_to_select, endpoint=False, dtype=int)


    def _limit_to_max_frame_count(self, all_detected_corners: List[np.ndarray]) -> List[np.ndarray]:
        sampling_idxs = np.linspace(0, len(all_detected_corners), endpoint=False, dtype=int)
        return all_detected_corners[sampling_idxs]


    def _run_checkerboard_corner_detection(self, idx: int) -> Tuple[bool, np.ndarray]:
        # The following line was in earlier version - still required?
        # image_shape = image.shape[:2]
        image = np.asarray(self.video_reader.get_data(idx))
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        checkerboard_detected, predicted_corners = cv2.findChessboardCorners(gray_scale_image, 
                                                                             self.checkerboard_rows_and_columns, 
                                                                             cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if checkerboard_detected:
            predicted_corners = cv2.cornerSubPix(gray_scale_image, predicted_corners, (3,3), (-1,-1), self.subpix_criteria)
        return checkerboard_detected, predicted_corners

        
        

class IntrinsicCalibratorFisheyeCamera(IntrinsicCameraCalibrator):
          
    def _compute_rvecs_and_tvecs(self, n_detected_boards: int) -> Tuple[np.ndarray, np.ndarray]:
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n_detected_boards)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n_detected_boards)]
        return rvecs, tvecs


    def _run_cam_type_specific_calibration(self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]) -> Tuple:
        rvecs, tvecs = self._compute_rvecs_and_tvecs(n_detected_boards = len(objpoints))
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        new_subpixel_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        return cv2.fisheye.calibrate(objpoints, imgpoints, self.imsize, self.k, self.d, rvecs, tvecs, calibration_flags, new_subpixel_criteria)      
    
    
        
class IntrinsicCalibratorRegularCamera(IntrinsicCameraCalibrator):
    
    def _run_cam_type_specific_calibration(self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]) -> Tuple:
        return cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)        





class TestPositionsGroundTruth:
    
    def __init__(self) -> None:
        self.marker_ids_with_distances = {}
        self.unique_marker_ids = []
        self._add_maze_corners()

    # ToDo:
    # remove marker id?
    
    def add_new_marker_id(self, marker_id: str, other_marker_ids_with_distances: List[Tuple[str, Union[int, float]]]) -> None:
        for other_marker_id, distance in other_marker_ids_with_distances:
            self._add_ground_truth_information(marker_id_a = marker_id, marker_id_b = other_marker_id, distance = distance)
            self._add_ground_truth_information(marker_id_a = other_marker_id, marker_id_b = marker_id, distance = distance)
            
    
    def add_marker_ids_to_be_connected_in_3d_plots(self, marker_ids: Tuple[str]) -> None:
        # ToDo
        # build scheme
        # set scheme as attribute


    def load_from_disk(self, filepath: Path) -> None:
        with open(filepath, 'rb') as io:
            marker_ids_with_distances = pickle.load(io)
        unique_marker_ids = list(marker_ids_with_distances.keys())
        setattr(self, 'marker_ids_with_distances', marker_ids_with_distances)
        setattr(self. 'unique_marker_ids', unique_marker_ids)


    def save_to_disk(self, filepath: Path) -> None:
        # ToDo: validate filepath, -name, -extension & provide default alternative
        with open(filepath, 'wb') as io:
            pickle.dump(self.marker_ids_with_distances, io)   

    
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



class SingleCamDataForAnipose:
    
    def __init__(self, cam_id: str, filepath_synchronized_calibration_video: Path) -> None:
        self.cam_id = cam_id
        self.filepath_synchronized_calibration_video = filepath_synchronized_calibration_video


    def add_cropping_offsets(self, x_or_column_offset: int=0, y_or_row_offset: int=0) -> None:
        setattr(self, 'cropping_offsets', (x_or_column_offset, y_or_row_offset))
    
    
    def add_flipping_details(self, flipped_horizontally: bool=False, flipped_vertically: bool=False) -> None:
        setattr(self, 'flipped_horizontally', flipped_horizontally)
        setattr(self, 'flipped_vertically', flipped_vertically)

        
    def add_manual_test_position_marker(self, marker_id: str, x_or_column_idx: int, y_or_row_idx: int, likelihood: float, overwrite: bool=False) -> None:
        if hasattr(self, 'manual_test_position_marker_coords_pred') == False:
            self.manual_test_position_marker_coords_pred = {}
        if (marker_id in self.manual_test_position_marker_coords_pred.keys()) & (overwrite == False):
            raise ValueError('There are already coordinates for the marker you '
                             f'tried to add: "{marker_id}: {self.manual_test_position_marker_coords_pred[marker_id]}'
                             '". If you would like to overwrite these coordinates, please pass '
                             '"overwrite = True" as additional argument to this method!')
        self.manual_test_position_marker_coords_pred[marker_id] = {'x': [x_or_column_idx], 'y': [y_or_row_idx], 'likelihood': [likelihood]}


    def export_as_aniposelib_Camera_object(self) -> ap_lib.cameras.Camera:
        camera = ap_lib.cameras.Camera(name = self.cam_id,
                                       size = self.intrinsic_calibration_for_anipose['size'], # of original intrinsic calibration video or of cropped? cropped!
                                       rvec = self.intrinsic_calibration_for_anipose['rvec'], # optional since extrinsic
                                       tvec = self.intrinsic_calibration_for_anipose['tvec'], # optional since extrinsic
                                       matrix = self.intrinsic_calibration_for_anipose['K'],
                                       dist = self.intrinsic_calibration_for_anipose['D'],
                                       extra_dist = False)
        return camera


    def load_intrinsic_camera_calibration(self, filepath_intrinsic_calibration: Path) -> None:
        with open(filepath_intrinsic_calibration, 'rb') as io:
            intrinsic_calibration = pickle.load(io)
        adjusting_required = self._is_adjusting_of_intrinsic_calibration_required(unadjusted_intrinsic_calibration = intrinsic_calibration)
        self._set_intrinsic_calibration(intrinsic_calibration = intrinsic_calibration, adjusting_required = adjusting_required)
        
    
    def load_test_position_markers_df_from_dlc_prediction(self, filepath_deeplabcut_prediction: Path) -> None:
        df = pd.read_hdf(filepath_deeplabcut_prediction)
        setattr(self, 'test_position_markers_df', df)
        setattr(self, 'filepath_test_position_marker_prediction', filepath_deeplabcut_prediction)
 
    
    def run_intrinsic_camera_calibration(self, filepath_checkerboard_video: Path, fisheye_cam: bool, save: bool=True, max_frame_count: int=300) -> None:
        if fisheye_cam:
            calibrator = IntrinsicCalibratorFisheyeCamera(filepath_calibration_video = filepath_checkerboard_video, max_frame_count = max_frame_count)
        else:
            calibrator = IntrinsicCalibratorRegularCamera(filepath_calibration_video = filepath_checkerboard_video, max_frame_count = max_frame_count)
        intrinsic_calibration = calibrator.run()
        if save:
            calibrator.save()
        adjusting_required = self._is_adjusting_of_intrinsic_calibration_required(unadjusted_intrinsic_calibration = intrinsic_calibration)
        self._set_intrinsic_calibration(intrinsic_calibration = intrinsic_calibration, adjusting_required = adjusting_required)

                
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


    def validate_test_position_marker_ids(self, test_positions_gt: TestPositionsGroundTruth, add_missing_marker_ids_with_0_likelihood: bool=True) -> None:
        if hasattr(self, 'test_position_markers_df') == False:
            raise ValueError('There was no DLC prediction of the test position markers loaded yet. '
                             'Please load it using the ".load_test_position_markers_df_from_dlc_prediction()" '
                             'method on this object (if you have DLC predictions to load) - or first add '
                             'the positions manually using the ".add_manual_test_position_marker()" method '
                             'on this object, and eventually load these data after adding all marker_ids '
                             'that you could identify via the ".save_manual_marker_coords_as_fake_dlc_output() '
                             'method on this object.')
        ground_truth_marker_ids = test_positions_gt.unique_marker_ids.copy()
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


    def _adjust_intrinsic_calibration(self, unadjusted_intrinsic_calibration: Dict) -> Dict:
        adjusted_intrinsic_calibration = unadjusted_intrinsic_calibration.copy()
        # is the following the correct size? current "size" value was determined on grayscale image
        intrinsic_calibration_video_size = unadjusted_intrinsic_calibration['size']
        new_video_size = self._get_anipose_calibration_video_size()
        x_offset, y_offset = self._get_correct_x_y_offsets(intrinsic_calibration_video_size = intrinsic_calibration_video_size, new_video_size = new_video_size)
        adjusted_K = self._get_adjusted_K(K = unadjusted_intrinsic_calibration['K'], x_offset = x_offset, y_offset = y_offset)
        adjusted_intrinsic_calibration = self._incorporate_adjustments_in_intrinsic_calibration(intrinsic_calibration = unadjusted_intrinsic_calibration.copy(),
                                                                                                new_size = new_video_size,
                                                                                                adjusted_K = adjusted_K)
        return adjusted_intrinsic_calibration


    def _construct_dlc_output_style_df_from_manual_marker_coords(self) -> pd.DataFrame:
        multi_index = self._get_multi_index()
        df = pd.DataFrame(data = {}, columns = multi_index)
        for scorer, marker_id, key in df.columns:
            df[(scorer, marker_id, key)] = self.manual_test_position_marker_coords_pred[marker_id][key]
        return df


    def _find_non_matching_marker_ids(self, marker_ids_to_match: List[str], template_marker_ids: List[str]) -> List:
        return [marker_id for marker_id in marker_ids_to_match if marker_id not in template_marker_ids]


    def _get_adjusted_K(self, K: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
        adjusted_K = K.copy()
        adjusted_K[0][2] = adjusted_K[0][2] - x_offset
        adjusted_K[1][2] = adjusted_K[1][2] - y_offset
        return adjusted_K  
    
    
    def _get_anipose_calibration_video_size(self) -> Tuple[int, int]:
        video_reader = iio.get_reader(self.filepath_synchronized_calibration_video)
        return video_reader.get_metadata()['size']
    
    
    def _get_correct_x_y_offsets(self, intrinsic_calibration_video_size: Tuple[int, int], new_video_size: Tuple[int, int]) -> Tuple[int, int]:
        # ToDo: 
        # determine which cropping offsets are the correct ones to compute, probably in dependency of flipping & rotation settings?
        if (not self.flipped_horizontally) and (not self.flipped_vertically):
            x_offset, y_offset = self.cropping_offsets
        # alternatives might be something like:
        # x_offset = intrinsic_calibration_video_size[0] - new_video_size[0] - self.cropping_offsets[0]
        # y_offset = intrinsic_calibration_video_size[1] - new_video_size[1] - self.cropping_offsets[1]
        return x_offset, y_offset

    
    def _get_multi_index(self) -> pd.MultiIndex:
        multi_index_column_names = [[], [], []]
        for marker_id in self.manual_test_position_marker_coords_pred.keys():
            for column_name in ("x", "y", "likelihood"):
                multi_index_column_names[0].append("manually_annotated_marker_positions")
                multi_index_column_names[1].append(marker_id)
                multi_index_column_names[2].append(column_name)
        return pd.MultiIndex.from_arrays(multi_index_column_names, names=('scorer', 'bodyparts', 'coords'))


    def _incorporate_adjustments_in_intrinsic_calibration(intrinsic_calibration: Dict, new_size: Tuple[int, int], adjusted_K: np.ndarray) -> Dict:
        intrinsic_calibration['size'] = new_size
        intrinsic_calibration['K'] = adjusted_K
        return intrinsic_calibration


    def _is_adjusting_of_intrinsic_calibration_required(self, unadjusted_intrinsic_calibration: Dict) -> bool:
        adjusting_required = False
        # ToDo:
        # add check, whether adjustment is required (i.e. if size of 
        # intrinsic calibration and size of anipose calibration
        # video are not matching, if cropping offsetts or if
        # flipping infos are passed & relevant)
        if any([self.cropping_offsets != (0, 0), self.flipped_horizontally, self.flipped_vertically]):
            adjusting_required = True
        return adjusting_required


    def _remove_marker_ids_not_in_ground_truth(self, marker_ids_to_remove: List[str]) -> None:
        df = self.test_position_markers_df
        columns_to_remove = [column_name for column_name in df.columns if column_name[1] in marker_ids_to_remove]
        df.drop(columns = columns_to_remove, inplace=True)

    
    def _set_intrinsic_calibration(self, intrinsic_calibration: Dict, adjusting_required: bool) -> None:
        if adjusting_required:
            intrinsic_calibration = self._adjust_intrinsic_calibration(unadjusted_intrinsic_calibration = intrinsic_calibration)
        setattr(self, 'intrinsic_calibration_for_anipose', intrinsic_calibration)
        
        
        
class CalibrationForAnipose3DTracking:

    def __init__(self, single_cams_to_calibrate: List[SingleCamDataForAnipose]) -> None:
        # ToDo: validate unique filepaths
        self._validate_unique_cam_ids(single_cams_to_calibrate = single_cams_to_calibrate)
        # potentially it would make sense to call the following method already here?
        # or is there a reason why you would
        # self._validate_test_position_markers_df_is_loaded_to_all_single_cam_objects()
        self.single_cam_objects = single_cams_to_calibrate
        self._get_all_calibration_video_filepaths()
        self._initialize_camera_group()


    @property
    def score_threshold(self) -> float:
        return 0.5


    def run_calibration(self, use_own_intrinsic_calibration: bool=True, charuco_calibration_board: Optional[ap_lib.boards.CharucoBoard]) -> None:
        # ToDo
        # possibility to add verbose=False in calibrate_videos() call to avoid lengthy output?
        # confirm type hinting 
        if charuco_calibration_board == None:
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            charuco_calibration_board = ap_lib.boards.CharucoBoard(7, 5, square_length=1, marker_length=0.8, marker_bits=6, aruco_dict=aruco_dict)
        self.camera_group.calibrate_videos(videos = self.calibration_video_filepaths, 
                                           board = charuco_calibration_board,
                                           init_intrinsics = not use_own_intrinsic_calibration, 
                                           init_extrinsics = True)
        
        
    def evaluate_triangulation_of_test_position_markers(self, test_positions_gt: TestPositionsGroundTruth, show_3D_plot: bool=True, verbose: bool=True) -> None:
        # ToDo
        # validate that all SingleCamDataForAnipose objects have the corresponding 'test_position_markers_df' attribute
        # run triangulation
        # compute all errors:
        #    - reprojection errors given by anipose
        #    - differences in triangulated vs. measured ground-truth distances
        # if verbose: report errors & ideally add what range of values are acceptable
        # if show_3D_plot: plot the triangulated test positions in 3D
        self._validate_test_position_markers_df_is_loaded_to_all_single_cam_objects()
        anipose_triangulation_io = self._preprocess_dlc_predictions_for_anipose()
        anipose_triangulation_io['p3ds_flat'] = self.camera_group.triangulate(anipose_triangulation_io['points_flat'], progress=True)
        anipose_triangulation_io = self._postprocess_triangulations_and_calculate_reprojection_error(anipose_triangulation_io = anipose_triangulation_io)
        # Continue here with implementation of the different test positions distance errors
        #   then add verbose outputs, like:
        # print("Mean reprojection error:", anipose_triangulation_io['reproj_nonan'].mean())
        #   and finally finish with the 3D Plot


    def save_calibration(self, filepath: Path) -> None:
        # ToDo
        # validate filepath and extension (.toml)
        # and add default alternative
        self.camera_group.dump(filepath)


    def _preprocess_dlc_predictions_for_anipose(self) -> Dict:
        fname_dict = {}
        for single_cam in self.single_cam_objects:
            fname_dict[single_cam.cam_id] = single_cam.filepath_test_position_marker_prediction
        anipose_triangulation_io = ap_lib.utils.load_pose2d_fnames(fname_dict = fname_dict)
        anipose_triangulation_io = self._add_additional_information_and_continue_preprocessing(d = anipose_triangulation_io)
        return anipose_triangulation_io


    def _postprocess_triangulations_and_calculate_reprojection_error(self, anipose_triangulation_io: Dict) -> Dict:
        anipose_triangulation_io['reprojerr_flat'] = self.camera_group.reprojection_error(anipose_triangulation_io['p3ds_flat'], 
                                                                                          anipose_triangulation_io['points_flat'],
                                                                                          mean=True)
        anipose_triangulation_io['p3ds'] = anipose_triangulation_io['p3ds_flat'].reshape(anipose_triangulation_io['n_points'], 
                                                                                         anipose_triangulation_io['n_joints'],
                                                                                         3)
        anipose_triangulation_io['reprojerr'] = anipose_triangulation_io['reprojerr_flat'].reshape(anipose_triangulation_io['n_points'],
                                                                                                   anipose_triangulation_io['n_joints'])
        anipose_triangulation_io['reproj_nonan'] = anipose_triangulation_io['reprojerr'][np.logical_not(np.isnan(anipose_triangulation_io['reprojerr']))]
        return anipose_triangulation_io
        

    def _add_additional_information_and_continue_preprocessing(self, d: Dict) -> Dict: 
        n_cams, d['n_points'], d['n_joints'], _ = d['points'].shape
        d['points'][d['scores'] < self.score_threshold] = np.nan
        d['points_flat'] = d['points'].reshape(n_cams, -1, 2)
        d['scores_flat'] = d['scores'].reshape(n_cams, -1)
        return d
        
        
    def _validate_test_position_markers_df_is_loaded_to_all_single_cam_objects(self):
        # re-run validation on single_cam object
        for single_cam in self.single_cam_objects:
            if hasattr(single_cam, 'test_position_markers_df') == False:
                raise ValueError('For this evaluation, all SingleCamDataForAnipose objects must have '
                                 'loaded the predicted coordinates of the test position marker ids. '
                                 'However, this data is missing for the SingleCamDataForAnipose object '
                                 f'with the cam_id: {single_cam.cam_id}. Please load it to this object '
                                 'by calling it´s ".load_test_position_markers_df_from_dlc_prediction()" '
                                 'method.')


    def _get_all_calibration_video_filepaths(self) -> None:
        video_filpaths = [single_cam.filepath_synchronized_calibration_video for single_cam in self.single_cam_objects]
        setattr(self, 'calibration_video_filepaths', video_filepaths)


    def _initialize_camera_group(self) -> None:
        all_Camera_objects = [single_cam.export_as_aniposelib_Camera_object() for single_cam in self.single_cam_objects]
        setattr(self, 'camera_group', ap_lib.cameras.CameraGroup(all_Camera_objects))


    def _validate_unique_cam_ids(self, single_cams_to_calibrate: List[SingleCamDataForAnipose]) -> None:
        cam_ids = []
        for single_cam in single_cams_to_calibrate:
            if single_cam.cam_id not in cam_ids:
                cam_ids.append(single_cam.cam_id)
            else:
                raise ValueError(f'You added multiple cameras with the cam_id {single_cam.cam_id}, '
                                 'however, all cam_ids must be unique! Please check for duplicates '
                                 'in the "single_cams_to_calibrate" list, or rename the respective '
                                 'cam_id attribute of the corresponding SingleCamDataForAnipose object.')