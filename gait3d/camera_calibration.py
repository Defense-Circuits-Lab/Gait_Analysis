from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path

import pandas as pd
import pickle


class SingleCamDataForAnipose:
    
    def __init__(self, cam_id: str, filepath_synchronized_calibration_video: Path) -> None:
        self.cam_id = cam_id
        self.filepath_synchronized_calibration_video = filepath_synchronized_calibration_video
        
        
    def add_manual_test_position_marker(self, marker_id: str, x_or_column_idx: int, y_or_row_idx: int, likelihood: float, overwrite: bool=False) -> None:
        if hasattr(self, test_position_marker_coords) == False:
            self.manual_test_position_marker_coords_pred = {}
        if (marker_id in self.manual_test_position_marker_coords_pred.keys()) & (overwrite == False):
            raise ValueError('There are already coordinates for the marker you '
                             f'tried to add: "{marker_id} - x: {x_or_column_idx}, y: {y_or_row_idx} ), '
                             f'likelihood: {likelihood}". If you would like to overwrite these coordinates, '
                             'please pass "overwrite = True" as additional argument to this method!')
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
        df_out.to_h5(output_filepath, "df")
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
    
    
    def add_cropping_offsets(self, x_or_column_offset: int=0, y_or_row_offset: int=0) -> None:
        setattr(self, 'cropping_offsets', (x_or_column_offset, y_or_row_offset))
    
    
    def add_flipping_details(self, flipped_horizontally: bool=False, flipped_vertically: bool=False) -> None:
        setattr(self, 'flipped_horizontally', flipped_horizontally)
        setattr(self, 'flipped_vertically', flipped_vertically)
        
    
    def run_intrinsic_camera_calibration(self, filepath_checkerboard_video: Path, save: bool=True) -> None:
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