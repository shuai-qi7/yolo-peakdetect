import pandas as pd
from predict2 import Detector
import torch
import numpy as np
from scipy.interpolate import interp1d
import warnings
from traditional_method_gel import LSF, baseline_tools
from scipy.signal import find_peaks, savgol_filter

warnings.simplefilter('ignore')  # Ignore warning messages


def read_xlsx_by_columns(sample_file_path, sheet_name: str | int = 0) -> dict:
    try:
        # Read xlsx file without setting header (header=None) to retain raw data
        df = pd.read_excel(sample_file_path, sheet_name=sheet_name, header=None)

        # Initialize result dictionary
        column_data = {}

        # Iterate over each column (df.columns are column indices: 0,1,2...)
        for col in df.columns:
            # Get all data of current column (convert to list)
            col_all_data = df[col].tolist()

            # 1. Take the first element of the column as key (convert to string + remove spaces to avoid type/space issues)
            key = str(col_all_data[0]).strip() if len(col_all_data) > 0 else ""
            # Skip empty keys (columns with empty first cell)
            if key == "" or key == "nan":
                continue

            # 2. Take all data from the second element of the column as value list
            value = col_all_data[1:] if len(col_all_data) > 1 else []

            value = [item for item in value if not (isinstance(item, float) and np.isnan(item))]

            # 3. Store in dictionary
            column_data[key] = value

        return column_data

    except FileNotFoundError:
        # Fix original bug: change undefined file_path to self.sample_file_path
        print(f"Error: File {sample_file_path} not found, please check if the path is correct")
        return {}
    except KeyError:
        print(f"Error: Worksheet {sheet_name} does not exist, please check worksheet name/index")
        return {}
    except Exception as e:
        print(f"Failed to read file: {str(e)}")
        return {}


def read_xlsx_by_rows(annot_records_path, sheet_name: str | int = 0) -> list:
    try:
        # Read xlsx file without setting header (header=None) to retain raw data
        df = pd.read_excel(annot_records_path, sheet_name=sheet_name, header=None)
        row_data = df.iloc[1:].values.tolist()
        return row_data

    except FileNotFoundError:
        print(f"Error: File {self.sample_file_path} not found, please check if the path is correct")
        return []
    except KeyError:
        print(f"Error: Worksheet {sheet_name} does not exist, please check worksheet name/index")
        return []
    except Exception as e:
        print(f"Failed to read file: {str(e)}")
        return []


class Neural_Test_Exp_Dataset:
    def __init__(self, sample_file_path, annot_records_path):
        self.sample_file_path = sample_file_path
        self.annot_records_path = annot_records_path
        self.detector = Detector(show_results=False)

    @staticmethod
    def adjust_sequence(sequence, max_value=0.85):
        x_original = np.arange(len(sequence))
        # Calculate x-axis scaling factor (scale sequence to length 1024)
        k_x = 1024 / len(sequence)
        # Target x coordinates (1024 points)
        x_new = np.linspace(0, len(sequence) - 1, 1024)
        # Use linear interpolation to adjust sequence to fixed length (1024)
        f_interp = interp1d(x_original, sequence, kind='linear', fill_value="extrapolate")
        new_data = f_interp(x_new)
        width_max = np.max(new_data)
        k_y = max_value
        sequence = new_data / width_max * k_y
        return sequence, k_x, k_y, width_max

    def __call__(self):
        sample_data = read_xlsx_by_columns(self.sample_file_path, "data")
        annot_data = read_xlsx_by_rows(self.annot_records_path, "records_data")
        peak_iou_all = []
        iou_limit = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        predict_correct_area_num = 0  # Number of correctly predicted overlapping peak areas
        predict_correct_num_num = 0  # Number of correctly predicted overlapping peak counts
        for i in range(len(sample_data)):
            print(i + 1)
            sequence, k_x, k_y, width_max = self.adjust_sequence(sample_data[f"sample{i + 1}"])
            sequence = torch.tensor(sequence).float().unsqueeze(0).unsqueeze(0)
            target = torch.zeros(1, 16, 4)
            output_left_edges, output_right_edges, output_locations, output_amplitudes, output_areas = self.detector(
                sequence, target, max_value=0.85, k_x=k_x, k_y=k_y, width_max=width_max)
            for j in range(len(annot_data)):
                if annot_data[j][0] == i + 1:
                    if annot_data[j][1] == 1:
                        left_edge, right_edge, location, height = annot_data[j][2:6]
                        for k in range(len(output_locations)):
                            if left_edge < output_locations[k] < right_edge:
                                in_start = max(left_edge, output_left_edges[k])
                                in_end = min(right_edge, output_right_edges[k])
                                in_height = min(height, output_amplitudes[k])
                                in_length = max(in_end - in_start, 0)
                                in_area = in_height * in_length
                                area = (right_edge - left_edge) * height
                                output_area = (output_right_edges[k] - output_left_edges[k]) * output_amplitudes[k]
                                iou = in_area / (area + output_area - in_area)
                                un_start = min(left_edge, output_left_edges[k])
                                un_end = max(right_edge, output_right_edges[k])
                                peak_iou = iou - abs(output_locations[k] - location) / (un_end - un_start)
                                peak_iou_all.append(peak_iou)
                    if annot_data[j][1] == 2:
                        left_edge, right_edge = annot_data[j][2:4]
                        area, overlap_num = annot_data[j][6:8]
                        output_overlap_num = 0
                        output_overlap_area = 0
                        for k in range(len(output_locations)):
                            if left_edge < output_locations[k] < right_edge:
                                output_overlap_num += 1
                                output_overlap_area += output_areas[k]
                        if abs(output_overlap_area - area) / area < 0.25:
                            predict_correct_area_num += 1
                        if output_overlap_num == overlap_num:
                            predict_correct_num_num += 1

        pre_all = np.array([len([p for p in peak_iou_all if p >= threshold]) for threshold in iou_limit]) / np.sum(
            np.array(annot_data)[:, 1] == 1)
        rec_all = np.array([len([p for p in peak_iou_all if p >= threshold]) for threshold in iou_limit]) / len(
            peak_iou_all)
        ap = [pre_all[i] * rec_all[i] for i in range(len(pre_all))]
        print(ap)
        print(np.mean(ap))
        pre_overlap_area = predict_correct_area_num / np.sum(np.array(annot_data)[:, 1] == 2)
        pre_overlap_num = predict_correct_num_num / np.sum(np.array(annot_data)[:, 1] == 2)
        print(pre_overlap_area, pre_overlap_num)


class Traditional_Test_Exp_Dataset:
    def __init__(self, sample_file_path, annot_records_path):
        self.sample_file_path = sample_file_path
        self.annot_records_path = annot_records_path
        self.lsf = LSF()

    @staticmethod
    def adjust_sequence(sequence, max_value=1.0):
        width_max = np.max(sequence)
        sequence = sequence / width_max
        return sequence, width_max

    def __call__(self, ):
        sample_data = read_xlsx_by_columns(self.sample_file_path, "data")
        annot_data = read_xlsx_by_rows(self.annot_records_path, "records_data")
        peak_iou_all = []
        overlap_area_radio_all = []
        overlap_correct_num_all = 0
        iou_limit = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        window_length_list = [5, 9, 15]
        polyorder_list = [2, 3, 4]
        height_list = [0.01, 0.05, 0.1]
        prominence_list = [0.01, 0.05, 0.1]
        width_list = [3, 5, 8]
        best_parameters_all = []
        for i in range(len(sample_data)):
            print(i + 1)
            sequence, width_max = self.adjust_sequence(sample_data[f"sample{i + 1}"])
            inputs_without_baseline = baseline_tools(sequence)
            best_peak_iou = []
            best_overlap_area_radio = []
            best_overlap_correct_num = 0
            best_parameters = []
            for window_length in window_length_list:
                for polyorder in polyorder_list:
                    for height in height_list:
                        for prominence in prominence_list:
                            for width in width_list:
                                peak_iou_list = []
                                overlap_area_radio_list = []
                                overlap_correct_num = 0
                                parameter_list = [window_length, polyorder, height, prominence, width]
                                smoothed_sequence = savgol_filter(inputs_without_baseline, window_length=window_length,
                                                                  polyorder=polyorder, deriv=0)
                                self.lsf = LSF(height=height, prominence=prominence, width=width)
                                output_left_edges, output_right_edges, output_locations, output_amplitudes, output_areas = self.lsf(
                                    smoothed_sequence, width_max)
                                for j in range(len(annot_data)):
                                    if annot_data[j][0] == i + 1:
                                        if annot_data[j][1] == 1:
                                            left_edge, right_edge, location, height = annot_data[j][2:6]
                                            for k in range(len(output_locations)):
                                                if left_edge < output_locations[k] < right_edge:
                                                    # print(left_edge,right_edge,location,height)
                                                    # print(output_left_edges[k],output_right_edges[k],output_locations[k],output_amplitudes[k])
                                                    in_start = max(left_edge, output_left_edges[k])
                                                    in_end = min(right_edge, output_right_edges[k])
                                                    in_height = min(height, output_amplitudes[k])
                                                    in_length = max(in_end - in_start, 0)
                                                    in_area = in_height * in_length
                                                    area = (right_edge - left_edge) * height
                                                    output_area = (output_right_edges[k] - output_left_edges[k]) * \
                                                                  output_amplitudes[k]
                                                    iou = in_area / (area + output_area - in_area)
                                                    un_start = min(left_edge, output_left_edges[k])
                                                    un_end = max(right_edge, output_right_edges[k])
                                                    peak_iou = iou - abs(output_locations[k] - location) / (
                                                            un_end - un_start)
                                                    peak_iou_list.append(peak_iou)
                                        if annot_data[j][1] == 2:
                                            left_edge, right_edge = annot_data[j][2:4]
                                            area, overlap_num = annot_data[j][6:8]
                                            output_overlap_num = 0
                                            output_overlap_area = 0
                                            for k in range(len(output_locations)):
                                                if left_edge < output_locations[k] < right_edge:
                                                    output_overlap_num += 1
                                                    output_overlap_area += output_areas[k]
                                            overlap_area_radio_list.append(abs(output_overlap_area - area) / area)
                                            if output_overlap_num == overlap_num:
                                                overlap_correct_num += 1
                                mean_peak_iou_list = 0 if peak_iou_list == [] else np.mean(peak_iou_list)
                                mean_overlap_area_radio_list = 100 if overlap_area_radio_list == [] else np.mean(
                                    overlap_area_radio_list)
                                mean_best_peak_iou = 0 if best_peak_iou == [] else np.mean(best_peak_iou)
                                mean_best_overlap_area_radio = 100 if best_overlap_area_radio == [] else np.mean(
                                    best_overlap_area_radio)
                                if mean_peak_iou_list - mean_overlap_area_radio_list > mean_best_peak_iou - mean_best_overlap_area_radio:
                                    best_peak_iou = peak_iou_list
                                    best_overlap_area_radio = overlap_area_radio_list
                                    best_overlap_correct_num = overlap_correct_num
                                    best_parameters = parameter_list
            peak_iou_all.extend(best_peak_iou)
            overlap_area_radio_all.extend(best_overlap_area_radio)
            overlap_correct_num_all += best_overlap_correct_num
            best_parameters_all.append(best_parameters)
        pre_all = np.array([len([p for p in peak_iou_all if p >= threshold]) for threshold in iou_limit]) / np.sum(
            np.array(annot_data)[:, 1] == 1)
        rec_all = np.array([len([p for p in peak_iou_all if p >= threshold]) for threshold in iou_limit]) / len(
            peak_iou_all)
        ap = [pre_all[i] * rec_all[i] for i in range(len(pre_all))]
        print(ap)
        print(np.mean(ap))
        pre_overlap_area = np.sum(np.array(overlap_area_radio_all) < 0.25) / np.sum(np.array(annot_data)[:, 1] == 2)
        pre_overlap_num = overlap_correct_num_all / np.sum(np.array(annot_data)[:, 1] == 2)
        print(pre_overlap_area, pre_overlap_num)


if __name__ == "__main__":
    neural_test = Neural_Test_Exp_Dataset("sample/sample_row_sums.xlsx", "sample/multi_annot_records.xlsx")
    neural_test()

    traditional_test = Traditional_Test_Exp_Dataset("sample/sample_row_sums.xlsx", "sample/multi_annot_records.xlsx")
    traditional_test()