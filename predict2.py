import os.path
from Net import *
from dataset import *
from utils import Utils
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math
import torch

warnings.simplefilter('ignore')  # Ignore warning messages

# Define color list for plotting (used to distinguish different peaks)
colors = [
    '#FF5733',  # Orange
    '#33FF57',  # Turquoise Green
    '#3357FF',  # Blue
    '#F44336',  # Red (Deeper)
    '#2196F3',  # Blue (Deeper)
    '#FFEB3B',  # Yellow
    '#4CAF50',  # Green
    '#9C27B0',  # Purple
    '#E91E63',  # Pink
    '#03DAC5',  # Cyan Blue
    '#FF9800',  # Orange Yellow
    '#795548',  # Brown
    '#607D8B',  # Slate Blue
    '#FFC107',  # Golden Yellow
    '#8BC34A',  # Light Green
    '#CDDC39',  # Yellow Green
    '#FF5252',  # Bright Red
    '#3F51B5',  # Navy Blue
    '#673AB7',  # Deep Purple
    '#00BCD4',  # Light Blue
    '#FF33E7',  # Fuchsia
    '#009688',  # Teal
    '#FF6F00',  # Orange Yellow
    '#3949AB',  # Dark Blue
    '#8E24AA',  # Deep Magenta
    '#00C853',  # Fresh Green
    '#FFD54F',  # Light Orange
    '#5D4037',  # Dark Brown
    '#B39DDB',  # Light Purple
    '#00897B',  # Teal (Blue Tint)
    '#FF1744',  # Vivid Red
    '#64DD17',  # Bright Green
    '#FF9100',  # Orange Red
    '#1E88E5',  # Sky Blue
    '#D81B60',  # Crimson
    '#00E676',  # Grass Green
    '#FFE082',  # Light Yellow
    '#424242',  # Dark Gray
    '#90CAF9',  # Pale Blue
    '#78909C',  # Silver Gray
    '#FF6699',  # Light Pink
    '#0091EA',  # Lake Blue
    '#FFD700',  # Gold
    '#689F38',  # Dark Green
    '#FF8A65',  # Salmon Pink
    '#455A64',  # Blue Black
    '#D4E157',  # Light Green
    '#FF00B8',  # Pink (Purple Tint)
    '#00796B',  # Dark Teal
    '#FF3D00'  # Orange Red (Bright)
]

def seq_to_excel(sequence, output_sequences, filename):
    # Minimal length check (optional, to prevent errors)
    assert len(sequence) == len(output_sequences[0]), "Length mismatch"
    df = pd.DataFrame({**{f'Sequence{i+1}': s for i, s in enumerate(output_sequences)}, 'Original Sequence': sequence})
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f'Exported to: {filename}')

def read_xlsx_by_columns(sample_file_path, sheet_name: str | int = 0) -> dict:
    try:
        # 读取xlsx文件，不设置表头（header=None），保留原始数据
        df = pd.read_excel(sample_file_path, sheet_name=sheet_name, header=None)

        # 初始化结果字典
        column_data = {}

        # 遍历每一列（df.columns是列索引：0,1,2...）
        for col in df.columns:
            # 获取当前列的所有数据（转为列表）
            col_all_data = df[col].tolist()

            # 1. 取该列第一个元素作为key（转字符串+去空格，避免类型/空格问题）
            key = str(col_all_data[0]).strip() if len(col_all_data) > 0 else ""
            # 跳过空key（第一单元格为空的列）
            if key == "" or key == "nan":
                continue

            # 2. 取该列从第二个元素开始的所有数据作为value列表
            value = col_all_data[1:] if len(col_all_data) > 1 else []

            value = [item for item in value if not (isinstance(item, float) and np.isnan(item))]

            # 3. 存入字典
            column_data[key] = value

        return column_data

    except FileNotFoundError:
        # 修复原bug：将未定义的file_path改为self.sample_file_path
        print(f"错误：找不到文件 {sample_file_path}，请检查路径是否正确")
        return {}
    except KeyError:
        print(f"错误：工作表 {sheet_name} 不存在，请检查工作表名称/索引")
        return {}
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return {}


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

class Detector(nn.Module):
    """
    Peak detection model wrapper class for loading models, processing input sequences, detecting peaks, and visualizing results
    Based on a trained network model, implements peak detection using Peak Intersection over Union (Peak IoU) and Non-Maximum Suppression (NMS)
    """

    def __init__(self, peak_iou=True, nms=True, three_head=True, dfl=True, bce_or_mse='bce', thresh=0.5,
                 peak_iou_thresh=0.5, show_results=True):
        """
        Initialize the detector
        Parameters:
            peak_iou: Whether to use Peak IoU as the matching metric
            nms: Whether to enable Non-Maximum Suppression
            three_head: Whether the model uses a three-detection-head structure
            dfl: Whether the model uses Distribution Focal Loss (DFL)
            bce_or_mse: Loss function type (BCE or MSE)
            thresh: Confidence threshold (used to filter valid predictions)
            peak_iou_thresh: Peak IoU threshold (used for NMS)
            show_results: Whether to display visualization results
        """
        super(Detector, self).__init__()
        # Initialize detection network
        self.net = Net(three_head=three_head, dfl=dfl)
        # Determine the optimal weight file path based on parameters
        if peak_iou:
            bce_or_mse = 'none'  # BCE/MSE is not used when Peak IoU is enabled
        self.best_weight_path = f"params/net_best_{peak_iou}_{nms}_{three_head}_{dfl}_{bce_or_mse}.pth"

        # Load pre-trained weights (if available)
        if os.path.exists(self.best_weight_path):
            self.net.load_state_dict(torch.load(self.best_weight_path, map_location='cpu'))
        self.net.eval()  # Set to evaluation mode

        self.thresh = thresh  # Confidence threshold
        self.peak_iou_thresh = peak_iou_thresh  # Peak IoU threshold
        self.show_results = show_results

    def get_true_position(self, sequence):
        """
        Process input sequence and obtain true position predictions of peaks through the network
        Parameters:
            sequence: Input sequence data (raw signal)
        Returns:
            Filtered peak prediction results (predictions with confidence above threshold)
        """
        output = self.net(sequence)  # Model prediction
        # Transform output format and apply grid offset
        output = Utils.transform_output(output, grid_offset=0.5)
        # Split output and retain valid part (remove redundant first/last dimensions)
        _, output, _ = torch.split(output, [1, output.size(-1) - 2, 1], dim=-1)
        # Filter prediction results based on confidence threshold
        mask = output[:, :, 0] > self.thresh
        output = output[mask]
        output = output.squeeze(0)  # Compress dimensions
        return output

    @staticmethod
    def peak_iou_operate(output1, output2):
        """
        Calculate Peak IoU (Peak Intersection over Union) between two peak prediction boxes
        Peak IoU adds a penalty term for peak position distance to traditional IoU, making it more suitable for peak detection tasks
        Parameters:
            output1: Prediction box parameters of the first peak [start position, end position, peak position, height]
            output2: Prediction box parameters of the second peak [start position, end position, peak position, height]
        Returns:
            Peak IoU value (higher value indicates better matching)
        """
        # Calculate left/right boundaries of intersection region (max as start, min as end)
        intersection_start = torch.max(output1[1], output2[1])
        intersection_end = torch.min(output1[2], output2[2])
        # Calculate length of intersection region (ensure non-negative)
        intersection_length = torch.clamp(intersection_end - intersection_start, min=0.0)
        # Calculate height of intersection region (take minimum of both heights)
        intersection_height = torch.min(output1[4], output2[4])
        intersection_area = intersection_length * intersection_height  # Intersection area

        # Calculate areas of the two prediction boxes
        output1_area = (output1[2] - output1[1]) * output1[4]
        output2_area = (output2[2] - output2[1]) * output2[4]

        # Calculate union area (sum of areas minus intersection)
        union_area = output1_area + output2_area - intersection_area

        # Calculate traditional Intersection over Union (IoU)
        iou = intersection_area / union_area

        # Calculate peak position distance and normalization factor
        peak_distance = torch.abs(output1[3] - output2[3])  # Absolute distance between peak positions
        union_start = torch.min(output1[1], output2[1])  # Union region start position
        union_end = torch.max(output1[2], output2[2])  # Union region end position
        union_distance = torch.abs(union_end - union_start)  # Union region length (used for normalization)

        # Calculate Peak IoU (IoU minus normalized peak distance penalty)
        peak_iou = iou - (peak_distance / union_distance)
        return peak_iou.item()  # Convert to Python numeric value

    def non_max_suppression(self, output):
        """
        Apply Non-Maximum Suppression (NMS) to remove overlapping peak predictions
        Filter using Peak IoU to retain peaks with highest confidence and low overlap with other predictions
        Parameters:
            output: Raw prediction results (unfiltered peak predictions)
        Returns:
            Filtered peak prediction boxes (redundant overlapping predictions removed)
        """
        # Process dimensions (ensure at least 2D)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        # Sort in descending order of confidence (prioritize high-confidence predictions)
        output = output[output[:, 0].argsort(descending=True)]
        anchor_box_select = []  # Store filtered prediction boxes

        # Iterate through all prediction boxes for non-maximum suppression
        for i in range(output.size(0)):
            select_or_not = True  # Mark whether current prediction box is selected
            # Compare Peak IoU with already selected prediction boxes
            for anchor_box_ready in anchor_box_select:
                peak_iou = self.peak_iou_operate(anchor_box_ready, output[i])
                # If Peak IoU exceeds threshold, consider overlap too high and skip current prediction box
                if peak_iou > self.peak_iou_thresh:
                    select_or_not = False
            if select_or_not:
                anchor_box_select.append(output[i])  # Select current prediction box

        # Convert to tensor and remove confidence dimension
        if len(anchor_box_select) == 0:
            anchor_box_select = [torch.zeros(output.size(-1), device=output.device)]
        anchor_box_select = torch.stack(anchor_box_select, dim=0)
        _, anchor_box_select = torch.split(anchor_box_select, [1, anchor_box_select.size(-1) - 1], dim=-1)
        return anchor_box_select

    @staticmethod
    def filterNestedBoxes(anchor_box):
        """
        Filter nested prediction boxes (remove prediction boxes completely contained within other boxes)
        Parameters:
            anchor_box: Collection of prediction boxes
        Returns:
            Filtered prediction boxes (no nested relationships)
        """
        anchor_box_select = []  # Store filtered prediction boxes
        # Iterate through all prediction boxes
        for i in range(anchor_box.size(0)):
            select_or_not = True  # Mark whether to select
            # Compare with already selected prediction boxes to determine nesting
            for anchor_box_ready in anchor_box_select:
                # If current box is completely contained in a selected box, skip selection
                if anchor_box[i][0] > anchor_box_ready[0] and anchor_box[i][1] < anchor_box_ready[1]:
                    select_or_not = False
            if select_or_not:
                anchor_box_select.append(anchor_box[i])  # Select current prediction box
        # Convert to tensor
        anchor_box_select = torch.stack(anchor_box_select, dim=0)
        return anchor_box_select

    @staticmethod
    def target_transform(target):
        """
        Transform target or prediction box parameters for generating Gaussian peak sequences
        Convert [start position, end position, peak position, height] to Gaussian distribution parameters
        Parameters:
            target: Raw target/prediction box parameters
        Returns:
            Gaussian distribution parameters (mean, amplitude, left standard deviation, right standard deviation)
        """
        target_second = target.clone()
        # Extract peak position as Gaussian distribution mean
        target_second[:, 0] = target[:, 2]
        # Extract height as amplitude
        target_second[:, 1] = target[:, 3]
        # Calculate left standard deviation (distance from peak to start / 3)
        target_second[:, 2] = (target[:, 2] - target[:, 0]) / 3
        # Calculate right standard deviation (distance from end to peak / 3)
        target_second[:, 3] = (target[:, 1] - target[:, 2]) / 3
        # Split parameters and convert to numpy arrays
        multi_gaussian_params = torch.split(target_second, 1, dim=-1)
        multi_gaussian_params = tuple(x.detach().numpy() for x in multi_gaussian_params)
        return multi_gaussian_params

    @staticmethod
    def generate_gaussian_sequence(gaussian_params):
        """
        Generate Gaussian peak sequences (superposition of multiple Gaussian peaks) based on Gaussian distribution parameters
        Parameters:
            gaussian_params: Tuple of Gaussian distribution parameters (location, amplitude, left std, right std)
        Returns:
            List of generated Gaussian peak sequences (each element is a Gaussian peak)
        """
        # Parse parameters
        location, amplitude, left_std, right_std = gaussian_params
        # Generate sequence indices (assuming data length is DATA_LENGTH)
        indices = np.arange(DATA_LENGTH)
        sequences = []

        # Generate Gaussian sequence for each peak
        for i in range(len(location)):
            loc = location[i]
            amp = amplitude[i]
            left = left_std[i]
            right = right_std[i]

            # Left Gaussian distribution (use left standard deviation for left side of peak)
            left_half = np.exp(-0.5 * ((indices - loc) / left) ** 2)
            # Right Gaussian distribution (use right standard deviation for right side of peak)
            right_half = np.exp(-0.5 * ((indices - loc) / right) ** 2)
            # Merge left and right halves to form asymmetric Gaussian peak
            gaussian = np.where(indices < loc, left_half, right_half)
            gaussian *= amp  # Apply amplitude
            sequences.append(gaussian)

        return sequences

    @staticmethod
    def adjust_back_to_length(sequence, k_x):
        # Original x coordinates
        x_original = np.arange(len(sequence))
        # Target x coordinates
        x_new = np.linspace(0, len(sequence) - 1, int(len(sequence) / k_x))
        # Use linear interpolation to adjust sequence to fixed length
        f_interp = interp1d(x_original, sequence, kind='linear', fill_value="extrapolate")
        new_data = f_interp(x_new)
        # Convert to PyTorch tensor and adjust shape
        sequence = torch.tensor(new_data).float()
        return sequence

    def __call__(self, sequence, target, max_value=1.0, k_x=1.0, k_y=1.0, width_max=20 * 255):
        """
        Main call function: process input sequence, perform peak detection, and visualize comparison between detection results and original sequence
        Parameters:
            sequence: Input raw signal sequence (e.g., waveform data)
            target: Ground truth peak labels (used for comparison with prediction results)
            max_value: Maximum amplitude value (used for normalization)
            k_x: X-axis scaling factor (used for coordinate conversion)
            k_y: Y-axis scaling factor (used for amplitude conversion)
            width_max: Maximum width value (used for area calculation)
        """
        # Filter non-zero target labels (remove invalid zero-value labels)
        target = (target[torch.any(torch.tensor(target != 0), dim=-1)])
        # Obtain filtered peak prediction results through model (confidence above threshold)
        output = self.get_true_position(sequence)

        # Apply Non-Maximum Suppression (NMS) to remove overlapping prediction boxes
        anchor_box_select = self.non_max_suppression(output)

        # Convert target labels and prediction boxes to Gaussian distribution parameters (for generating Gaussian peaks)
        target_multi_gaussian_params = self.target_transform(target)
        output_multi_gaussian_params = self.target_transform(anchor_box_select)

        # Parse Gaussian parameters of prediction results (location, amplitude, left std, right std)
        output_locations, output_amplitude, output_left_stds, output_right_stds = output_multi_gaussian_params
        # Parse Gaussian parameters of ground truth targets
        target_locations, target_amplitude, target_left_stds, target_right_stds = target_multi_gaussian_params

        # Sort by peak position (ensure left-to-right display during plotting)
        sorted_indices = np.argsort(output_locations, axis=0)
        output_locations = output_locations[sorted_indices].squeeze(-1)  # Compress dimensions
        output_amplitude = output_amplitude[sorted_indices].squeeze(-1)
        output_left_stds = output_left_stds[sorted_indices].squeeze(-1)
        output_right_stds = output_right_stds[sorted_indices].squeeze(-1)

        # Calculate area of each predicted peak (based on amplitude and standard deviation)
        output_areas = output_amplitude / k_y * width_max * (output_left_stds + output_right_stds) * math.sqrt(
            2 * math.pi)/2 / k_x

        # Reconstruct sorted prediction Gaussian parameters
        output_multi_gaussian_params = output_locations, output_amplitude, output_left_stds, output_right_stds

        # Generate predicted Gaussian peak sequences and ground truth Gaussian peak sequences based on Gaussian parameters
        output_sequences = self.generate_gaussian_sequence(output_multi_gaussian_params)
        target_sequences = self.generate_gaussian_sequence(target_multi_gaussian_params)

        # Process original sequence: remove extra dimensions and apply y-axis scaling
        sequence = sequence.squeeze(0).squeeze(0)
        sequence = self.adjust_back_to_length(sequence / k_y * width_max, k_x)
        output_sequences_adjust = []
        for i in range(len(output_sequences)):
            output_sequences_adjust.append(self.adjust_back_to_length(output_sequences[i] / k_y * width_max, k_x))
        output_sequences = output_sequences_adjust

        output_left_edges = ((output_locations - 3 * output_left_stds) / k_x).squeeze(-1)
        output_right_edges = ((output_locations + 3 * output_right_stds) / k_x).squeeze(-1)
        output_locations = (output_locations / k_x).squeeze(-1)
        output_amplitude = (output_amplitude / k_y * width_max).squeeze(-1)
        output_areas = output_areas.squeeze(-1)

        if not self.show_results:
            return output_left_edges, output_right_edges, output_locations, output_amplitude, output_areas

        # Print predicted peak positions and corresponding areas (for quantitative analysis)
        print(output_left_edges)
        print(output_right_edges)
        print(output_locations)
        print(output_amplitude)
        print(output_areas)

        # -------------------------- Visualization: Original Sequence and Predicted Gaussian Peaks --------------------------
        # Create 2 rows × 1 column subplots (for displaying original sequence and predicted Gaussian peaks)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 4))

        # Plot each predicted Gaussian peak in ax2 (distinguished by different colors)
        for i in range(len(output_sequences)):
            ax2.plot(output_sequences[i], color=colors[i])

        # seq_to_excel(sequence,output_sequences,"sequence.xlsx")

        # Plot original sequence in ax1 (black curve)
        ax1.plot(sequence, color='black')

        # Set axis labels
        ax1.set_ylabel('Amplitude ')  # Y-axis label for ax1
        ax2.set_ylabel('Amplitude ')  # Y-axis label for ax2
        # Set axis ranges
        ax1.set_ylim(0, 1.2 * max(sequence))
        ax2.set_ylim(0, 1.2 * max(sequence))
        ax1.set_xlim(0, len(sequence))
        ax2.set_xlim(0, len(sequence))
        # Auto-adjust layout
        plt.tight_layout()
        plt.show()  # Display second set of figures

        return sequence, output_sequences


if __name__ == "__main__":
    """Main function: test detector functionality, including two modes - loading data from dataset and converting data from images"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Ignore warning messages

        # Mode 1: Load prediction data from dataset and perform detection
        for i in range(3):
            data = DataSet(num_samples=10, data_type='predict')  # Load prediction dataset
            sequence, target, _ = data[i]  # Get one sample
            sequence = sequence.unsqueeze(0)  # Add batch dimension
            print(sequence.shape)
            target = target.unsqueeze(0)
            print(target.shape)
            detector = Detector()  # Instantiate detector
            detector(sequence, target)  # Perform detection

        from utils import Utils

        utils = Utils()
        sample_data = read_xlsx_by_columns("sample/sample_row_sums.xlsx", "data")
        for i in range(85,102):
            sequence, k_x, k_y, width_max = adjust_sequence(sample_data[f"sample{i + 1}"],max_value=0.85)
            # print(sequence)
            sequence = torch.tensor(sequence).unsqueeze(0).unsqueeze(0).float()
            # print(sequence.shape)
            target = torch.zeros(1, 16, 4)
            # print(target.shape)
            detector = Detector()  # Instantiate detector
            sequence, output_sequences = detector(sequence, target, max_value=0.85, k_x=k_x, k_y=k_y,
                                                  width_max=width_max)  # Perform detection

        for i in range(85,102):
            print(i + 1)
            sequence, k_x, k_y, width_max = utils.transport_image(
                image_path=f"sample/sample_gel_lane_image/{i + 1}.jpg")
            # print(sequence)
            sequence = sequence.unsqueeze(0)
            # print(sequence.shape)
            target = torch.zeros(1, 16, 4)
            # print(target.shape)
            detector = Detector()  # Instantiate detector
            sequence, output_sequences = detector(sequence, target, max_value=0.85, k_x=k_x, k_y=k_y,
                                                  width_max=width_max)  # Perform detection