import torch
from config import *  # Import configuration parameters (e.g., grid length, data length, etc.)
from PIL import Image
import numpy as np
from scipy.interpolate import interp1d


class Utils:
    """Utility class containing various helper functions used in model training and inference,
    such as anchor calculation, output transformation, target transformation, etc."""

    @staticmethod
    def get_anchor_points(batch_size, outputs, all_or_small=True, grid_offset=0.5):
        """
        Calculate and return anchor positions (base points for object detection) based on batch size,
        output data, and grid offset.

        Parameters:
            batch_size (int): Batch size, used to replicate anchors to match each sample in the batch
            outputs (torch.Tensor): Model output tensor, used to determine device (CPU/GPU)
            all_or_small (bool): Whether to use anchors of all sizes (small + medium + large) or only small size
            grid_offset (float): Grid offset ratio (0-1), adjusting anchor position within the grid

        Returns:
            torch.Tensor: Adjusted anchor tensor with shape (batch_size, number of anchors, 1)
        """
        # Get interval of different size grids from configuration
        small_grid_length = GRID_LENGTH[0]  # Small grid length
        medium_grid_length = GRID_LENGTH[1]  # Medium grid length
        large_grid_length = GRID_LENGTH[2]  # Large grid length

        # Calculate anchor positions for different size grids (start point + offset)
        small_anchor_points = torch.arange(0, DATA_LENGTH, small_grid_length) + grid_offset * small_grid_length
        medium_anchor_points = torch.arange(0, DATA_LENGTH, medium_grid_length) + grid_offset * medium_grid_length
        large_anchor_points = torch.arange(0, DATA_LENGTH, large_grid_length) + grid_offset * large_grid_length

        # Merge anchors (select all or only small size as needed) and adjust shape
        if all_or_small:
            anchor_points = torch.cat((small_anchor_points, medium_anchor_points, large_anchor_points)).reshape(1, -1,
                                                                                                                1)
        else:
            anchor_points = small_anchor_points.reshape(1, -1, 1)

        # Replicate anchors by batch size (same anchors for each sample in the batch)
        anchor_points = anchor_points.repeat(batch_size, 1, 1)

        # Ensure anchors are on the same device as outputs (CPU/GPU synchronization)
        anchor_points = anchor_points.to(outputs.device)

        # Return detached float tensor (not participating in gradient calculation)
        return anchor_points.detach().float()

    @staticmethod
    def transform_output_dfl(outputs, three_head=True, grid_offset=0.5):
        """
        Process model outputs using Distribution Focal Loss (DFL), converting predicted parameters
        to actual physical coordinates.

        Parameters:
            outputs: Model output tensor
            three_head: Whether to use three-detector-head structure (corresponding to different size grids)
            grid_offset: Grid offset value

        Returns:
            Transformed output tensor containing anchor points, confidence, start/end positions,
            peak positions, height, etc.
        """
        # Get anchor positions
        anchor_points = Utils.get_anchor_points(int(outputs.shape[0]), outputs, all_or_small=three_head,
                                                grid_offset=grid_offset)

        # Split the last dimension of output (confidence p, left/right offset, peak position ratio, height, etc.)
        parts = torch.split(outputs, [1, REG_MAX + 1, REG_MAX + 1, REG_MAX + 1, REG_MAX + 1], dim=-1)
        outputs_p, outputs_left, outputs_right, outputs_plr, outputs_height = parts

        # Initialize maximum value tensor for DFL calculation (sequence from 0 to REG_MAX)
        reg_max = torch.arange(0, REG_MAX + 1, device=outputs.device).float()

        # Calculate actual values of DFL outputs (convert discrete distribution to continuous values via weighted sum)
        def cal_value(outputs, max_val, scale_factor):
            return torch.sum(outputs * max_val, dim=-1) * scale_factor / max_val[-1]

        # Calculate actual values of left/right offset, peak position ratio, and height
        outputs_left = cal_value(outputs_left, reg_max, 1).reshape(outputs_left.shape[0], -1, 1)
        outputs_right = cal_value(outputs_right, reg_max, 1).reshape(outputs_right.shape[0], -1, 1)

        # For three-head structure, scale offsets for different size grids separately
        if three_head:
            outputs_left_s, outputs_left_m, outputs_left_l = torch.split(outputs_left, [128, 64, 32], dim=-2)
            outputs_right_s, outputs_right_m, outputs_right_l = torch.split(outputs_right, [128, 64, 32], dim=-2)
            outputs_left_s = outputs_left_s * (PEAK_WIDTH_MAX // 4)  # Small grid scale factor
            outputs_left_m = outputs_left_m * (PEAK_WIDTH_MAX // 2)  # Medium grid scale factor
            outputs_left_l = outputs_left_l * PEAK_WIDTH_MAX  # Large grid scale factor
            outputs_right_s = outputs_right_s * (PEAK_WIDTH_MAX // 4)
            outputs_right_m = outputs_right_m * (PEAK_WIDTH_MAX // 2)
            outputs_right_l = outputs_right_l * PEAK_WIDTH_MAX
            outputs_left = torch.cat((outputs_left_s, outputs_left_m, outputs_left_l), dim=-2)
            outputs_right = torch.cat((outputs_right_s, outputs_right_m, outputs_right_l), dim=-2)
        else:
            outputs_left = outputs_left * PEAK_WIDTH_MAX
            outputs_right = outputs_right * PEAK_WIDTH_MAX

        # Calculate actual height values (scale to maximum peak height)
        outputs_height = cal_value(outputs_height, reg_max, PEAK_HEIGHT_MAX).reshape(outputs_height.shape[0], -1, 1)
        outputs_plr = cal_value(outputs_plr, reg_max, 1).reshape(outputs_plr.shape[0], -1, 1)  # Peak position ratio

        # Calculate start and end positions of peaks
        outputs_start = anchor_points - outputs_left
        outputs_end = anchor_points + outputs_right

        # Calculate specific peak positions (start position + ratio * width)
        outputs_peak = outputs_start + (outputs_end - outputs_start) * outputs_plr

        # Concatenate all results (anchors, confidence, start/end/peak positions, height, ratio)
        parts = anchor_points, outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height, outputs_plr
        outputs = torch.cat(parts, dim=-1)

        return outputs

    @staticmethod
    def transform_output_non_dfl(outputs, three_head=True, grid_offset=0.5):
        """
        Process model outputs without DFL, directly converting predicted parameters to actual physical coordinates.

        Parameters and return values are similar to transform_output_dfl, applicable to non-DFL mode
        """
        # Get anchor positions
        anchor_points = Utils.get_anchor_points(int(outputs.shape[0]), outputs, all_or_small=three_head,
                                                grid_offset=grid_offset)

        # Split the last dimension of output (confidence p, left/right offset, peak position ratio, height)
        parts = torch.split(outputs, [1, 1, 1, 1, 1], dim=-1)
        outputs_p, outputs_left, outputs_right, outputs_plr, outputs_height = parts

        # For three-head structure, scale offsets for different size grids separately
        if three_head:
            outputs_left_s, outputs_left_m, outputs_left_l = torch.split(outputs_left, [128, 64, 32], dim=-2)
            outputs_right_s, outputs_right_m, outputs_right_l = torch.split(outputs_right, [128, 64, 32], dim=-2)
            outputs_left_s = outputs_left_s * (PEAK_WIDTH_MAX // 4)
            outputs_left_m = outputs_left_m * (PEAK_WIDTH_MAX // 2)
            outputs_left_l = outputs_left_l * PEAK_WIDTH_MAX
            outputs_right_s = outputs_right_s * (PEAK_WIDTH_MAX // 4)
            outputs_right_m = outputs_right_m * (PEAK_WIDTH_MAX // 2)
            outputs_right_l = outputs_right_l * PEAK_WIDTH_MAX
            outputs_left = torch.cat((outputs_left_s, outputs_left_m, outputs_left_l), dim=-2)
            outputs_right = torch.cat((outputs_right_s, outputs_right_m, outputs_right_l), dim=-2)
        else:
            outputs_left = outputs_left * PEAK_WIDTH_MAX
            outputs_right = outputs_right * PEAK_WIDTH_MAX

        # Scale height and peak position ratio
        outputs_height = outputs_height * PEAK_HEIGHT_MAX
        outputs_plr = outputs_plr * 1  # Ratio requires no additional scaling

        # Calculate start, end, and peak positions
        outputs_start = anchor_points - outputs_left
        outputs_end = anchor_points + outputs_right
        outputs_peak = outputs_start + (outputs_end - outputs_start) * outputs_plr

        # Concatenate all results
        parts = anchor_points, outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height, outputs_plr
        outputs = torch.cat(parts, dim=-1)

        return outputs

    @staticmethod
    def transform_output_cnnnet(outputs, grid_offset=0.5):
        """
        Process model outputs without DFL, directly converting predicted parameters to actual physical coordinates.

        Parameters and return values are similar to transform_output_dfl, applicable to non-DFL mode
        """
        # Get anchor positions
        anchor_points = Utils.get_anchor_points(int(outputs.shape[0]), outputs, all_or_small=three_head,
                                                grid_offset=grid_offset)

        # Split the last dimension of output (confidence p, left/right offset, peak position ratio, height)
        parts = torch.split(outputs, [1, 1, 1, 1, 1], dim=-1)
        outputs_p, outputs_left, outputs_right, outputs_plr, outputs_height = parts

        outputs_left = outputs_left * PEAK_WIDTH_MAX
        outputs_right = outputs_right * PEAK_WIDTH_MAX

        # Calculate start, end, and peak positions
        outputs_start = anchor_points - outputs_left
        outputs_end = anchor_points + outputs_right
        outputs_peak = outputs_start + (outputs_end - outputs_start) * outputs_plr

        # Concatenate all results
        parts = anchor_points, outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height, outputs_plr
        outputs = torch.cat(parts, dim=-1)

        return outputs

    @staticmethod
    def transform_output(outputs, grid_offset=0.5, cnnnet=False):
        """
        Unified output transformation entry, automatically determining to use DFL or non-DFL mode
        based on output shape.

        Parameters:
            outputs: Model output tensor
            grid_offset: Grid offset value

        Returns:
            Transformed output tensor (containing actual physical coordinates)
        """
        if cnnnet:
            outputs = Utils.transform_output_cnnnet(outputs)
            return outputs

        # Determine if three-head structure and DFL are used based on output shape
        if outputs.shape[-2] == 128:
            # Single detector head (small grid)
            if outputs.shape[-1] == 5:
                outputs = Utils.transform_output_non_dfl(outputs, three_head=False, grid_offset=grid_offset)
            else:
                outputs = Utils.transform_output_dfl(outputs, three_head=False, grid_offset=grid_offset)
        else:
            # Three detector heads (small + medium + large grids)
            if outputs.shape[-1] == 5:
                outputs = Utils.transform_output_non_dfl(outputs, three_head=True, grid_offset=grid_offset)
            else:
                outputs = Utils.transform_output_dfl(outputs, three_head=True, grid_offset=grid_offset)

        return outputs

    @staticmethod
    def transform_target(target, grid_offset=0.5):
        """
        Convert ground truth labels (targets) to a format matching model outputs for loss calculation.

        Parameters:
            target: Original ground truth label tensor
            grid_offset: Grid offset value

        Returns:
            Transformed target tensor corresponding to model output structure
        """
        # Get anchor positions for small grids
        small_grid_length = GRID_LENGTH[0]
        small_anchor_points = torch.arange(0, DATA_LENGTH, small_grid_length) + grid_offset * small_grid_length
        small_anchor_points = small_anchor_points.to(target.device)
        small_anchor_points = small_anchor_points.repeat(target.size(0), 1)  # Replicate by batch

        # Initialize target tensor (batch size, number of anchors, 5 parameters)
        target_tensor = torch.zeros((target.size(0), 128, 5), device=target.device)

        # Filter valid targets (exclude invalid zero-value labels)
        valid_mask = target[:, :, 2] > 1e-9  # Peak position is valid
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)
        i_indices = valid_indices[0]  # Batch indices
        j_indices = valid_indices[1]  # Target indices

        # Calculate anchor indices corresponding to targets (divided into different grids based on peak position)
        index = (target[i_indices, j_indices, 2] // 8).long()

        # Populate target tensor (confidence, left/right offset, peak ratio, height)
        target_tensor[i_indices, index, 0] = 0.99  # Set confidence to 0.99 (close to 1)
        # Left offset = (anchor - start position) / maximum width
        value_1 = (small_anchor_points[i_indices, index] - target[i_indices, j_indices, 0]) / PEAK_WIDTH_MAX
        target_tensor[i_indices, index, 1] = value_1.to(target_tensor.dtype)
        # Right offset = (end position - anchor) / maximum width
        value_2 = (target[i_indices, j_indices, 1] - small_anchor_points[i_indices, index]) / PEAK_WIDTH_MAX
        target_tensor[i_indices, index, 2] = value_2.to(target_tensor.dtype)
        # Peak position ratio = (peak position - start position) / width
        value_3 = (target[i_indices, j_indices, 2] - target[i_indices, j_indices, 0]) / (
                target[i_indices, j_indices, 1] - target[i_indices, j_indices, 0])
        target_tensor[i_indices, index, 3] = value_3.to(target_tensor.dtype)
        # Height normalization
        target_tensor[i_indices, index, 4] = target[i_indices, j_indices, 3].to(target_tensor.dtype)

        return target_tensor

    @staticmethod
    def read_image(image_path):
        """
        Read image and calculate the average value of pixel values per row (normalized).

        Parameters:
            image_path: Image file path

        Returns:
            row_sums: Normalized sum of pixel values per row
            width_max: Width normalization factor
        """
        # Open image
        img = Image.open(image_path)

        # Convert image to grayscale (if image is color)
        img = img.convert('L')

        # Get image width and height
        width, height = img.size

        # Initialize list to store sum of pixel values for each row
        row_sums = []

        width_max = width * 255  # Calculate maximum possible value (for normalization)

        # Iterate over each row of the image
        for y in range(height):
            row_sum = 0
            for x in range(width):
                # Get grayscale value of current pixel
                pixel = img.getpixel((x, y))
                # Accumulate all pixel values of current row
                row_sum += pixel
            # Add sum of pixel values of current row to list (normalization)
            row_sums.append(row_sum / width_max)

        return row_sums, width_max

    @staticmethod
    def transport_image(image_path, three_channel=False, max_value=0.85):
        """
        Process image data and convert it to the format required for model input.

        Parameters:
            image_path: Image path
            three_channel: Whether to generate three-channel data (original data, first-order difference, second-order difference)
            max_value: Maximum value after normalization

        Returns:
            sequence: Processed sequence data
            k_x: X-axis scale factor
            k_y: Y-axis scale factor
            width_max: Width normalization factor
        """
        # Read image and get sum of pixels per row
        sequence, width_max = Utils.read_image(image_path)

        # Original x coordinates
        x_original = np.arange(len(sequence))
        # Calculate x-axis scale factor (scale sequence to length 1024)
        k_x = 1024 / len(sequence)

        # Target x coordinates (1024 points)
        x_new = np.linspace(0, len(sequence) - 1, 1024)

        # Resize sequence to fixed length (1024) using linear interpolation
        f_interp = interp1d(x_original, sequence, kind='linear', fill_value="extrapolate")
        new_data = f_interp(x_new)

        # Normalization (make maximum value equal to specified value)
        sequence = (new_data / np.max(new_data)) * max_value
        k_y = max_value / np.max(new_data)  # Calculate y-axis scale factor

        # Convert to PyTorch tensor and adjust shape
        sequence = torch.tensor(sequence).float()
        sequence = sequence.reshape(1, -1)

        # Calculate first-order difference (for detecting change rate)
        dx = sequence[:, 1:] - sequence[:, :-1]  # Shape [1, 1023]
        # Add zero at the beginning to maintain shape consistency
        dx_pad = torch.cat((torch.zeros_like(dx[:, 0].unsqueeze(-1)), dx), dim=-1)  # Shape [1, 1024]

        # Calculate second-order difference (for detecting curvature)
        ddx = dx[:, 1:] - dx[:, :-1]  # Shape [1, 1022]
        # Add zeros at the beginning and end to maintain shape consistency
        ddx_pad = torch.cat(
            (torch.zeros_like(ddx[:, 0].unsqueeze(-1)), ddx, torch.zeros_like(ddx[:, 0].unsqueeze(-1))),
            dim=-1)  # Shape [1, 1024]

        # Whether to use three-channel sequence (original data + first-order difference + second-order difference)
        if three_channel:
            sequence = torch.cat((sequence, dx_pad, ddx_pad), dim=0)

        return sequence, k_x, k_y, width_max


if __name__ == "__main__":
    """Test code: demonstrate the use of utility functions"""

    # Test transform_output function (non-DFL version)
    outputs1 = torch.zeros(1, 224, 5)  # Create sample output (non-DFL, 5 output dimensions)
    replace_values = torch.tensor([[0.9, 0.002, 0.003, 0.01, 0.7]])
    outputs1[:, 1, :] = replace_values  # Set predicted values for one anchor

    # Test transform_output function (DFL version)
    outputs2 = torch.zeros(1, 224, 65)  # Create sample output (DFL, 65 output dimensions)
    # Construct predicted values in DFL format (confidence + 4 distribution parameters)
    replace_values1 = torch.tensor([[0.9]])
    replace_values2 = torch.tensor(
        [[0.01, 0.002, 0.003, 0.004, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    replace_values3 = torch.tensor(
        [[0.01, 0.002, 0.003, 0.004, 0.005, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    replace_values4 = torch.tensor(
        [[0.01, 0.02, 0.03, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    replace_values5 = torch.tensor(
        [[0.01, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    replace_values = torch.cat(
        (replace_values1, replace_values2, replace_values3, replace_values4, replace_values5), dim=-1)
    outputs2[:, 1, :] = replace_values  # Set predicted values for one anchor

    # Print shape of transformed outputs
    print(Utils.transform_output(outputs1, 0.5).shape)  # Non-DFL version
    print(Utils.transform_output(outputs2, 0.5).shape)  # DFL version

    # Test transform_target function
    target = torch.zeros(2, 8, 4)  # Create sample target labels
    replace_values1 = torch.tensor([[3, 6, 5, 0.4]])  # Start position, end position, peak position, height
    replace_values2 = torch.tensor([[9, 14, 10, 0.6]])
    target[:, 0, :] = replace_values1
    target[:, 1, :] = replace_values2

    # Test transform_target with actual data
    target = torch.tensor([[[6.1146e+02, 6.3063e+02, 6.2105e+02, 1.8497e-01],
                            [9.1194e+01, 1.0572e+02, 1.0114e+02, 7.8371e-01],
                            [2.2919e+02, 2.4571e+02, 2.3634e+02, 3.0513e-01],
                            [1.9348e+02, 2.0586e+02, 2.0250e+02, 1.5922e-01],
                            [8.0211e+02, 8.2182e+02, 8.1196e+02, 6.1746e-01],
                            [8.6941e+02, 8.8157e+02, 8.7341e+02, 3.4644e-01],
                            [8.1627e+02, 8.2905e+02, 8.2473e+02, 3.2180e-01],
                            [1.0995e+02, 1.2530e+02, 1.1758e+02, 3.6587e-01],
                            [1.7190e+02, 1.8629e+02, 1.8130e+02, 2.0642e-01],
                            [4.6973e+02, 4.8588e+02, 4.7642e+02, 3.8803e-01],
                            [3.6045e+02, 3.6873e+02, 3.6393e+02, 6.6721e-01],
                            [8.4857e+02, 8.6776e+02, 8.5816e+02, 3.4538e-01],
                            [3.6798e+02, 3.8805e+02, 3.7868e+02, 1.6544e-01],
                            [6.5249e+02, 6.6983e+02, 6.6162e+02, 2.4863e-01],
                            [7.9790e+02, 8.1263e+02, 8.0532e+02, 7.5976e-01],
                            [7.5650e+02, 7.6516e+02, 7.5950e+02, 6.3348e-01]]])