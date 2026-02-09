import warnings
import torch
from utils import Utils
from dataset import *
from config import *
from Net import Net


class Analysis:
    """
    Peak detection result analysis class, used to process model outputs and calculate evaluation metrics
    """

    def __init__(self, nms=True, confidence_threshold=0.5, peak_iou_thresh=0.5, traditional_or_neural='neural'):
        """
        Initialize the analyzer
        Args:
            nms (bool): Whether to enable Non-Maximum Suppression (NMS)
            confidence_threshold(float): Confidence threshold for initial screening of positive samples
            peak_iou_thresh (float): Peak_IoU threshold for NMS and matching judgment
        """
        self.nms = nms
        self.confidence_threshold = confidence_threshold
        self.peak_iou_thresh = peak_iou_thresh
        self.traditional_or_neural = traditional_or_neural

    def get_true_position(self, output):
        """
        Convert network output to valid detection boxes initially filtered by confidence threshold

        Args:
            output (Tensor): Raw model output tensor (shape: [batch, grid, channels])

        Returns:
            Tensor: Filtered valid detection boxes (shape: [num_boxes, 5], including confidence, coordinates, etc.)
            outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height
        """
        # Reshape model output to [1,224,7]
        # 7-anchor_points, outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height, outputs_plr
        output = Utils.transform_output(output, grid_offset=0.5)

        # Remove anchor_points and outputs_plr, shape becomes [1,224,5]
        _, output, _ = torch.split(output, [1, output.size(-1) - 2, 1], dim=-1)

        # Create confidence mask to retain detection boxes above threshold
        mask = output[:, :, 0] > self.confidence_threshold
        output = output[mask]

        return output

    @staticmethod
    def nms_peak_iou_operate(output1, output2):
        """
        Calculate Peak-IoU between two detection boxes for subsequent NMS

        Args:
            output1 (Tensor): Detection box 1 (shape: [5], including outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height)
            output2 (Tensor): Detection box 2 (shape: [5], including outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height)

        Returns:
            float: Peak-IoU value
        """
        # Calculate left and right boundaries of intersection area (max as start, min as end)
        intersection_start = torch.max(output1[1], output2[1])
        intersection_end = torch.min(output1[2], output2[2])

        # Calculate intersection length, ensuring it is not less than 0
        intersection_length = torch.clamp(intersection_end - intersection_start, min=0.0)
        intersection_height = torch.min(output1[4], output2[4])

        # Calculate intersection area
        intersection_area = intersection_length * intersection_height

        # Calculate area of the two boxes
        output1_area = (output1[2] - output1[1]) * output1[4]
        output2_area = (output2[2] - output2[1]) * output2[4]

        # Calculate union area
        union_area = output1_area + output2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        # Calculate peak distance (difference of x-coordinates of box centers)
        peak_distance = torch.abs(output1[3] - output2[3])

        # Calculate distance between left and right boundaries of union area
        union_start = torch.min(output1[1], output2[1])
        union_end = torch.max(output1[2], output2[2])
        union_distance = torch.abs(union_end - union_start)

        # Calculate peak_iou
        peak_iou = iou - (peak_distance / union_distance)

        return peak_iou

    def non_max_suppression(self, output):
        """
        Non-Maximum Suppression (NMS) algorithm to eliminate redundant boxes based on Peak_IoU threshold
        Args:
            output (Tensor): List of detection boxes (shape: [num_boxes, 5])
        Returns:
            Tensor: Filtered detection boxes (shape: [PEAK_NUM_MAX, 5], padded with zeros to maximum number)
        """
        # Sort detection boxes in descending order of confidence
        output = output[output[:, 0].argsort(descending=True)]

        # Initialize selected anchor box tensor (padded with zeros)
        anchor_box_select = torch.zeros(PEAK_NUM_MAX, 5)

        # Initialize counter for selected boxes
        j = int(0)

        for i in range(output.size(0)):
            select_or_not = True

            # Check if Peak_IoU with selected boxes exceeds threshold
            for anchor_box_ready in anchor_box_select:
                peak_iou = self.nms_peak_iou_operate(anchor_box_ready, output[i])

                if peak_iou > self.peak_iou_thresh:
                    # Boxes with excessive Peak_IoU have high overlap and are not selected
                    select_or_not = False

            if select_or_not:
                # Add to filtered detection boxes if selected
                if j < PEAK_NUM_MAX:
                    anchor_box_select[j] = output[i]
                j += 1

        return anchor_box_select

    @staticmethod
    def align_first_dim(output, target):
        """
        Align the first dimension (usually batch or sequence length) of two tensors to make them the same size.
        Mainly used to align model predictions with ground truth labels for subsequent calculations (e.g., loss function calculation).
        Args:
            output: Model prediction tensor
            target: Ground truth label tensor
        Returns:
            tuple: Aligned output tensor, aligned target tensor, and the aligned first dimension size
        """
        # If output is 1D tensor, add a dimension to make it 2D (e.g., from [5] to [1, 5])
        if output.dim() == 1:
            output = output.unsqueeze(0)

        # Sort output in descending order by the first column (usually for sorting predictions by confidence)
        output = output[output[:, 0].argsort(descending=True)]

        # Calculate the maximum length of the first dimension of output and target as the aligned length
        max_len = max(output.size(0), target.size(0))

        # Define padding function to extend tensor to max_len
        # Pad with 0 at the end of the tensor
        pad = lambda t: torch.cat(
            [t, torch.zeros(max_len - t.size(0), *t.shape[1:], dtype=t.dtype, device=t.device)], dim=0)

        # Apply padding function to output and target to make their first dimension length max_len
        return pad(output), pad(target), max_len

    @staticmethod
    def peak_iou_operate(output, target, max_len):
        """
        Calculate the "Peak IoU" metric between predictions and ground truth, combining traditional IoU and peak distance information.
        Args:
            output: Model prediction tensor, shape [max_len, 5], each row is [confidence, start time, end time, peak position, height]
            target: Ground truth tensor, shape [max_len, 5], each row is [?, start time, end time, peak position, height]
            max_len: Maximum sequence length for alignment
        Returns:
            peak_iou: Calculated Peak IoU tensor, shape [max_len, max_len]
        """
        # Expand dimensions to calculate all pairwise combinations between predictions and ground truth
        output = output.unsqueeze(0)  # [1, max_len, 5]
        target = target.unsqueeze(1)  # [max_len, 1, 5]
        output = output.repeat(max_len, 1, 1)  # [max_len, max_len, 5]
        target = target.repeat(1, max_len, 1)  # [max_len, max_len, 5]

        # Calculate intersection boundaries (max for left, min for right)
        intersection_start = torch.max(output[:, :, 1], target[:, :, 0])  # Intersection start
        intersection_end = torch.min(output[:, :, 2], target[:, :, 1])  # Intersection end

        # Calculate intersection length, ensuring non-negative (0 if no intersection)
        intersection_length = torch.clamp(intersection_end - intersection_start, min=0.0)

        # Calculate intersection height (minimum of predicted and ground truth height)
        intersection_height = torch.min(output[:, :, 4], target[:, :, 3])

        # Calculate intersection area (time length × height)
        intersection_area = intersection_length * intersection_height

        # Calculate area of predicted and ground truth regions
        pred_area = (output[:, :, 2] - output[:, :, 1]) * output[:, :, 4]
        gt_area = (target[:, :, 1] - target[:, :, 0]) * target[:, :, 3]

        # Calculate union area
        union_area = pred_area + gt_area - intersection_area

        # Calculate traditional IoU (Intersection over Union)
        iou = intersection_area / union_area

        # Get predicted and ground truth peak positions
        output_peak = output[:, :, 3]
        target_peak = target[:, :, 2]

        # Calculate absolute distance between peaks
        peak_distance = torch.abs(output_peak - target_peak)

        # Calculate start and end time of union area
        union_start = torch.min(output[:, :, 1], target[:, :, 0])
        union_end = torch.max(output[:, :, 2], target[:, :, 1])

        # Calculate total length of union area
        union_distance = torch.abs(union_end - union_start)

        # Calculate Peak IoU: traditional IoU minus normalized peak distance
        # Smaller peak distance results in higher Peak IoU
        peak_iou = iou - (peak_distance / union_distance)

        return peak_iou

    @staticmethod
    def anchor_box_select(scores):
        """
        Select optimal anchor box combinations from score matrix to ensure one-to-one correspondence between predictions and ground truth.
        Args:
            scores: Matching score matrix between predictions and ground truth, shape [num_predictions, num_targets]
        Returns:
            mask: Binary mask matrix indicating final matching between selected anchors and targets
        """
        # Step 1: Row-wise selection - keep column with max value per row, set others to 0
        max_row, _ = torch.max(scores, dim=-1)  # Calculate max value per row [num_predictions]
        row_mask = torch.tensor((scores == max_row.unsqueeze(1))).float()  # Create row mask matrix
        masked_tensor = scores * row_mask  # Set non-max positions to 0 per row

        # Step 2: Column-wise selection - keep row with max value per column from selected row maxima, set others to 0
        max_col, _ = torch.max(masked_tensor, dim=0)  # Calculate max value per column [num_targets]
        col_mask = (masked_tensor == max_col.unsqueeze(0)).float()  # Create column mask matrix

        # Step 3: Combine row and column selection results
        final_tensor = masked_tensor * col_mask  # Apply column mask for further filtering

        # Step 4: Create final binary mask, threshold to eliminate numerical errors
        mask = final_tensor > 1e-9  # Set values above threshold to True, others to False

        return mask

    @staticmethod
    def gaussian(x, mean, std1, std2, height):
        """
        Generate asymmetric Gaussian function values with different standard deviations on both sides of the mean.
        Args:
            x: Input tensor, positions to calculate Gaussian values
            mean: Mean of Gaussian distribution (peak position)
            std1: Left standard deviation (used when x < mean)
            std2: Right standard deviation (used when x >= mean)
            height: Peak height of Gaussian function
        Returns:
            result: Asymmetric Gaussian function values at x
        """
        # Set left and right standard deviations separately
        left_std = std1
        right_std = std2

        # Calculate asymmetric Gaussian values
        if std1 != 0 and std2 != 0:
            # Use left std when x < mean, else use right std
            result = torch.where(
                x < mean,  # Condition judgment
                height * torch.exp(-((x - mean) ** 2) / (2 * left_std ** 2)),  # Gaussian value for x < mean
                height * torch.exp(-((x - mean) ** 2) / (2 * right_std ** 2))  # Gaussian value for x >= mean
            )
        else:
            # Return all-zero tensor if any std is 0 (avoid division by zero)
            result = torch.zeros_like(x)

        return result

    def generate_peak_data(self, parameters, num_points=DATA_LENGTH):
        """
        Generate synthetic data by superimposing multiple asymmetric Gaussian peaks.
        Args:
            parameters: Tensor containing parameters for each Gaussian peak, shape [num_peaks, 5]
                        Each row format: [?, left std, right std, mean, height]
            num_points: Number of points for generated data, default to global constant DATA_LENGTH
        Returns:
            x: Tensor of sampling positions
            total_data: Synthetic data tensor after superimposing all Gaussian peaks
        """
        # Create uniformly distributed sampling points
        x = torch.linspace(0, num_points, num_points)

        # Initialize total data as all-zero vector
        total_data = torch.zeros(num_points)

        # Iterate over parameters of each Gaussian peak
        for param in parameters:
            # Skip all-zero parameters (may indicate invalid peaks)
            if torch.all(param == 0):
                continue

            # Parse parameters: [_, left std, right std, mean, height]
            # First parameter unused (may be confidence or other auxiliary info)
            _, std1, std2, mean, height = param

            # Generate data for current Gaussian peak
            peak_data = self.gaussian(x, mean, std1, std2, height)

            # Superimpose current peak to total data
            total_data += peak_data

        return total_data

    @staticmethod
    def plot_peak_data(x, result):
        """
        Visualize synthetic data generated by superimposing multiple asymmetric Gaussian peaks.
        Args:
            x: Tensor of sampling positions (X-axis coordinates)
            result: Synthetic data tensor after superimposing all Gaussian peaks (Y-axis values)
        """
        # Create 12x4 inch figure window
        plt.figure(figsize=(12, 4))

        # Plot data curve
        # Convert PyTorch tensor to NumPy array using .detach().numpy()
        # Required as matplotlib typically takes NumPy arrays as input
        plt.plot(x.detach().numpy(), result.detach().numpy())

        # Set axis labels and title
        plt.xlabel('X')  # X-axis label
        plt.ylabel('Y')  # Y-axis label
        plt.title('Sum of Peak Data')  # Plot title

        # Display the plot
        plt.show()

    def __call__(self, output, target, sequence):
        """
        Evaluate matching degree between model predictions and ground truth, return multiple performance metrics.
        Args:
            output: Model prediction tensor, shape [num_predictions, 5]
                    Each row format: [confidence, start position, end position, peak position, height]
            target: Ground truth tensor, shape [num_targets, 4]
                    Each row format: [start position, end position, peak position, height]
            sequence: Original sequence data for calculating reconstruction error
        Returns:
            tuple: Tuple containing the following metrics
                - output_num: Number of valid predictions
                - target_num: Number of ground truth targets
                - iou_num: List of matching counts under different IoU thresholds
                - rss: Reconstruction error (Residual Sum of Squares)
        """
        if self.traditional_or_neural == "neural":
            # Step 1: Get true positions and apply NMS or alignment
            output = self.get_true_position(output)  # Process predictions to get true positions

            if self.nms:
                output = self.non_max_suppression(output)  # Apply NMS to filter overlapping predictions
                max_len = PEAK_NUM_MAX  # Maximum peak number limit
            else:
                output, target, max_len = self.align_first_dim(output, target)  # Align dimensions of predictions and targets
        else:
            output = output
            max_len = PEAK_NUM_MAX
        # Step 2: Extract parameters from predictions to generate Gaussian peak synthetic data
        output2 = output.clone()
        output_p, output_start, output_end, output_peak, output_height = torch.split(output2, 1, dim=-1)

        output_mean = output_peak  # Use peak position as mean of Gaussian distribution
        output_height = output_height  # Gaussian peak height
        output_std1 = (output_peak - output_start) / 3  # Left standard deviation
        output_std2 = (output_end - output_peak) / 3  # Right standard deviation

        # Combine parameters and generate synthetic data
        parameters = torch.cat((output_p, output_std1, output_std2, output_mean, output_height), dim=-1)
        pred_sequence = self.generate_peak_data(parameters)  # Generate synthetic sequence with superimposed Gaussian peaks

        # Step 3: Calculate reconstruction error (Residual Sum of Squares)
        rss = torch.sum((pred_sequence - sequence) ** 2)  # Evaluate difference between synthetic and original sequence

        # Step 4: Calculate Peak IoU and select optimal matches
        peak_iou = self.peak_iou_operate(output, target, max_len)  # Calculate Peak IoU between predictions and ground truth
        mask = self.anchor_box_select(peak_iou)  # Select optimal matches to ensure one-to-one correspondence
        output_p = output_p.repeat(1, max_len)  # [max_len, max_len, 5]

        p_and_peakiou = torch.cat((output_p[mask].unsqueeze(0), peak_iou[mask].unsqueeze(0)), dim=0)

        target_num = (target[:, 0] > 1e-9).sum().cpu().item()  # Number of ground truth targets

        return p_and_peakiou, target_num, rss.item()


class AnAlysis:
    def __init__(self, plr=True, num_analysis=200, nms=True, traditional_or_neural="neural", show_result=True):
        """
        Initialize the analyzer
        Args:
            plr: Reserved parameter (may control PLR-related functionality)
            num_analysis: Number of samples to analyze
            nms: Whether to use Non-Maximum Suppression (NMS) in analysis
        """
        self.plr = plr
        self.num_analysis = num_analysis  # Total number of samples to analyze
        self.nms = nms  # Whether to enable Non-Maximum Suppression
        self.traditional_or_neural = traditional_or_neural
        self.show_result = show_result

    def __call__(self, net, weight_path, data, outputs=None):
        """
        Execute model evaluation process: load model, batch prediction, calculate metrics, summarize results
        Args:
            net: Model network structure
            weight_path: Path to model weight file
            data: Test dataset, each sample contains input, target label and original sequence
        Returns:
            Tuple containing evaluation metrics such as precision, recall, AP
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Ignore warning messages

            # Load model to CPU
            if net:
                self.net = net.to('cpu')
                self.net.eval()  # Set to evaluation mode
                self.net.load_state_dict(torch.load(weight_path, map_location='cpu'))  # Load weights
            else:
                self.net = None

            p_and_peakiou = torch.empty(2, 0)
            target_num = 0
            rss = 0

            for i in range(self.num_analysis):
                inputs, target, sequence = data[i]  # Get sample data
                inputs = inputs.unsqueeze(0)  # Add batch dimension
                output = self.net(inputs) if self.traditional_or_neural == "neural" else outputs[i]  # Model prediction
                # Call analysis tool to calculate single-sample metrics
                analysis = Analysis(nms=self.nms, traditional_or_neural=self.traditional_or_neural)
                p_and_peakiou_one, target_num_one, rss_one = analysis(output, target, sequence)
                p_and_peakiou = torch.cat((p_and_peakiou, p_and_peakiou_one), dim=-1)
                target_num = target_num + target_num_one
                rss = rss + rss_one

            # Sort first row and return sorted values and original indices
            sorted_values, sorted_indices = torch.sort(p_and_peakiou[0, :], descending=True)

            # Rearrange two rows by sorted indices
            p_and_peakiou = p_and_peakiou[:, sorted_indices]

            iou_limit = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # COCO standard IoU thresholds

            total_gt = target_num  # Total number of ground truth targets (accumulated in loop)
            num_preds = p_and_peakiou.shape[1]  # Total number of valid prediction boxes across all samples
            ap_list = []  # Store AP values for 10 IoU thresholds

            # 2. Iterate over each IoU threshold to calculate corresponding AP
            for iou_thresh in iou_limit:
                # 2.1 Dynamically count TP/FP (judge one by one in descending order of confidence)
                cumulative_tp = 0  # Cumulative TP count
                cumulative_fp = 0  # Cumulative FP count
                precisions = []  # Store precision for each prediction box
                recalls = []  # Store recall for each prediction box

                # Iterate over all prediction boxes in descending order of confidence (p_and_peakiou already sorted by first row)
                for idx in range(num_preds):
                    conf = p_and_peakiou[0, idx].item()  # Confidence of current prediction box
                    peak_iou = p_and_peakiou[1, idx].item()  # Peak IoU between current prediction and GT

                    # Judge if current prediction is TP or FP
                    if peak_iou >= iou_thresh and conf > 1e-9:  # IoU meets threshold and confidence is valid → TP
                        cumulative_tp += 1
                    else:  # IoU not met or confidence invalid → FP
                        cumulative_fp += 1

                    # Calculate current precision and recall
                    total = cumulative_tp + cumulative_fp
                    precision = cumulative_tp / total if total > 0 else 0.0  # Avoid division by zero
                    recall = cumulative_tp / total_gt if total_gt > 0 else 0.0  # Avoid division by zero

                    precisions.append(precision)
                    recalls.append(recall)

                # 2.2 Process PR curve (complete head/tail + calculate upper envelope to eliminate sawtooth)
                # Complete PR curve head and tail points (ensure coverage of [0,0] and [1,0])
                recalls = np.concatenate(([0.0], recalls, [1.0]))
                precisions = np.concatenate(([0.0], precisions, [0.0]))

                # Calculate upper envelope (take max precision from back to front to ensure monotonically non-increasing curve)
                for j in range(len(precisions) - 2, -1, -1):
                    precisions[j] = np.maximum(precisions[j], precisions[j + 1])

                # 2.3 Integrate to calculate AP (trapezoidal integration method)
                # Find recall change points (avoid repeated calculation of intervals with same recall)
                change_points = np.where(recalls[1:] != recalls[:-1])[0]
                ap = 0.0
                for j in change_points:
                    # Trapezoidal area = recall interval length × (current precision + next precision) / 2
                    recall_diff = recalls[j + 1] - recalls[j]
                    precision_avg = (precisions[j] + precisions[j + 1]) / 2
                    ap += recall_diff * precision_avg

                ap_list.append(ap)  # Save AP for current IoU threshold

            # 3. Calculate final metrics
            # 3.1 Extract AP for each threshold (AP50~AP95)
            ap50 = ap_list[0]  # AP for IoU=0.5
            ap55 = ap_list[1]  # AP for IoU=0.55
            ap60 = ap_list[2]  # AP for IoU=0.6
            ap65 = ap_list[3]  # AP for IoU=0.65
            ap70 = ap_list[4]  # AP for IoU=0.7
            ap75 = ap_list[5]  # AP for IoU=0.75
            ap80 = ap_list[6]  # AP for IoU=0.8
            ap85 = ap_list[7]  # AP for IoU=0.85
            ap90 = ap_list[8]  # AP for IoU=0.9
            ap95 = ap_list[9]  # AP for IoU=0.95

            # 3.2 Calculate AP50-95 (average of 10 AP values)
            ap50_95 = sum(ap_list) / len(ap_list)

            # 3.3 Calculate overall precision and recall (based on IoU=0.5 threshold, reference standard evaluation)
            # Re-count total TP/FP for IoU=0.5
            total_tp_50 = sum(
                1 for idx in range(num_preds) if p_and_peakiou[1, idx] >= 0.5 and p_and_peakiou[0, idx] > 1e-9)
            total_fp_50 = num_preds - total_tp_50
            overall_precision = total_tp_50 / (total_tp_50 + total_fp_50) if (total_tp_50 + total_fp_50) > 0 else 0.0
            overall_recall = total_tp_50 / total_gt if total_gt > 0 else 0.0

            # 4. Print all evaluation results
            if self.show_result:
                print("=" * 50)
                print("AP Calculation Results (IoU 0.5~0.95):")
                print(f"AP50: {ap50:.4f}, AP55: {ap55:.4f}, AP60: {ap60:.4f}, AP65: {ap65:.4f}, AP70: {ap70:.4f}")
                print(f"AP75: {ap75:.4f}, AP80: {ap80:.4f}, AP85: {ap85:.4f}, AP90: {ap90:.4f}, AP95: {ap95:.4f}")
                print(f"AP50: {ap50:.4f}")
                print(f"AP50-95 (Average AP): {ap50_95:.4f}")
                print(f"Overall Precision (IoU=0.5): {overall_precision:.4f}")
                print(f"Overall Recall (IoU=0.5): {overall_recall:.4f}")
                print(f"Average Reconstruction Error (RSS): {rss / self.num_analysis:.4f}")
                print("=" * 50)

            # 5. Return results (compatible with original code evaluation process)
            return [overall_precision], [overall_recall], ap50, ap50_95, ap_list, rss / self.num_analysis