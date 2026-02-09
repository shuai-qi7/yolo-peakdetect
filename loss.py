import torch.nn
from dataset import *
from utils import Utils
import warnings
warnings.simplefilter('ignore')  # Ignore warning messages

class Loss_Calculate:
    """
    Loss calculation class for computing the loss function of peak detection models
    Supports two calculation methods: Peak-IoU loss and traditional MSE/BCE loss
    """

    def __init__(self, peak_iou=True, bce_or_mse='bce'):
        """
        Initialize the loss calculator
        Args:
            peak_iou (bool): Whether to use Peak-IoU loss (loss based on peak intersection over union)
            bce_or_mse (str): Basic loss type, 'bce' (Binary Cross Entropy) or 'mse' (Mean Squared Error)
        """
        super(Loss_Calculate, self).__init__()
        self.factor_p = 1  # Weight factor for confidence loss
        self.factor_diou = 15  # Weight factor for Peak-IoU loss
        # Offset value of anchor points relative to grid cells (0.5 means anchor point is at grid center)
        self.grid_offset = 0.5
        self.peak_iou = peak_iou  # Whether to enable Peak-IoU loss
        self.bce_or_mse = bce_or_mse  # Type of basic loss function

    def outputs_and_targets_transform(self, outputs, targets):
        """
        Transform the shape and format of model outputs and target labels to match for loss calculation

        Args:
            outputs: Original output tensor of the model
            targets: Ground truth target label tensor

        Returns:
            Transformed output and target tensors (matched shape, can be directly used for loss calculation)
        """
        # Apply grid offset transformation to adjust coordinate format in outputs
        outputs = Utils.transform_output(outputs, grid_offset=self.grid_offset)

        # Add a dimension to the third axis of outputs and repeat to match maximum peak count (PEAK_NUM_MAX)
        outputs = outputs.unsqueeze(2).repeat(1, 1, PEAK_NUM_MAX, 1)

        # Calculate width of target regions, handle abnormal cases where width is 0 (avoid division by zero error)
        targets_width = targets[:, :, 1] - targets[:, :, 0]
        mask = targets_width > 0  # Filter valid targets (width > 0)
        targets_width[~mask] = 0.01  # Set width of invalid targets to 0.01

        # Calculate relative position ratio of peaks in target regions (offset of peak position from target start / target width)
        targets_pkl = (targets[:, :, 2] - targets[:, :, 0]) / targets_width
        targets_pkl = targets_pkl.unsqueeze(-1)  # Add dimension for concatenation
        targets = torch.cat((targets, targets_pkl), dim=-1)  # Concatenate ratio information to targets

        # Add a dimension to the second axis of targets and repeat to match feature map size of outputs
        targets = targets.unsqueeze(1).repeat(1, outputs.shape[1], 1, 1)

        return outputs, targets

    @staticmethod
    def peak_iou_loss_operate(outputs, targets):
        """
        Calculate Peak-IoU related loss components, including Intersection over Union (IoU) and peak distance penalty

        Args:
            outputs: Transformed model outputs (containing predicted confidence, coordinates, peaks, etc.)
            targets: Transformed ground truth targets (containing ground truth coordinates, peaks, etc.)

        Returns:
            Processed outputs, targets, and scores for each prediction box (used to select optimal matches)
        """
        # Separate and remove anchor point information from outputs (retain last n-1 dimensions)
        _, outputs = torch.split(outputs, [1, outputs.size(-1) - 1], dim=-1)

        # Calculate intersection region between predicted and ground truth boxes
        intersection_start = torch.max(outputs[:, :, :, 1], targets[:, :, :, 0])  # Start position of intersection (take maximum of both)
        intersection_end = torch.min(outputs[:, :, :, 2], targets[:, :, :, 1])  # End position of intersection (take minimum of both)
        # Intersection length (ensure non-negative, 0 when no intersection)
        intersection_length = torch.clamp(intersection_end - intersection_start, min=0.0)
        # Intersection height (take minimum of both heights)
        intersection_height = torch.min(outputs[:, :, :, 4], targets[:, :, :, 3])
        intersection_area = intersection_length * intersection_height  # Intersection area

        # Calculate areas of predicted and ground truth boxes
        pred_area = (outputs[:, :, :, 2] - outputs[:, :, :, 1]) * outputs[:, :, :, 4]  # Predicted box area
        gt_area = (targets[:, :, :, 1] - targets[:, :, :, 0]) * targets[:, :, :, 3]  # Ground truth box area
        union_area = pred_area + gt_area - intersection_area  # Union area (sum of areas minus intersection)

        # Calculate traditional Intersection over Union (IoU)
        iou = intersection_area / union_area

        # Calculate Peak-IoU (subtract normalized penalty of peak distance from IoU)
        outputs_peak = outputs[:, :, :, 3]  # Predicted peak position
        targets_peak = targets[:, :, :, 2]  # Ground truth peak position
        peak_distance = torch.abs(outputs_peak - targets_peak)  # Absolute distance between peaks

        # Calculate total length of union region (used to normalize peak distance)
        union_start = torch.min(outputs[:, :, :, 1], targets[:, :, :, 0])
        union_end = torch.max(outputs[:, :, :, 2], targets[:, :, :, 1])
        union_distance = torch.abs(union_end - union_start)  # Length of union region

        # Peak-IoU = IoU - (peak distance / union length)
        peak_iou = iou - (peak_distance / union_distance)

        # Handle abnormal values (ensure Peak-IoU is within [0,1], otherwise set to 0.01)
        peak_iou = torch.where(
            (peak_iou >= 0) & (peak_iou <= 1),
            peak_iou,
            torch.tensor(0.01, device=peak_iou.device)
        )

        # Concatenate IoU and Peak-IoU to outputs for subsequent calculation
        peak_iou = peak_iou.unsqueeze(-1)
        iou = iou.unsqueeze(-1)
        outputs = torch.cat((outputs, iou, peak_iou), dim=-1)

        # Calculate score for each prediction box (Peak-IoU × confidence)
        scores = peak_iou[:, :, :, 0] * outputs[:, :, :, 0]
        scores = scores.unsqueeze(-1)  # Add dimension

        return outputs, targets, scores

    @staticmethod
    def anchor_box_select(scores):
        """
        Select optimal prediction boxes (anchor boxes) based on scores, ensuring each ground truth target matches a unique prediction box

        Args:
            scores: Score matrix of prediction boxes (reflecting matching degree between prediction boxes and targets)

        Returns:
            max_row: Maximum score of each row (used to filter low-quality predictions)
            mask: Binary mask indicating selected prediction boxes (1 for selected, 0 for unselected)
        """
        scores = scores.squeeze(-1)  # Remove last dimension (dimension compression)
        max_row, _ = torch.max(scores, dim=-1)  # Calculate maximum score of each row (along target dimension)
        # Create row mask: only the position of maximum score in each row is 1, others are 0
        row_mask = torch.tensor((scores == max_row.unsqueeze(2))).float()
        masked_tensor = scores * row_mask  # Apply row mask to retain maximum value of each row

        max_col, _ = torch.max(masked_tensor, dim=1)  # Calculate maximum score of each column (along prediction dimension)
        # Create column mask: only the position of maximum score in each column is 1, others are 0
        col_mask = (masked_tensor == max_col.unsqueeze(1)).float()
        final_tensor = masked_tensor * col_mask  # Apply column mask to ensure each target matches only one prediction

        mask = final_tensor > 0  # Generate final mask (positions with score > 0 are selected prediction boxes)
        return max_row, mask

    def __call__(self, outputs, targets):
        """
        Calculate total loss (main function)

        Args:
            outputs: Model output tensor
            targets: Ground truth target label tensor

        Returns:
            Total loss value (scalar tensor)
        """
        if self.peak_iou:
            # Step 1: Transform format of outputs and targets to ensure shape matching
            outputs, targets = self.outputs_and_targets_transform(outputs, targets)

            # Step 2: Calculate Peak-IoU and scores
            outputs, targets, scores = self.peak_iou_loss_operate(outputs, targets)

            # Step 3: Select optimally matched prediction boxes
            max_row, mask = self.anchor_box_select(scores)

            # Step 4: Calculate Peak-IoU loss (1 - average Peak-IoU, multiplied by weight factor)
            peak_iou_iou_loss = 1 - torch.mean(outputs[:, :, :, -1][mask])
            peak_iou_iou_loss = peak_iou_iou_loss * self.factor_diou

            # Step 5: Calculate confidence loss (using BCE loss)
            # Confidence of low-quality predictions (score < 0.5) should be close to 0
            mask_non = max_row < 0.5
            loss_p_fn = torch.nn.BCELoss()
            # Negative sample loss: difference between confidence of low-quality predictions and 0
            p_loss1 = loss_p_fn(
                outputs[:, :, :, 0][mask_non],
                torch.zeros_like(outputs[:, :, :, 0][mask_non])
            )
            # Positive sample loss: difference between confidence of high-quality predictions and IoU
            p_loss2 = loss_p_fn(
                outputs[:, :, :, 0][mask],
                outputs[:, :, :, 6][mask]  # outputs[:, :, :, 6] is IoU value
            )
            p_loss = (p_loss1 + p_loss2) * self.factor_p  # Total confidence loss (multiplied by weight)

            # Total loss = Peak-IoU loss + confidence loss
            loss = peak_iou_iou_loss + p_loss

        else:
            # When not using Peak-IoU, directly use MSE or BCE loss
            loss_fn = torch.nn.MSELoss() if self.bce_or_mse == 'mse' else torch.nn.BCELoss()
            targets_tensor = Utils.transform_target(targets)  # Transform target format
            mask = targets_tensor[:,:,0]>0.5
            loss = loss_fn(outputs[mask], targets_tensor[mask])+loss_fn(outputs[:,:,0],targets_tensor[:,:,0])  # Calculate basic loss

        return loss


if __name__ == "__main__":
    # ===================== Test Peak-IoU loss calculation and backpropagation =====================
    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 2. Load data
    from dataset import DataSet
    data = DataSet(num_samples=3, three_channel=False, data_type='test')

    sequences_list = []
    targets_list = []
    for i in range(3):
        sequence, targets, _ = data[i]
        sequences_list.append(sequence)
        targets_list.append(targets)
    # 1. Prepare data
    sequences = torch.stack(sequences_list).to(device)
    targets = torch.stack(targets_list).to(device)
    print(f"input_shape: {sequences.shape}, Label_shape: {targets.shape}")
    # 2. Load the model (and mark parameters to compute gradients)
    from Net import Net
    net = Net(False, False, False).to(device)
    # Ensure all model parameters can compute gradients
    for param in net.parameters():
        param.requires_grad = True
    # 2. Model forward propagation
    outputs = net(sequences)
    # 3. Initialize loss calculator
    loss_calculator = Loss_Calculate(peak_iou=True)
    # 4. Calculate Peak-IoU loss
    print("\n=== Start calculating Peak-IoU loss ===")
    loss = loss_calculator(outputs, targets)
    print(f"Total Peak-IoU loss value: {loss.item():.4f}")
    # 5. Verify backpropagation (Core: Check gradients of model parameters)
    print("\n=== Verify backpropagation capability ===")
    # Clear existing gradients of the model
    net.zero_grad()
    # Perform backpropagation
    loss.backward()
    # Check if model parameters have valid gradients (Core verification)
    has_grad = any(param.grad is not None and param.grad.sum() != 0 for param in net.parameters())
    if has_grad:
        print("✅ Backpropagation successful: Model parameter gradients are generated")
        # Print gradient info of the first parameter (example)
        first_param = next(net.parameters())
        print(f"Gradient shape of the first model parameter: {first_param.grad.shape}")
        print(f"Gradient mean value of the first model parameter: {first_param.grad.mean().item():.6f}")
    else:
        print("❌ Backpropagation failed: No valid gradients for model parameters")