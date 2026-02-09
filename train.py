import torch.nn
import warnings
from nni.compression.utils.counter import count_flops_params
from analysis import AnAlysis
from Net import Net, CNNNet
from loss import Loss_Calculate
from dataset import *
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import time
import os
import random
import numpy as np
import warnings

warnings.simplefilter('ignore')  # Ignore warning messages
# Set environment variables (disable TensorFlow's oneDNN optimization which may conflict with PyTorch)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def set_seed(seed):
    """Set random seed to ensure experimental reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class Train:
    """
    Model training class responsible for the complete process of network training, validation and testing
    Supports comparative experiments with multiple model configuration combinations
    """

    def __init__(self, epochs=100, peak_iou=True, nms=True, three_head=True, dfl=True, bce_or_mse='bce', cnnnet=False,
                 num_train=50000, num_test=20000, num_validation=10000):
        """
        Initialize the trainer
        Parameters:
            epochs: Number of training epochs
            peak_iou: Whether to use Peak IoU loss
            nms: Whether to use Non-Maximum Suppression
            three_head: Whether to use three-detector-head structure
            dfl: Whether to use Distribution Focal Loss
            bce_or_mse: Classification loss type (BCE or MSE)
            num_train: Size of training dataset
            num_test: Size of test dataset
            num_validation: Size of validation dataset
        """
        # Set device (GPU or CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load datasets
        self.train_set = DataSet(num_samples=num_train, data_type='train')
        self.test_set = DataSet(num_samples=num_test, data_type='test')
        self.validation_set = DataSet(num_samples=num_validation, data_type='validation')
        self.num_test = num_test
        self.num_validation = num_validation

        # Create data loaders
        self.train_dataloader = DataLoader(self.train_set, batch_size=32, shuffle=True, num_workers=8)
        self.test_dataloader = DataLoader(self.test_set, batch_size=32, shuffle=False)
        self.validation_dataloader = DataLoader(self.validation_set, batch_size=32, shuffle=False)

        # Determine model weight save path based on parameter combinations
        if peak_iou:
            bce_or_mse = 'none'  # Do not use BCE/MSE when using Peak IoU
        self.best_weight_path = f"params/net_best_{peak_iou}_{nms}_{three_head}_{dfl}_{bce_or_mse}.pth"
        self.new_weight_path = f"params/net_new_{peak_iou}_{nms}_{three_head}_{dfl}_{bce_or_mse}.pth"

        if cnnnet:
            self.best_weight_path = f"params/net_best_cnnnet.pth"
            self.new_weight_path = f"params/net_new_cnnnet.pth"

        # Initialize model and move to device
        self.net = Net(three_head=three_head, dfl=dfl).to(self.device)

        if cnnnet:
            self.net = CNNNet().to(self.device)

        # Calculate model FLOPs and parameters (for performance analysis)
        x = torch.randn(2, 1, 1024)
        count_flops_params(self.net, x)

        # Load pre-trained weights (if exist)
        if os.path.exists(self.best_weight_path):
            pass  # Comment out loading operation to avoid overwriting existing training

        # Initialize optimizer (using Stochastic Gradient Descent)
        self.net_optim = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        # Initialize loss function
        self.loss_fn = Loss_Calculate(peak_iou=peak_iou, bce_or_mse=bce_or_mse)

        # Training configuration
        self.epochs = epochs
        self.peak_iou = peak_iou
        self.nms = nms
        self.file_name = f"results/results_{peak_iou}_{nms}_{three_head}_{dfl}_{bce_or_mse}.txt"
        if cnnnet:
            self.file_name = f"results/results_cnnnet.txt"

    def __call__(self):
        """Execute complete training, testing and validation process"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Ignore warnings

            # Initialize best metrics
            best_map50, best_map50_95, best_rss = 0, 0, 0
            best_ap = []
            best_pre = []
            best_rec = []
            net_loss_train = 0

            # Start training loop
            for epoch in range(self.epochs):
                self.net.train()  # Set to training mode
                self.net = self.net.to(self.device)  # Ensure model is on correct device

                # Train for one epoch
                for step, (sequences, targets, _) in tqdm(enumerate(self.train_dataloader),
                                                          desc="Epoch {} Training".format(epoch + 1),
                                                          total=len(self.train_dataloader)):
                    # Move data to device
                    sequences, targets = torch.tensor(sequences).to(self.device), torch.tensor(targets).to(self.device)

                    # Forward propagation
                    outputs = self.net(sequences)
                    loss = self.loss_fn(outputs, targets)

                    # Backward propagation and optimization
                    self.net_optim.zero_grad()
                    loss.backward()
                    self.net_optim.step()

                    net_loss_train += loss.item()

                # Calculate average training loss
                net_loss_train = net_loss_train / len(self.train_dataloader)
                print("Network loss: {}".format(net_loss_train))
                time.sleep(1)  # Short pause for easier output viewing

                # Save current model weights
                torch.save(self.net.state_dict(), self.new_weight_path)

                # Evaluate model on test set
                with torch.no_grad():
                    self.net.eval()  # Set to evaluation mode
                    print("Epoch {} Testing".format(epoch + 1))
                    analysis = AnAlysis(nms=self.nms, num_analysis=self.num_test)
                    pre, rec, map50, map50_95, ap, rss = analysis(self.net, self.new_weight_path, self.test_set)

                # Record current epoch training results to file
                with open(self.file_name, 'a') as file:
                    file.write(f'Epoch {epoch + 1}\n')
                    file.write(f'train_loss: {net_loss_train}\n')
                    file.write(f"pre: {pre}, rec: {rec}\n")
                    file.write(f"ap: {ap}\n")
                    file.write(f"map50: {map50}, map50_95: {map50_95}\n")
                    file.write(f"rss: {rss}\n")

                # Save as best model if current performance on test set is better
                if map50_95 >= best_map50_95:
                    torch.save(self.net.state_dict(), self.best_weight_path)
                    best_map50, best_map50_95, best_rss = map50, map50_95, rss
                    best_ap = ap[:]
                    best_pre = pre[:]
                    best_rec = rec[:]

            # Record best test results
            with open(self.file_name, 'a') as file:
                file.write('Best test results\n')
                file.write(f"best_pre: {best_pre}, best_rec: {best_rec}\n")
                file.write(f"best_ap: {best_ap}\n")
                file.write(f"best_map50: {best_map50}, best_map50_95: {best_map50_95}\n")
                file.write(f"best_rss: {best_rss}\n")

            with torch.no_grad():
                self.net.eval()  # Set to evaluation mode
                # Evaluate best model on validation set
                analysis = AnAlysis(nms=self.nms, num_analysis=self.num_validation)
                print('Validation results')
                pre, rec, map50, map50_95, ap, rss = analysis(self.net, self.best_weight_path, self.validation_set)

            # Record validation results
            with open(self.file_name, 'a') as file:
                file.write('Validation results\n')
                file.write(f"pre: {pre}, rec: {rec}\n")
                file.write(f"ap: {ap}\n")
                file.write(f"map50: {map50}, map50_95: {map50_95}\n")
                file.write(f"rss: {rss}\n")


if __name__ == "__main__":
    """Main function: execute multiple groups of comparative experiments"""
    set_seed(0)  # Set random seed to ensure experimental reproducibility

    train = Train(peak_iou=False, cnnnet=True)
    train()

    # Define multiple groups of experimental parameter combinations for comparing performance of different model configurations
    param = {
        1: {'peak_iou': False, 'nms': False, 'three_head': False, 'dfl': False, 'bce_or_mse': 'mse'},
        2: {'peak_iou': False, 'nms': False, 'three_head': False, 'dfl': False, 'bce_or_mse': 'bce'},
        3: {'peak_iou': True, 'nms': False, 'three_head': False, 'dfl': False, 'bce_or_mse': 'none'},
        4: {'peak_iou': True, 'nms': True, 'three_head': False, 'dfl': False, 'bce_or_mse': 'none'},
        5: {'peak_iou': True, 'nms': True, 'three_head': False, 'dfl': True, 'bce_or_mse': 'none'},
        6: {'peak_iou': True, 'nms': True, 'three_head': True, 'dfl': False, 'bce_or_mse': 'none'},
        7: {'peak_iou': True, 'nms': True, 'three_head': True, 'dfl': True, 'bce_or_mse': 'none'},
    }

    # Execute each group of experiments in sequence
    for i in range(7):
        selected_number = i + 1
        params = param[selected_number]
        train = Train(**params)  # Initialize trainer using parameter unpacking
        train()  # Execute training and evaluation
