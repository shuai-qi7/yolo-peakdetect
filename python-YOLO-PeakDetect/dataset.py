import torch
import numpy as np
import math
from torch.utils.data import Dataset
from config import *
import matplotlib.pyplot as plt
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -----------------------------------------------#
#               数据集的产生                       #
#           对称高斯峰与非对称高斯峰                 #
#               噪声和基线漂移                     #
#               生成序列和标签                     #
# -----------------------------------------------#

def set_seed(seed):
    """设置随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class DataSet(Dataset):
    """生成包含多个高斯峰的合成数据集，用于峰值检测模型的训练和评估"""

    def __init__(self, num_samples=10000, three_channel=False, data_type='predict'):
        """
        初始化数据集生成器

        参数:
            num_samples: 生成的样本数量
            three_channel: 是否生成三通道数据（原始数据、一阶差分、二阶差分）
            data_type: 数据集类型（训练/测试/验证/预测），用于设置随机种子
        """
        super(DataSet, self).__init__()
        self.num_samples = num_samples  # 样本数量
        self.three_channel = three_channel  # 是否使用三通道
        self.sequence_length = DATA_LENGTH  # 序列长度
        self.num_peaks_range = (1, PEAK_NUM_MAX)  # 每个样本中峰的数量范围
        self.amplitude_range = (0.01, 0.9)  # 峰高范围
        self.location_range = (0.05 * self.sequence_length, 0.95 * self.sequence_length)  # 峰位置范围
        self.std_range = (0.0015 * self.sequence_length, 0.04 * self.sequence_length)  # 标准差范围
        self.noise_level = self.amplitude_range[0] / 20  # 噪声水平
        self.amplitude_max = 1  # 序列幅值上限
        self.multiplier_range = (0.05, 0.2)  # 基线漂移乘数范围
        self.num_sigmoid = 10

        # 根据数据集类型设置随机种子，确保结果可复现
        if data_type == 'train':        set_seed(0)
        if data_type == 'test':         set_seed(1)
        if data_type == 'validation':   set_seed(2)

        self.multi_gaussian_params = self.generate_random_list()

    def __len__(self):
        """返回数据集的样本数量"""
        return self.num_samples

    def generate_random(self):
        """
        生成随机高斯峰参数

        返回:
            locations: 峰位置列表
            amplitudes: 峰高列表
            left_stds: 左侧标准差列表
            right_stds: 右侧标准差列表
        """
        # 随机确定最大幅值和最大的标准差
        amplitude_max = np.random.uniform(*self.amplitude_range)
        amplitude_range = (self.amplitude_range[0], amplitude_max)
        std_max = np.random.uniform(*self.std_range)
        std_min = self.std_range[0]

        # 确定峰的数量（避免过于密集）
        num_gaussian = np.random.randint(1, min(PEAK_NUM_MAX, math.floor((self.sequence_length * 0.9) / (6 * std_max))))

        # 随机生成标准差对（对称或非对称）
        pairs = []
        choice = np.random.choice([1, 2, 3])
        if choice == 1:
            pairs.append((std_max, std_max))  # 对称峰
        elif choice == 2:
            pairs.append((std_max, np.random.uniform(max(std_min, std_max / 4), std_max)))  # 右偏峰
        else:
            pairs.append((np.random.uniform(max(std_min, std_max / 4), std_max), std_max))  # 左偏峰

        # 生成剩余的峰参数（部分对称，部分非对称）：修复high<=0问题
        if num_gaussian == 1:
            # 仅1个峰时，无剩余峰需补充，对称峰数量直接设为0
            equal_num_gaussian = 0
        else:
            # 峰数量>1时，正常随机生成对称峰数量（范围0 ~ num_gaussian-1）
            equal_num_gaussian = np.random.randint(0, num_gaussian - 1)

        unequal_num_gaussian = num_gaussian - 1 - equal_num_gaussian

        # 生成非对称峰参数
        for _ in range(unequal_num_gaussian):
            left_val = np.random.uniform(std_min, std_max)
            right_val = np.random.uniform(max(std_min, left_val / 4), min(std_max, left_val * 4))
            pairs.append((left_val, right_val))

        # 生成对称峰参数
        for _ in range(equal_num_gaussian):
            equal_val = np.random.uniform(std_min, std_max)
            pairs.append((equal_val, equal_val))

        # 打乱顺序
        np.random.shuffle(np.array(pairs))
        left_stds = np.array(pairs)[:, 0]
        right_stds = np.array(pairs)[:, 1]

        # 生成峰高（确保至少有一个峰达到最大幅值）
        amplitudes = [np.random.uniform(*amplitude_range) for _ in range(num_gaussian - 1)]
        amplitudes.append(amplitude_max)

        # 生成不重叠的峰位置
        locations = []
        for i in range(num_gaussian):
            while True:
                # 计算候选位置范围（避免太靠近序列边缘）
                location_range = (max(self.location_range[0], float(left_stds[i]) * 3),
                                  min(self.location_range[1], self.sequence_length - right_stds[i] * 3))
                location = np.random.uniform(*location_range)

                # 检查与已存在的峰是否重叠
                is_valid = True
                for j in range(i):
                    min_left_width = (left_stds[j] + left_stds[i]) * 3 / 2
                    min_right_width = (right_stds[j] + right_stds[i]) * 3 / 2
                    lower_bound = locations[j] - min_left_width
                    upper_bound = locations[j] + min_right_width
                    if lower_bound <= location <= upper_bound:
                        is_valid = False
                        break

                if is_valid:
                    locations.append(location)
                    break

        sigmoid_params = []
        for _ in range(self.num_sigmoid):
            multiplier = np.random.uniform(0.05, 0.95 - max(amplitudes))
            a = np.random.uniform(-20, 20)  # 斜率
            b = np.random.uniform(-20, 20)  # 偏移
            sigmoid_params.append((a, b, multiplier))
        sigmoid_params = np.array(sigmoid_params, dtype=np.float32)  # 转为数组便于存储

        # 返回完整参数：峰参数 + 噪声 + 基线参数
        peak_params = (locations, amplitudes, left_stds, right_stds)

        # 添加正噪声
        noise = np.random.normal(0, self.noise_level, self.sequence_length).astype(np.float32)
        noise = np.maximum(noise, 0)

        return peak_params, sigmoid_params, noise


    def apply_baseline_drift(self, sequence, sigmoid_params,peak_params):

        def sigmoid(x, a, b, multiplier):
            """Sigmoid函数，用于生成平滑的基线漂移"""
            return 1 / (1 + np.exp(-(x * a + b))) * multiplier

        locations, amplitudes, left_stds, right_stds = peak_params

        # 生成x轴上的点
        x = np.linspace(-1, 1, len(sequence))

        # 初始化基线漂移
        baseline_drift = np.zeros(len(sequence), dtype='float32')

        # 叠加多个sigmoid曲线生成复杂的基线漂移
        for i in range(self.num_sigmoid):
            a,b,multiplier = sigmoid_params[i]
            baseline_drift += sigmoid(x, a, b, multiplier) / self.num_sigmoid

        # 归一化并调整基线漂移的幅度
        baseline_drift = (baseline_drift / max(baseline_drift)) * (0.95 - max(amplitudes))

        return baseline_drift

    def generate_gaussian_sequence(self, gaussian_params):
        """
        生成单个高斯峰序列

        参数:
            gaussian_params: 高斯峰参数 (位置, 幅值, 左标准差, 右标准差)

        返回:
            gaussian: 单个高斯峰序列
        """
        location, amplitude, left_std, right_std = gaussian_params

        # 左侧和右侧使用不同的标准差，生成非对称高斯峰
        left_half = np.exp(-0.5 * ((np.arange(self.sequence_length) - location) / left_std) ** 2)
        right_half = np.exp(-0.5 * ((np.arange(self.sequence_length) - location) / right_std) ** 2)

        # 组合左右两部分
        gaussian = np.where((np.arange(self.sequence_length)) < location, left_half, right_half)
        gaussian *= amplitude

        return gaussian

    def generate_multi_gaussian_sequence(self, multi_gaussian_params):
        """
        生成多个高斯峰叠加的序列，并添加噪声和基线漂移

        参数:
            multi_gaussian_params: 多个高斯峰的参数

        返回:
            sequence: 添加噪声和基线漂移后的序列
            sequence_without_noise_baseline: 未添加噪声和基线漂移的序列
        """
        # 初始化序列
        sequence = np.zeros(self.sequence_length)

        # 叠加所有高斯峰
        for gaussian_param in zip(*multi_gaussian_params):
            sequence += self.generate_gaussian_sequence(gaussian_param)

        # 保存未添加噪声和基线漂移的序列（用于评估）
        sequence_without_noise_baseline = np.copy(sequence)

        return sequence, sequence_without_noise_baseline

    def generate_sequence_target(self, multi_gaussian_params):
        """
        生成序列的目标标签

        参数:
            multi_gaussian_params: 多个高斯峰的参数

        返回:
            targets: 目标标签，包含每个峰的起始位置、结束位置、峰值位置和高度
        """
        # 初始化目标标签（最多支持PEAK_NUM_MAX个峰）
        targets = np.zeros((self.num_peaks_range[1], 4), dtype=float)

        # 为每个峰生成标签
        for i, (location, amplitude, left_std, right_std) in enumerate(zip(*multi_gaussian_params)):
            x_start = location - 3 * left_std  # 起始位置
            x_end = location + 3 * right_std  # 结束位置
            x_peak = location  # 峰值位置
            height = amplitude / self.amplitude_max  # 归一化的峰高

            targets[i][0:4] = [x_start, x_end, x_peak, height]

        return torch.tensor(targets)

    def generate_random_list(self):
        multi_gaussian_params = []

        for idx in range(self.num_samples):
            multi_gaussian_param = self.generate_random()
            multi_gaussian_params.append(multi_gaussian_param)

        return multi_gaussian_params

    def __getitem__(self, index):
        """
        获取单个样本

        参数:
            index: 样本索引

        返回:
            sequence: 输入序列（可能是三通道：原始数据、一阶差分、二阶差分）
            targets: 目标标签
            sequence_without_noise_baseline: 未添加噪声和基线漂移的序列
        """
        # 生成随机高斯峰参数
        multi_gaussian_param,sigmoid_param,noise = self.multi_gaussian_params[index]

        # 生成序列（添加噪声和基线漂移）
        sequence, sequence_without_noise_baseline = self.generate_multi_gaussian_sequence(multi_gaussian_param)


        sequence += noise

        # 添加基线漂移
        sequence += self.apply_baseline_drift(sequence, sigmoid_param, multi_gaussian_param)

        # 生成目标标签
        targets = self.generate_sequence_target(multi_gaussian_param)

        # 转换为PyTorch张量并调整形状
        sequence = sequence.reshape(1, len(sequence))
        sequence = torch.tensor(sequence).float()

        # 计算一阶差分和二阶差分
        dx = sequence[:, 1:] - sequence[:, :-1]
        dx_pad = torch.cat((torch.zeros_like(dx[:, 0].unsqueeze(-1)), dx), dim=-1)

        ddx = dx[:, 1:] - dx[:, :-1]
        ddx_pad = torch.cat((torch.zeros_like(ddx[:, 0].unsqueeze(-1)), ddx, torch.zeros_like(ddx[:, 0].unsqueeze(-1))),
                            dim=-1)

        # 是否使用三通道数据
        if self.three_channel:
            sequence = torch.cat((sequence, dx_pad, ddx_pad), dim=0)

        return sequence, targets, torch.tensor(sequence_without_noise_baseline)


if __name__ == "__main__":
    """测试数据集生成器"""
    data = DataSet(num_samples=3, three_channel=False, data_type='test')

    # 可视化生成的样本
    for i in range(3):
        sequence, targets, _ = data[i]
        sequence = sequence.squeeze(0)

        print(f"序列形状: {sequence.shape}")
        print(f"目标标签形状: {targets.shape}")
        print(f"目标标签内容:\n{targets}")

        # 绘制序列
        plt.figure(figsize=(16, 3))
        plt.plot(sequence)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Multi-Gaussian Sequence')
        plt.ylim(0, 1)
        plt.show()

    a = sequence

    # 可视化生成的样本
    for i in range(3):
        sequence, targets, _ = data[i]
        sequence = sequence.squeeze(0)

        print(f"序列形状: {sequence.shape}")
        print(f"目标标签形状: {targets.shape}")
        print(f"目标标签内容:\n{targets}")

        # 绘制序列
        plt.figure(figsize=(16, 3))
        plt.plot(sequence)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Multi-Gaussian Sequence')
        plt.ylim(0, 1)
        plt.show()

    b = sequence

    print(max(a-b))
