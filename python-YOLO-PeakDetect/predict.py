import os.path

import torch

from Net import *
from utils import Utils
import warnings
import matplotlib.pyplot as plt
import numpy as np
from config import *

# 定义绘图使用的颜色列表（用于区分不同的峰值）
colors = [
    '#FF5733',  # 橙色
    '#33FF57',  # 青绿色
    '#3357FF',  # 蓝色
    '#F44336',  # 红色（更深）
    '#2196F3',  # 蓝色（更深）
    '#FFEB3B',  # 黄色
    '#4CAF50',  # 绿色
    '#9C27B0',  # 紫色
    '#E91E63',  # 粉色
    '#03DAC5',  # 青蓝色
    '#FF9800',  # 橘黄色
    '#795548',  # 棕色
    '#607D8B',  # 蓝灰色
    '#FFC107',  # 金黄色
    '#8BC34A',  # 浅绿色
    '#CDDC39',  # 黄绿色
    '#FF5252',  # 亮红色
    '#3F51B5',  # 藏青色
    '#673AB7',  # 深紫色
    '#00BCD4',  # 浅蓝色
    '#FF33E7',  # 紫红色
    '#009688',  # 蓝绿色
    '#FF6F00',  # 橙黄色
    '#3949AB',  # 深蓝色
    '#8E24AA',  # 深紫红色
    '#00C853',  # 鲜绿色
    '#FFD54F',  # 浅橘色
    '#5D4037',  # 深棕色
    '#B39DDB',  # 浅紫色
    '#00897B',  # 青绿色（偏蓝）
    '#FF1744',  # 艳红色
    '#64DD17',  # 亮绿色
    '#FF9100',  # 橙红色
    '#1E88E5',  # 天蓝色
    '#D81B60',  # 玫红色
    '#00E676',  # 草绿色
    '#FFE082',  # 浅黄色
    '#424242',  # 深灰色
    '#90CAF9',  # 淡蓝色
    '#78909C',  # 银灰色
    '#FF6699',  # 浅粉色
    '#0091EA',  # 湖蓝色
    '#FFD700',  # 金色
    '#689F38',  # 深绿色
    '#FF8A65',  # 肉粉色
    '#455A64',  # 蓝黑色
    '#D4E157',  # 嫩绿色
    '#FF00B8',  # 粉红色（偏紫）
    '#00796B',  # 墨绿色
    '#FF3D00'  # 橙红色（偏亮）
]


class Detector(nn.Module):
    """
    峰值检测模型封装类，用于加载模型、处理输入序列、检测峰值并可视化结果
    基于已训练的网络模型，结合峰值交并比（Peak IoU）和非极大值抑制（NMS）实现峰值检测
    """

    def __init__(self, peak_iou=True, nms=True, three_head=True, dfl=True, bce_or_mse='bce', thresh=0.5,
                 peak_iou_thresh=0.4):
        """
        初始化检测器
        参数:
            peak_iou: 是否使用Peak IoU作为匹配指标
            nms: 是否启用非极大值抑制
            three_head: 模型是否使用三检测头结构
            dfl: 模型是否使用分布焦点损失（DFL）
            bce_or_mse: 损失函数类型（BCE或MSE）
            thresh: 置信度阈值（用于筛选有效预测）
            peak_iou_thresh: Peak IoU阈值（用于NMS）
        """
        super(Detector, self).__init__()
        # 初始化检测网络
        self.net = Net(three_head=three_head, dfl=dfl)
        # 根据参数确定最佳权重文件路径
        if peak_iou:
            bce_or_mse = 'none'  # 使用Peak IoU时不使用BCE/MSE
        self.best_weight_path = f"params/net_best_{peak_iou}_{nms}_{three_head}_{dfl}_{bce_or_mse}.pth"

        # 加载预训练权重（如果存在）
        if os.path.exists(self.best_weight_path):
            self.net.load_state_dict(torch.load(self.best_weight_path, map_location='cpu'))
        self.net.eval()  # 设置为评估模式

        self.thresh = thresh  # 置信度阈值
        self.peak_iou_thresh = peak_iou_thresh  # Peak IoU阈值

    def get_true_position(self, sequence):
        """
        处理输入序列，通过网络获取峰值的真实位置预测
        参数:
            sequence: 输入序列数据（原始信号）
        返回:
            筛选后的峰值预测结果（置信度高于阈值的预测）
        """
        output = self.net(sequence)  # 模型预测
        # 转换输出格式，应用网格偏移
        output = Utils.transform_output(output, grid_offset=0.5)
        # 分割输出，保留有效部分（去除首尾冗余维度）
        _, output, _ = torch.split(output, [1, output.size(-1) - 2, 1], dim=-1)
        # 根据置信度阈值筛选预测结果
        mask = output[:, :, 0] > self.thresh
        output = output[mask]
        output = output.squeeze(0)  # 压缩维度
        return output

    @staticmethod
    def peak_iou_operate(output1, output2):
        """
        计算两个峰值预测框之间的Peak IoU（峰值交并比）
        Peak IoU在传统IoU基础上增加了峰值位置距离的惩罚项，更适合峰值检测任务
        参数:
            output1: 第一个峰值的预测框参数 [开始位置, 结束位置, 峰值位置, 高度]
            output2: 第二个峰值的预测框参数 [开始位置, 结束位置, 峰值位置, 高度]
        返回:
            Peak IoU值（越大表示匹配度越高）
        """
        # 计算交集区域的左右边界（取最大值作为开始，最小值作为结束）
        intersection_start = torch.max(output1[1], output2[1])
        intersection_end = torch.min(output1[2], output2[2])
        # 计算交集区域的长度（确保非负）
        intersection_length = torch.clamp(intersection_end - intersection_start, min=0.0)
        # 计算交集区域的高度（取两者高度的最小值）
        intersection_height = torch.min(output1[4], output2[4])
        intersection_area = intersection_length * intersection_height  # 交集面积

        # 计算两个预测框的面积
        output1_area = (output1[2] - output1[1]) * output1[4]
        output2_area = (output2[2] - output2[1]) * output2[4]

        # 计算并集面积（面积和减去交集）
        union_area = output1_area + output2_area - intersection_area

        # 计算传统交并比（IoU）
        iou = intersection_area / union_area

        # 计算峰值位置距离及归一化因子
        peak_distance = torch.abs(output1[3] - output2[3])  # 峰值位置绝对距离
        union_start = torch.min(output1[1], output2[1])  # 并集区域起始位置
        union_end = torch.max(output1[2], output2[2])  # 并集区域结束位置
        union_distance = torch.abs(union_end - union_start)  # 并集区域长度（用于归一化）

        # 计算Peak IoU（IoU减去归一化的峰值距离惩罚）
        peak_iou = iou - (peak_distance / union_distance)
        return peak_iou.item()  # 转换为Python数值

    def non_max_suppression(self, output):
        """
        应用非极大值抑制（NMS）去除重叠的峰值预测
        通过Peak IoU筛选，保留置信度最高且与其他预测重叠度低的峰值
        参数:
            output: 原始预测结果（未经过滤的峰值预测）
        返回:
            筛选后的峰值预测框（去除冗余重叠预测）
        """
        # 处理维度（确保至少为2维）
        if output.dim() == 1:
            output = output.unsqueeze(0)
        # 按置信度降序排序（优先保留高置信度预测）
        output = output[output[:, 0].argsort(descending=True)]
        anchor_box_select = []  # 存储筛选后的预测框

        # 遍历所有预测框，进行非极大值抑制
        for i in range(output.size(0)):
            select_or_not = True  # 标记当前预测框是否被选中
            # 与已选中的预测框比较Peak IoU
            for anchor_box_ready in anchor_box_select:
                peak_iou = self.peak_iou_operate(anchor_box_ready, output[i])
                # 若Peak IoU超过阈值，则认为重叠过高，不选中当前预测框
                if peak_iou > self.peak_iou_thresh:
                    select_or_not = False
            if select_or_not:
                anchor_box_select.append(output[i])  # 选中当前预测框

        # 转换为张量并去除置信度维度
        anchor_box_select = torch.stack(anchor_box_select, dim=0)
        _, anchor_box_select = torch.split(anchor_box_select, [1, anchor_box_select.size(-1) - 1], dim=-1)
        return anchor_box_select

    @staticmethod
    def target_transform(target):
        """
        转换目标或预测框参数，用于生成高斯峰序列
        将[开始位置, 结束位置, 峰值位置, 高度]转换为高斯分布参数
        参数:
            target: 原始目标/预测框参数
        返回:
            高斯分布参数（均值、振幅、左右标准差）
        """
        target_second = target.clone()
        # 提取峰值位置作为高斯分布均值
        target_second[:, 0] = target[:, 2]
        # 提取高度作为振幅
        target_second[:, 1] = target[:, 3]
        # 计算左侧标准差（峰值到开始位置的距离/3）
        target_second[:, 2] = (target[:, 2] - target[:, 0]) / 3
        # 计算右侧标准差（结束位置到峰值的距离/3）
        target_second[:, 3] = (target[:, 1] - target[:, 2]) / 3
        # 分割参数并转换为numpy数组
        multi_gaussian_params = torch.split(target_second, 1, dim=-1)
        multi_gaussian_params = tuple(x.detach().numpy() for x in multi_gaussian_params)
        return multi_gaussian_params

    @staticmethod
    def generate_gaussian_sequence(gaussian_params):
        """
        根据高斯分布参数生成高斯峰序列（多个高斯峰的叠加）
        参数:
            gaussian_params: 高斯分布参数元组 (位置, 振幅, 左侧标准差, 右侧标准差)
        返回:
            生成的高斯峰序列列表（每个元素为一个高斯峰）
        """
        # 解析参数
        location, amplitude, left_std, right_std = gaussian_params
        # 生成序列索引（假设数据长度为DATA_LENGTH）
        indices = np.arange(DATA_LENGTH)
        sequences = []

        # 为每个峰值生成高斯序列
        for i in range(len(location)):
            loc = location[i]
            amp = amplitude[i]
            left = left_std[i]
            right = right_std[i]

            # 左侧高斯分布（位置左侧使用左侧标准差）
            left_half = np.exp(-0.5 * ((indices - loc) / left) ** 2)
            # 右侧高斯分布（位置右侧使用右侧标准差）
            right_half = np.exp(-0.5 * ((indices - loc) / right) ** 2)
            # 合并左右两侧，形成非对称高斯峰
            gaussian = np.where(indices < loc, left_half, right_half)
            gaussian *= amp  # 应用振幅
            sequences.append(gaussian)

        return sequences

    def __call__(self, sequence, target, max_value=1.0, k_x=1.0, k_y=1.0, width_max=1):
        """
        主调用函数：处理输入序列，执行峰值检测，并可视化检测结果与原始序列的对比
        参数:
            sequence: 输入的原始信号序列（如波形数据）
            target: 真实峰值标签（用于与预测结果对比）
            max_value: 振幅最大值（用于归一化）
            k_x: x轴缩放因子（用于坐标转换）
            k_y: y轴缩放因子（用于振幅转换）
            width_max: 宽度最大值（用于面积计算）
        """
        # 筛选出非零的目标标签（去除无效的零值标签）
        mask = (target != 0).detach().clone()
        target = target[torch.any(mask, dim=-1)]
        # 通过模型获取筛选后的峰值预测结果（置信度高于阈值）
        output = self.get_true_position(sequence)

        if not output.numel():
            return [],[],[],[],[],[]

        # 应用非极大值抑制（NMS）去除重叠的预测框
        anchor_box_select = self.non_max_suppression(output)

        # 将目标标签和预测框转换为高斯分布参数（用于生成高斯峰）
        target_multi_gaussian_params = self.target_transform(target)
        output_multi_gaussian_params = self.target_transform(anchor_box_select)

        # 解析预测结果的高斯参数（位置、振幅、左右标准差）
        output_locations, output_amplitude, output_left_stds, output_right_stds = output_multi_gaussian_params

        # 调整预测振幅（应用y轴缩放因子）
        output_amplitude = output_amplitude / k_y * width_max

        # 按峰值位置排序（确保绘图时从左到右显示）
        sorted_indices = np.argsort(output_locations, axis=0)
        output_locations = output_locations[sorted_indices].squeeze(-1)  # 压缩维度
        output_amplitude = output_amplitude[sorted_indices].squeeze(-1)
        output_left_stds = output_left_stds[sorted_indices].squeeze(-1)
        output_right_stds = output_right_stds[sorted_indices].squeeze(-1)

        # 重组排序后的预测高斯参数
        output_multi_gaussian_params = output_locations, output_amplitude, output_left_stds, output_right_stds

        # 根据高斯参数生成预测的高斯峰序列和真实目标的高斯峰序列
        output_sequences = self.generate_gaussian_sequence(output_multi_gaussian_params)
        target_sequences = self.generate_gaussian_sequence(target_multi_gaussian_params)

        output_locations = output_locations / k_x
        output_left_stds = output_left_stds / k_x
        output_right_stds = output_right_stds / k_x

        # 计算每个预测峰值的面积（基于振幅和标准差）
        output_areas = output_amplitude * (output_left_stds + output_right_stds)

        output_locations = np.array(output_locations).flatten().tolist()
        output_areas = np.array(output_areas).flatten().tolist()
        output_amplitude = np.array(output_amplitude).flatten().tolist()
        output_left_stds = np.array(output_left_stds * 3).flatten().tolist()
        output_right_stds = np.array(output_right_stds * 3).flatten().tolist()
        '''
        plt.plot(sequence.squeeze(0).squeeze(0))
        plt.show()
        for j in range(len(output_sequences)):
            plt.plot(output_sequences[j])
        plt.show()
        for k in range(len(target_sequences)):
            plt.plot(target_sequences[k])
        plt.show()
        '''

        return output_sequences, output_locations, output_areas, output_amplitude, output_left_stds, output_right_stds

