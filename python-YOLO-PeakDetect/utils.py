import torch
from config import *  # 导入配置参数（如网格长度、数据长度等）
from PIL import Image
import numpy as np
from scipy.interpolate import interp1d


class Utils:
    """工具类，包含模型训练和推理中用到的各种辅助函数，如锚点计算、输出转换、目标转换等"""

    @staticmethod
    def get_anchor_points(batch_size, outputs, all_or_small=True, grid_offset=0.5):
        """
        根据批次大小、输出数据和网格偏移量，计算并返回锚点位置（用于目标检测中的基准点）

        参数:
            batch_size (int): 批次大小，用于复制锚点以匹配批次中的每个样本
            outputs (torch.Tensor): 模型输出张量，用于确定设备（CPU/GPU）
            all_or_small (bool): 是否使用全部尺寸的锚点（小+中+大）或仅小尺寸
            grid_offset (float): 网格偏移比例（0-1），调整锚点在网格中的位置

        返回:
            torch.Tensor: 调整后的锚点张量，形状为(batch_size, 锚点数量, 1)
        """
        # 从配置中获取不同尺寸网格的间隔
        small_grid_length = GRID_LENGTH[0]  # 小网格长度
        medium_grid_length = GRID_LENGTH[1]  # 中网格长度
        large_grid_length = GRID_LENGTH[2]  # 大网格长度

        # 计算不同尺寸网格的锚点位置（起始点+偏移量）
        small_anchor_points = torch.arange(0, DATA_LENGTH, small_grid_length) + grid_offset * small_grid_length
        medium_anchor_points = torch.arange(0, DATA_LENGTH, medium_grid_length) + grid_offset * medium_grid_length
        large_anchor_points = torch.arange(0, DATA_LENGTH, large_grid_length) + grid_offset * large_grid_length

        # 合并锚点（根据需求选择全部或仅小尺寸）并调整形状
        if all_or_small:
            anchor_points = torch.cat((small_anchor_points, medium_anchor_points, large_anchor_points)).reshape(1, -1,
                                                                                                                1)
        else:
            anchor_points = small_anchor_points.reshape(1, -1, 1)

        # 按批次大小复制锚点（每个样本使用相同的锚点）
        anchor_points = anchor_points.repeat(batch_size, 1, 1)

        # 确保锚点与输出在同一设备上（CPU/GPU同步）
        anchor_points = anchor_points.to(outputs.device)

        # 返回分离的浮点张量（不参与梯度计算）
        return anchor_points.detach().float()

    @staticmethod
    def transform_output_dfl(outputs, three_head=True, grid_offset=0.5):
        """
        处理使用分布焦点损失（DFL）的模型输出，将预测参数转换为实际物理坐标

        参数:
            outputs: 模型输出张量
            three_head: 是否使用三检测头结构（对应不同尺寸网格）
            grid_offset: 网格偏移量

        返回:
            转换后的输出张量，包含锚点、置信度、起始/结束位置、峰值位置、高度等信息
        """
        # 获取锚点位置
        anchor_points = Utils.get_anchor_points(int(outputs.shape[0]), outputs, all_or_small=three_head,
                                                grid_offset=grid_offset)

        # 分割输出的最后一个维度（置信度p、左/右偏移、峰值位置比例、高度等）
        parts = torch.split(outputs, [1, REG_MAX + 1, REG_MAX + 1, REG_MAX + 1, REG_MAX + 1], dim=-1)
        outputs_p, outputs_left, outputs_right, outputs_plr, outputs_height = parts

        # 初始化用于DFL计算的最大值张量（0到REG_MAX的序列）
        reg_max = torch.arange(0, REG_MAX + 1, device=outputs.device).float()

        # 计算DFL输出的实际值（通过加权求和转换离散分布为连续值）
        def cal_value(outputs, max_val, scale_factor):
            return torch.sum(outputs * max_val, dim=-1) * scale_factor / max_val[-1]

        # 计算左/右偏移、峰值位置比例、高度的实际值
        outputs_left = cal_value(outputs_left, reg_max, 1).reshape(outputs_left.shape[0], -1, 1)
        outputs_right = cal_value(outputs_right, reg_max, 1).reshape(outputs_right.shape[0], -1, 1)

        # 对三检测头结构，分别缩放不同尺寸网格的偏移量
        if three_head:
            outputs_left_s, outputs_left_m, outputs_left_l = torch.split(outputs_left, [128, 64, 32], dim=-2)
            outputs_right_s, outputs_right_m, outputs_right_l = torch.split(outputs_right, [128, 64, 32], dim=-2)
            outputs_left_s = outputs_left_s * (PEAK_WIDTH_MAX // 4)  # 小网格缩放因子
            outputs_left_m = outputs_left_m * (PEAK_WIDTH_MAX // 2)  # 中网格缩放因子
            outputs_left_l = outputs_left_l * PEAK_WIDTH_MAX  # 大网格缩放因子
            outputs_right_s = outputs_right_s * (PEAK_WIDTH_MAX // 4)
            outputs_right_m = outputs_right_m * (PEAK_WIDTH_MAX // 2)
            outputs_right_l = outputs_right_l * PEAK_WIDTH_MAX
            outputs_left = torch.cat((outputs_left_s, outputs_left_m, outputs_left_l), dim=-2)
            outputs_right = torch.cat((outputs_right_s, outputs_right_m, outputs_right_l), dim=-2)
        else:
            outputs_left = outputs_left * PEAK_WIDTH_MAX
            outputs_right = outputs_right * PEAK_WIDTH_MAX

        # 计算高度的实际值（缩放至峰值最大高度）
        outputs_height = cal_value(outputs_height, reg_max, PEAK_HEIGHT_MAX).reshape(outputs_height.shape[0], -1, 1)
        outputs_plr = cal_value(outputs_plr, reg_max, 1).reshape(outputs_plr.shape[0], -1, 1)  # 峰值位置比例

        # 计算峰值的起始和结束位置
        outputs_start = anchor_points - outputs_left
        outputs_end = anchor_points + outputs_right

        # 计算峰值的具体位置（起始位置 + 比例 * 宽度）
        outputs_peak = outputs_start + (outputs_end - outputs_start) * outputs_plr

        # 拼接所有结果（锚点、置信度、起始/结束/峰值位置、高度、比例）
        parts = anchor_points, outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height, outputs_plr
        outputs = torch.cat(parts, dim=-1)

        return outputs

    @staticmethod
    def transform_output_non_dfl(outputs, three_head=True, grid_offset=0.5):
        """
        处理不使用DFL的模型输出，直接转换预测参数为实际物理坐标

        参数和返回值与transform_output_dfl类似，适用于非DFL模式
        """
        # 获取锚点位置
        anchor_points = Utils.get_anchor_points(int(outputs.shape[0]), outputs, all_or_small=three_head,
                                                grid_offset=grid_offset)

        # 分割输出的最后一个维度（置信度p、左/右偏移、峰值位置比例、高度）
        parts = torch.split(outputs, [1, 1, 1, 1, 1], dim=-1)
        outputs_p, outputs_left, outputs_right, outputs_plr, outputs_height = parts

        # 对三检测头结构，分别缩放不同尺寸网格的偏移量
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

        # 缩放高度和峰值位置比例
        outputs_height = outputs_height * PEAK_HEIGHT_MAX
        outputs_plr = outputs_plr * 1  # 比例无需额外缩放

        # 计算起始、结束和峰值位置
        outputs_start = anchor_points - outputs_left
        outputs_end = anchor_points + outputs_right
        outputs_peak = outputs_start + (outputs_end - outputs_start) * outputs_plr

        # 拼接所有结果
        parts = anchor_points, outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height, outputs_plr
        outputs = torch.cat(parts, dim=-1)

        return outputs

    @staticmethod
    def transform_output(outputs, grid_offset=0.5):
        """
        统一的输出转换入口，根据输出形状自动判断使用DFL还是非DFL模式

        参数:
            outputs: 模型输出张量
            grid_offset: 网格偏移量

        返回:
            转换后的输出张量（包含实际物理坐标）
        """
        # 根据输出形状判断是否为三检测头及是否使用DFL
        if outputs.shape[-2] == 128:
            # 单检测头（小网格）
            if outputs.shape[-1] == 5:
                outputs = Utils.transform_output_non_dfl(outputs, three_head=False, grid_offset=grid_offset)
            else:
                outputs = Utils.transform_output_dfl(outputs, three_head=False, grid_offset=grid_offset)
        else:
            # 三检测头（小+中+大网格）
            if outputs.shape[-1] == 5:
                outputs = Utils.transform_output_non_dfl(outputs, three_head=True, grid_offset=grid_offset)
            else:
                outputs = Utils.transform_output_dfl(outputs, three_head=True, grid_offset=grid_offset)

        return outputs

    @staticmethod
    def transform_target(target, grid_offset=0.5):
        """
        将真实标签（目标）转换为与模型输出匹配的格式，用于计算损失

        参数:
            target: 原始真实标签张量
            grid_offset: 网格偏移量

        返回:
            转换后的目标张量，与模型输出结构对应
        """
        # 获取小网格的锚点位置
        small_grid_length = GRID_LENGTH[0]
        small_anchor_points = torch.arange(0, DATA_LENGTH, small_grid_length) + grid_offset * small_grid_length
        small_anchor_points = small_anchor_points.to(target.device)
        small_anchor_points = small_anchor_points.repeat(target.size(0), 1)  # 按批次复制

        # 初始化目标张量（批次大小，锚点数量，5个参数）
        target_tensor = torch.zeros((target.size(0), 128, 5), device=target.device)

        # 筛选有效的目标（排除无效的零值标签）
        valid_mask = target[:, :, 2] > 1e-9  # 峰值位置有效
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)
        i_indices = valid_indices[0]  # 批次索引
        j_indices = valid_indices[1]  # 目标索引

        # 计算目标对应的锚点索引（根据峰值位置划分到不同网格）
        index = (target[i_indices, j_indices, 2] // 8).long()

        # 填充目标张量（置信度、左/右偏移、峰值比例、高度）
        target_tensor[i_indices, index, 0] = 0.99  # 置信度设为0.99（接近1）
        # 左偏移 = (锚点 - 起始位置) / 最大宽度
        value_1 = (small_anchor_points[i_indices, index] - target[i_indices, j_indices, 0]) / PEAK_WIDTH_MAX
        target_tensor[i_indices, index, 1] = value_1.to(target_tensor.dtype)
        # 右偏移 = (结束位置 - 锚点) / 最大宽度
        value_2 = (target[i_indices, j_indices, 1] - small_anchor_points[i_indices, index]) / PEAK_WIDTH_MAX
        target_tensor[i_indices, index, 2] = value_2.to(target_tensor.dtype)
        # 峰值位置比例 = (峰值位置 - 起始位置) / 宽度
        value_3 = (target[i_indices, j_indices, 2] - target[i_indices, j_indices, 0]) / (
                target[i_indices, j_indices, 1] - target[i_indices, j_indices, 0])
        target_tensor[i_indices, index, 3] = value_3.to(target_tensor.dtype)
        # 高度归一化
        target_tensor[i_indices, index, 4] = target[i_indices, j_indices, 3].to(target_tensor.dtype)

        return target_tensor

    @staticmethod
    def read_image(image_path):
        """
        读取图像并计算每行像素值的平均值（归一化）

        参数:
            image_path: 图像文件路径

        返回:
            row_sums: 每行像素值的归一化和
            width_max: 宽度归一化因子
        """
        # 打开图片
        img = Image.open(image_path)

        # 将图片转换为灰度图（如果图片是彩色的）
        img = img.convert('L')

        # 获取图片的宽度和高度
        width, height = img.size

        # 初始化一个列表来存储每一行的像素值之和
        row_sums = []

        width_max = width * 255  # 计算最大可能值（用于归一化）

        # 遍历图片的每一行
        for y in range(height):
            row_sum = 0
            for x in range(width):
                # 获取当前像素的灰度值
                pixel = img.getpixel((x, y))
                # 累加当前行的所有像素值
                row_sum += pixel
            # 将当前行的像素值之和添加到列表中（归一化处理）
            row_sums.append(row_sum / width_max)

        return row_sums, width_max

    @staticmethod
    def transport_image(image_path, three_channel=False, max_value=0.85):
        """
        处理图像数据，将其转换为模型输入所需的格式

        参数:
            image_path: 图像路径
            three_channel: 是否生成三通道数据（原始数据、一阶差分、二阶差分）
            max_value: 归一化后的最大值

        返回:
            sequence: 处理后的序列数据
            k_x: x轴缩放因子
            k_y: y轴缩放因子
            width_max: 宽度归一化因子
        """
        # 读取图像并获取每行像素和
        sequence, width_max = Utils.read_image(image_path)

        # 原始x坐标
        x_original = np.arange(len(sequence))
        # 计算x轴缩放因子（将序列缩放到1024长度）
        k_x = 1024 / len(sequence)

        # 目标x坐标（1024个点）
        x_new = np.linspace(0, len(sequence) - 1, 1024)

        # 使用线性插值将序列调整为固定长度（1024）
        f_interp = interp1d(x_original, sequence, kind='linear', fill_value="extrapolate")
        new_data = f_interp(x_new)

        # 归一化处理（使最大值为指定值）
        sequence = (new_data / np.max(new_data)) * max_value
        k_y = max_value / np.max(new_data)  # 计算y轴缩放因子

        # 转换为PyTorch张量并调整形状
        sequence = torch.tensor(sequence).float()
        sequence = sequence.reshape(1, -1)

        # 计算一阶差分（用于检测变化率）
        dx = sequence[:, 1:] - sequence[:, :-1]  # 形状为[1, 1023]
        # 为保持形状一致，在开头添加零
        dx_pad = torch.cat((torch.zeros_like(dx[:, 0].unsqueeze(-1)), dx), dim=-1)  # 形状为[1, 1024]

        # 计算二阶差分（用于检测曲率）
        ddx = dx[:, 1:] - dx[:, :-1]  # 形状为[1, 1022]
        # 为保持形状一致，在开头和结尾添加零
        ddx_pad = torch.cat(
            (torch.zeros_like(ddx[:, 0].unsqueeze(-1)), ddx, torch.zeros_like(ddx[:, 0].unsqueeze(-1))),
            dim=-1)  # 形状为[1, 1024]

        # 是否采用三通道序列（原始数据+一阶差分+二阶差分）
        if three_channel:
            sequence = torch.cat((sequence, dx_pad, ddx_pad), dim=0)

        return sequence, k_x, k_y, width_max


if __name__ == "__main__":
    """测试代码：演示工具函数的使用"""

    # 测试transform_output函数（非DFL版本）
    outputs1 = torch.zeros(1, 224, 5)  # 创建示例输出（非DFL，5个输出维度）
    replace_values = torch.tensor([[0.9, 0.002, 0.003, 0.01, 0.7]])
    outputs1[:, 1, :] = replace_values  # 设置一个锚点的预测值

    # 测试transform_output函数（DFL版本）
    outputs2 = torch.zeros(1, 224, 65)  # 创建示例输出（DFL，65个输出维度）
    # 构建DFL格式的预测值（置信度+4个分布参数）
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
    outputs2[:, 1, :] = replace_values  # 设置一个锚点的预测值

    # 打印转换后的输出形状
    print(Utils.transform_output(outputs1, 0.5).shape)  # 非DFL版本
    print(Utils.transform_output(outputs2, 0.5).shape)  # DFL版本

    # 测试transform_target函数
    target = torch.zeros(2, 8, 4)  # 创建示例目标标签
    replace_values1 = torch.tensor([[3, 6, 5, 0.4]])  # 起始位置、结束位置、峰值位置、高度
    replace_values2 = torch.tensor([[9, 14, 10, 0.6]])
    target[:, 0, :] = replace_values1
    target[:, 1, :] = replace_values2

    # 使用实际数据测试transform_target
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

    # 测试图像转换函数
    for i in range(5):
        select_number = i + 1
        # 处理指定路径的图像并打印信息
        Utils.transport_image(f'image/2/{select_number}.jpg', False)
