import torch.nn
from dataset import *
from utils import Utils


class Loss_Calculate:
    """
    损失计算类，用于计算峰值检测模型的损失函数
    支持Peak-IoU损失和传统的MSE/BCE损失两种计算方式
    """

    def __init__(self, peak_iou=True, bce_or_mse='bce'):
        """
        初始化损失计算器
        Args:
            peak_iou (bool): 是否使用Peak-IoU损失（基于峰值交并比的损失）
            bce_or_mse (str): 基础损失类型，'bce'（二分类交叉熵）或'mse'（均方误差）
        """
        super(Loss_Calculate, self).__init__()
        self.factor_p = 1  # 置信度损失的权重因子
        self.factor_diou = 15  # Peak-IoU损失的权重因子
        # anchor点相对于网格单元的偏移值（0.5表示anchor点在网格中心）
        self.grid_offset = 0.5
        self.peak_iou = peak_iou  # 是否启用Peak-IoU损失
        self.bce_or_mse = bce_or_mse  # 基础损失函数类型

    def outputs_and_targets_transform(self, outputs, targets):
        """
        转换模型输出和目标标签的形状与格式，使两者能够匹配计算损失

        Args:
            outputs: 模型原始输出张量
            targets: 真实目标标签张量

        Returns:
            转换后的输出和目标张量（形状匹配，可直接用于损失计算）
        """
        # 应用网格偏移变换，调整输出中的坐标格式
        outputs = Utils.transform_output(outputs, grid_offset=self.grid_offset)

        # 在输出的第三维增加维度，并重复以匹配最大峰值数量（PEAK_NUM_MAX）
        outputs = outputs.unsqueeze(2).repeat(1, 1, PEAK_NUM_MAX, 1)

        # 计算目标区域的宽度，处理宽度为0的异常情况（避免除零错误）
        targets_width = targets[:, :, 1] - targets[:, :, 0]
        mask = targets_width > 0  # 筛选有效目标（宽度>0）
        targets_width[~mask] = 0.01  # 无效目标宽度设为0.01

        # 计算峰值在目标区域中的相对位置比例（峰值位置相对于目标起始点的偏移/目标宽度）
        targets_pkl = (targets[:, :, 2] - targets[:, :, 0]) / targets_width
        targets_pkl = targets_pkl.unsqueeze(-1)  # 增加维度以便拼接
        targets = torch.cat((targets, targets_pkl), dim=-1)  # 将比例信息拼接到目标中

        # 在目标的第二维增加维度，并重复以匹配输出的特征图大小
        targets = targets.unsqueeze(1).repeat(1, outputs.shape[1], 1, 1)

        return outputs, targets

    @staticmethod
    def peak_iou_loss_operate(outputs, targets):
        """
        计算Peak-IoU相关的损失组件，包括交并比（IoU）和峰值距离惩罚

        Args:
            outputs: 转换后的模型输出（包含预测的置信度、坐标、峰值等信息）
            targets: 转换后的真实目标（包含真实的坐标、峰值等信息）

        Returns:
            处理后的输出、目标，以及每个预测框的得分（用于筛选最优匹配）
        """
        # 分离并去除输出中的anchor点信息（保留后n-1个维度）
        _, outputs = torch.split(outputs, [1, outputs.size(-1) - 1], dim=-1)

        # 计算预测框与真实框的交集区域
        intersection_start = torch.max(outputs[:, :, :, 1], targets[:, :, :, 0])  # 交集的起始位置（取两者的最大值）
        intersection_end = torch.min(outputs[:, :, :, 2], targets[:, :, :, 1])  # 交集的结束位置（取两者的最小值）
        # 交集长度（确保非负，无交集时为0）
        intersection_length = torch.clamp(intersection_end - intersection_start, min=0.0)
        # 交集高度（取两者高度的最小值）
        intersection_height = torch.min(outputs[:, :, :, 4], targets[:, :, :, 3])
        intersection_area = intersection_length * intersection_height  # 交集面积

        # 计算预测框和真实框的面积
        pred_area = (outputs[:, :, :, 2] - outputs[:, :, :, 1]) * outputs[:, :, :, 4]  # 预测框面积
        gt_area = (targets[:, :, :, 1] - targets[:, :, :, 0]) * targets[:, :, :, 3]  # 真实框面积
        union_area = pred_area + gt_area - intersection_area  # 并集面积（面积和减去交集）

        # 计算传统交并比（IoU）
        iou = intersection_area / union_area

        # 计算Peak-IoU（在IoU基础上减去峰值距离的归一化惩罚）
        outputs_peak = outputs[:, :, :, 3]  # 预测的峰值位置
        targets_peak = targets[:, :, :, 2]  # 真实的峰值位置
        peak_distance = torch.abs(outputs_peak - targets_peak)  # 峰值之间的绝对距离

        # 计算并集区域的总长度（用于归一化峰值距离）
        union_start = torch.min(outputs[:, :, :, 1], targets[:, :, :, 0])
        union_end = torch.max(outputs[:, :, :, 2], targets[:, :, :, 1])
        union_distance = torch.abs(union_end - union_start)  # 并集区域的长度

        # Peak-IoU = IoU - （峰值距离 / 并集长度）
        peak_iou = iou - (peak_distance / union_distance)

        # 处理异常值（确保Peak-IoU在[0,1]范围内，否则设为0.01）
        peak_iou = torch.where(
            (peak_iou >= 0) & (peak_iou <= 1),
            peak_iou,
            torch.tensor(0.01, device=peak_iou.device)
        )

        # 将IoU和Peak-IoU拼接到输出中，用于后续计算
        peak_iou = peak_iou.unsqueeze(-1)
        iou = iou.unsqueeze(-1)
        outputs = torch.cat((outputs, iou, peak_iou), dim=-1)

        # 计算每个预测框的得分（Peak-IoU × 置信度）
        scores = peak_iou[:, :, :, 0] * outputs[:, :, :, 0]
        scores = scores.unsqueeze(-1)  # 增加维度

        return outputs, targets, scores

    @staticmethod
    def anchor_box_select(scores):
        """
        根据得分选择最优的预测框（anchor box），确保每个真实目标匹配唯一的预测框

        Args:
            scores: 预测框的得分矩阵（反映预测框与目标的匹配程度）

        Returns:
            max_row: 每行的最大得分（用于筛选低质量预测）
            mask: 二进制掩码，指示被选中的预测框（1表示选中，0表示未选中）
        """
        scores = scores.squeeze(-1)  # 去除最后一个维度（维度压缩）
        max_row, _ = torch.max(scores, dim=-1)  # 计算每行的最大得分（按目标维度）
        # 创建行掩码：每行中只有最大得分位置为1，其余为0
        row_mask = torch.tensor((scores == max_row.unsqueeze(2))).float()
        masked_tensor = scores * row_mask  # 应用行掩码，保留每行最大值

        max_col, _ = torch.max(masked_tensor, dim=1)  # 计算每列的最大得分（按预测维度）
        # 创建列掩码：每列中只有最大得分位置为1，其余为0
        col_mask = (masked_tensor == max_col.unsqueeze(1)).float()
        final_tensor = masked_tensor * col_mask  # 应用列掩码，确保每个目标只匹配一个预测

        mask = final_tensor > 0  # 生成最终掩码（得分>0的位置为选中的预测框）
        return max_row, mask

    def __call__(self, outputs, targets):
        """
        计算总损失（主函数）

        Args:
            outputs: 模型输出张量
            targets: 真实目标标签张量

        Returns:
            总损失值（标量张量）
        """
        if self.peak_iou:
            # 步骤1：转换输出和目标的格式，确保形状匹配
            outputs, targets = self.outputs_and_targets_transform(outputs, targets)

            # 步骤2：计算Peak-IoU和得分
            outputs, targets, scores = self.peak_iou_loss_operate(outputs, targets)

            # 步骤3：选择最优匹配的预测框
            max_row, mask = self.anchor_box_select(scores)

            # 步骤4：计算Peak-IoU损失（1 - 平均Peak-IoU，乘以权重因子）
            peak_iou_iou_loss = 1 - torch.mean(outputs[:, :, :, -1][mask])
            peak_iou_iou_loss = peak_iou_iou_loss * self.factor_diou

            # 步骤5：计算置信度损失（使用BCE损失）
            # 低质量预测（得分<0.5）的置信度应接近0
            mask_non = max_row < 0.5
            loss_p_fn = torch.nn.BCELoss()
            # 负样本损失：低质量预测的置信度与0的差异
            p_loss1 = loss_p_fn(
                outputs[:, :, :, 0][mask_non],
                torch.zeros_like(outputs[:, :, :, 0][mask_non])
            )
            # 正样本损失：高质量预测的置信度与IoU的差异
            p_loss2 = loss_p_fn(
                outputs[:, :, :, 0][mask],
                outputs[:, :, :, 6][mask]  # outputs[:, :, :, 6]为IoU值
            )
            p_loss = (p_loss1 + p_loss2) * self.factor_p  # 置信度总损失（乘以权重）

            # 总损失 = Peak-IoU损失 + 置信度损失
            loss = peak_iou_iou_loss + p_loss
        else:
            # 不使用Peak-IoU时，直接使用MSE或BCE损失
            loss_fn = torch.nn.MSELoss() if self.bce_or_mse == 'mse' else torch.nn.BCELoss()
            targets_tensor = Utils.transform_target(targets)  # 转换目标格式
            loss = loss_fn(outputs, targets_tensor)  # 计算基础损失

        return loss
