import warnings

import torch

from utils import Utils
from dataset import *
from config import *  # 配置文件，包含PEAK_NUM_MAX、DATA_LENGTH等配置参数
from Net import Net


class Analysis:
    """
    峰值检测结果分析类，用于处理模型输出并计算评估指标
    """

    def __init__(self, nms=True, confidence_threshold=0.5, peak_iou_thresh=0.5):
        """
        初始化分析器
        Args:
            nms (bool): 是否启用非极大值抑制（NMS）
            confidence_threshold(float)：置信度阈值，用于初筛正样本
            peak_iou_thresh (float): peak_IoU阈值，用于NMS和匹配判断
        """
        self.nms = nms
        self.confidence_threshold = confidence_threshold
        self.peak_iou_thresh = peak_iou_thresh

    def get_true_position(self, output):
        """
        将网络输出转成根据置信度阈值初步筛选有效检测框

        Args:
            output (Tensor): 模型原始输出张量（形状为[batch, grid, channels]）

        Returns:
            Tensor: 筛选后的有效检测框（形状为[num_boxes, 5]，包含置信度、坐标等）
            outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height
        """
        # 将模型输出转换形状[1,224,7]
        # 7-anchor_points, outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height, outputs_plr
        output = Utils.transform_output(output, grid_offset=0.5)

        # 去掉anchor_points和outputs_plr,[1,224,5]
        _, output, _ = torch.split(output, [1, output.size(-1) - 2, 1], dim=-1)

        # 创建置信度掩码，保留大于阈值的检测框
        mask = output[:, :, 0] > self.confidence_threshold
        output = output[mask]

        return output

    @staticmethod
    def nms_peak_iou_operate(output1, output2):
        """
        计算两个检测框的peak_IoU（Peak-IoU），用于后续的nms

        Args:
            output1 (Tensor): 检测框1（形状为[5]，包含outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height）
            output2 (Tensor): 检测框2（形状为[5]，包含outputs_p, outputs_start, outputs_end, outputs_peak, outputs_height）

        Returns:
            float: Peak-IoU
        """
        # 计算交集区域的左右边界（取最大值作为开始，最小值作为结束）
        intersection_start = torch.max(output1[1], output2[1])
        intersection_end = torch.min(output1[2], output2[2])

        # 计算交集区域的长度，并确保长度不小于0
        intersection_length = torch.clamp(intersection_end - intersection_start, min=0.0)
        intersection_height = torch.min(output1[4], output2[4])

        # 计算交集的面积
        intersection_area = intersection_length * intersection_height

        # 计算两个框的面积
        output1_area = (output1[2] - output1[1]) * output1[4]
        output2_area = (output2[2] - output2[1]) * output2[4]

        # 计算并集面积
        union_area = output1_area + output2_area - intersection_area

        # 计算IoU
        iou = intersection_area / union_area

        # 计算峰值点距离（检测框中心x坐标差）
        peak_distance = torch.abs(output1[3] - output2[3])

        # 计算并集区域的左右边界距离
        union_start = torch.min(output1[1], output2[1])
        union_end = torch.max(output1[2], output2[2])
        union_distance = torch.abs(union_end - union_start)

        # 计算peak_iou
        peak_iou = iou - (peak_distance / union_distance)

        return peak_iou

    def non_max_suppression(self, output):
        """
        非极大值抑制（NMS）算法，基于Peak_IoU阈值消除冗余框
        Args:
            output (Tensor): 检测框列表（形状为[num_boxes, 5]）
        Returns:
            Tensor: 筛选后的检测框（形状为[PEAK_NUM_MAX, 5]，填充零到最大数量）
        """
        # 按置信度降序排列检测框
        output = output[output[:, 0].argsort(descending=True)]

        # 初始化选中的锚框张量（填充零）
        anchor_box_select = torch.zeros(PEAK_NUM_MAX, 5)

        # 初始化已选框计数器
        j = int(0)

        for i in range(output.size(0)):
            select_or_not = True

            # 检查与已选框的Peak_IoU是否超过阈值
            for anchor_box_ready in anchor_box_select:
                peak_iou = self.nms_peak_iou_operate(anchor_box_ready, output[i])

                if peak_iou > self.peak_iou_thresh:
                    # 超出peak_iou阈值的框重叠度过高不选
                    select_or_not = False

            if select_or_not:
                # 如果选中加入筛选后的检测框中
                if j < PEAK_NUM_MAX:
                    anchor_box_select[j] = output[i]
                j += 1

        return anchor_box_select

    @staticmethod
    def align_first_dim(output, target):
        """
        对齐两个张量的第一维（通常是批量维度或序列长度），使它们具有相同的大小。
        主要用于将模型预测结果与真实标签对齐，以便进行后续的计算（如损失函数计算）。
        参数:
            output: 模型的预测结果张量
            target: 真实标签张量
        返回:
            tuple: 对齐后的output张量、对齐后的target张量、以及对齐后的第一维大小
        """
        # 如果output是一维张量，增加一个维度使其变为二维张量（例如从[5]变为[1, 5]）
        if output.dim() == 1:
            output = output.unsqueeze(0)

        # 按照output的第一列进行降序排序（通常用于按置信度对预测结果排序）
        output = output[output[:, 0].argsort(descending=True)]

        # 计算output和target第一维的最大长度，作为对齐后的长度
        max_len = max(output.size(0), target.size(0))

        # 定义一个填充函数，用于将张量扩展到max_len长度
        # 填充值为0，填充在张量的末尾
        pad = lambda t: torch.cat(
            [t, torch.zeros(max_len - t.size(0), *t.shape[1:], dtype=t.dtype, device=t.device)], dim=0)

        # 对output和target应用填充函数，使它们的第一维长度都为max_len
        return pad(output), pad(target), max_len

    @staticmethod
    def peak_iou_operate(output, target, max_len):
        """
        计算预测结果与真实标签之间的"Peak IoU"指标，结合了传统IoU和峰值距离信息。
        参数:
            output: 模型预测结果张量，形状为[max_len, 5]，每行为[置信度, 开始时间, 结束时间, 峰值位置, 高度]
            target: 真实标签张量，形状为[max_len, 5]，每行为[?, 开始时间, 结束时间, 峰值位置, 高度]
            max_len: 序列最大长度，用于对齐操作
        返回:
            peak_iou: 计算得到的Peak IoU张量，形状为[max_len, max_len]
        """
        # 扩展维度以计算所有预测与真实标签之间的两两组合
        output = output.unsqueeze(0)  # [1, max_len, 5]
        target = target.unsqueeze(1)  # [max_len, 1, 5]
        output = output.repeat(max_len, 1, 1)  # [max_len, max_len, 5]
        target = target.repeat(1, max_len, 1)  # [max_len, max_len, 5]

        # 计算交集区域的边界（左右边界取max和min）
        intersection_start = torch.max(output[:, :, 1], target[:, :, 0])  # 交集开始
        intersection_end = torch.min(output[:, :, 2], target[:, :, 1])  # 交集结束

        # 计算交集长度，确保不为负（无交集时为0）
        intersection_length = torch.clamp(intersection_end - intersection_start, min=0.0)

        # 计算交集区域的高度（取预测和真实高度的最小值）
        intersection_height = torch.min(output[:, :, 4], target[:, :, 3])

        # 计算交集面积（时间长度 × 高度）
        intersection_area = intersection_length * intersection_height

        # 计算预测区域和真实区域的面积
        pred_area = (output[:, :, 2] - output[:, :, 1]) * output[:, :, 4]
        gt_area = (target[:, :, 1] - target[:, :, 0]) * target[:, :, 3]

        # 计算并集面积
        union_area = pred_area + gt_area - intersection_area

        # 计算传统IoU（交并比）
        iou = intersection_area / union_area

        # 获取预测和真实的峰值位置
        output_peak = output[:, :, 3]
        target_peak = target[:, :, 2]

        # 计算峰值之间的绝对距离
        peak_distance = torch.abs(output_peak - target_peak)

        # 计算并集区域的开始和结束时间
        union_start = torch.min(output[:, :, 1], target[:, :, 0])
        union_end = torch.max(output[:, :, 2], target[:, :, 1])

        # 计算并集区域的总长度
        union_distance = torch.abs(union_end - union_start)

        # 计算Peak IoU：传统IoU减去归一化的峰值距离
        # 峰值距离越小，Peak IoU越高
        peak_iou = iou - (peak_distance / union_distance)

        return peak_iou

    @staticmethod
    def anchor_box_select(scores):
        """
        从分数矩阵中选择最优的锚框组合，确保每个预测框和真实目标一一对应。
        参数:
            scores: 预测框与真实目标之间的匹配分数矩阵，形状为[num_predictions, num_targets]
        返回:
            mask: 二进制掩码矩阵，指示最终选择的锚框与目标的匹配关系
        """
        # 步骤1: 按行选择，每行保留最大值对应的列，其余置为0
        max_row, _ = torch.max(scores, dim=-1)  # 计算每行的最大值 [num_predictions]
        row_mask = torch.tensor((scores == max_row.unsqueeze(1))).float()  # 创建行掩码矩阵
        masked_tensor = scores * row_mask  # 将每行非最大值位置置为0

        # 步骤2: 按列选择，在已选的行最大值中，保留每列最大值对应的行，其余置为0
        max_col, _ = torch.max(masked_tensor, dim=0)  # 计算每列的最大值 [num_targets]
        col_mask = (masked_tensor == max_col.unsqueeze(0)).float()  # 创建列掩码矩阵

        # 步骤3: 结合行和列的选择结果
        final_tensor = masked_tensor * col_mask  # 应用列掩码，进一步筛选

        # 步骤4: 创建最终的二进制掩码，阈值处理消除数值误差
        mask = final_tensor > 1e-9  # 将大于阈值的值设为True，其余设为False

        return mask

    @staticmethod
    def gaussian(x, mean, std1, std2, height):
        """
        生成非对称高斯函数值，在均值两侧使用不同的标准差。
        参数:
            x: 输入张量，计算高斯函数值的位置
            mean: 高斯分布的均值（峰值位置）
            std1: 左侧标准差（x < mean时使用）
            std2: 右侧标准差（x >= mean时使用）
            height: 高斯函数的峰值高度
        返回:
            result: 非对称高斯函数在x处的值
        """
        # 分别设置左右两侧的标准差
        left_std = std1
        right_std = std2

        # 计算非对称高斯函数值
        if std1 != 0 and std2 != 0:
            # 当x小于均值时使用左侧标准差，否则使用右侧标准差
            result = torch.where(
                x < mean,  # 条件判断
                height * torch.exp(-((x - mean) ** 2) / (2 * left_std ** 2)),  # x < mean时的高斯值
                height * torch.exp(-((x - mean) ** 2) / (2 * right_std ** 2))  # x >= mean时的高斯值
            )
        else:
            # 如果任一标准差为0，返回全零张量（避免除零错误）
            result = torch.zeros_like(x)

        return result

    def generate_peak_data(self, parameters, num_points=DATA_LENGTH):
        """
        生成由多个非对称高斯峰叠加而成的合成数据。
        参数:
            parameters: 包含每个高斯峰参数的张量，形状为 [num_peaks, 5]
                        每行格式为 [?, 左标准差, 右标准差, 均值, 高度]
            num_points: 生成数据的点数，默认为全局常量 DATA_LENGTH
        返回:
            x: 采样点位置的张量
            total_data: 叠加所有高斯峰后的合成数据张量
        """
        # 创建均匀分布的采样点
        x = torch.linspace(0, num_points, num_points)

        # 初始化总数据为全零向量
        total_data = torch.zeros(num_points)

        # 遍历每个高斯峰的参数
        for param in parameters:
            # 跳过全零参数的情况（可能表示无效峰）
            if torch.all(param == 0):
                continue

            # 解析参数：[_, 左标准差, 右标准差, 均值, 高度]
            # 第一个参数未使用（可能是置信度或其他辅助信息）
            _, std1, std2, mean, height = param

            # 生成当前高斯峰的数据
            peak_data = self.gaussian(x, mean, std1, std2, height)

            # 将当前峰叠加到总数据上
            total_data += peak_data

        return total_data

    @staticmethod
    def plot_peak_data(x, result):
        """
        可视化由多个非对称高斯峰叠加生成的合成数据。
        参数:
            x: 采样点位置的张量，代表X轴坐标
            result: 叠加所有高斯峰后的合成数据张量，代表Y轴值
        """
        # 创建一个12x4英寸的图形窗口
        plt.figure(figsize=(12, 4))

        # 绘制数据曲线
        # 使用.detach().numpy()将PyTorch张量转换为NumPy数组
        # 这是因为matplotlib通常需要NumPy数组作为输入
        plt.plot(x.detach().numpy(), result.detach().numpy())

        # 设置坐标轴标签和标题
        plt.xlabel('X')  # X轴标签
        plt.ylabel('Y')  # Y轴标签
        plt.title('Sum of Peak Data')  # 图表标题

        # 显示图形
        plt.show()

    def __call__(self, output, target, sequence):
        """
        评估模型预测结果与真实目标的匹配程度，返回多种性能指标。
        参数:
            output: 模型预测结果张量，形状为[num_predictions, 5]
                    每行格式为[置信度, 开始位置, 结束位置, 峰值位置, 高度]
            target: 真实目标张量，形状为[num_targets, 4]
                    每行格式为[开始位置, 结束位置, 峰值位置, 高度]
            sequence: 原始序列数据，用于计算重建误差
        返回:
            tuple: 包含以下指标的元组
                - output_num: 有效预测的数量
                - target_num: 真实目标的数量
                - iou_num: 在不同IoU阈值下的匹配数量列表
                - rss: 重建误差（残差平方和）
        """
        # 步骤1: 获取真实位置并应用非极大值抑制（NMS）或对齐操作
        output = self.get_true_position(output)  # 处理预测结果，获取真实位置

        if self.nms:
            output = self.non_max_suppression(output)  # 应用非极大值抑制，过滤重叠预测
            max_len = PEAK_NUM_MAX  # 最大峰值数量限制
        else:
            output, target, max_len = self.align_first_dim(output, target)  # 对齐预测和目标的维度

        # 步骤2: 从预测结果中提取参数，生成高斯峰合成数据
        output2 = output.clone()
        output_p, output_start, output_end, output_peak, output_height = torch.split(output2, 1, dim=-1)

        output_mean = output_peak  # 使用峰值位置作为高斯分布的均值
        output_height = output_height  # 高斯峰高度
        output_std1 = (output_peak - output_start) / 3  # 左侧标准差
        output_std2 = (output_end - output_peak) / 3  # 右侧标准差

        # 组合参数并生成合成数据
        parameters = torch.cat((output_p, output_std1, output_std2, output_mean, output_height), dim=-1)
        pred_sequence = self.generate_peak_data(parameters)  # 生成高斯峰叠加的合成序列

        # 步骤3: 计算重建误差（残差平方和）
        rss = torch.sum((pred_sequence - sequence) ** 2)  # 评估合成序列与原始序列的差异

        # 步骤4: 计算Peak IoU并选择最优匹配
        peak_iou = self.peak_iou_operate(output, target, max_len)  # 计算预测与真实目标的Peak IoU
        mask = self.anchor_box_select(peak_iou)  # 选择最优匹配，确保一对一对应
        output_p = output_p.repeat(1,max_len)  # [max_len, max_len, 5]

        p_and_peakiou = torch.cat((output_p[mask].unsqueeze(0),peak_iou[mask].unsqueeze(0)),dim=0)

        target_num = (target[:, 0] > 1e-9).sum().cpu().item()  # 真实目标数量

        return p_and_peakiou,target_num,rss.item()

class AnAlysis:
    def __init__(self, plr=True, num_analysis=200, nms=True):
        """
        初始化分析器
        参数:
            plr: 预留参数（可能用于控制是否使用PLR相关功能）
            num_analysis: 要分析的样本数量
            nms: 是否在分析中使用非极大值抑制（NMS）
        """
        self.plr = plr
        self.num_analysis = num_analysis  # 分析的样本总数
        self.nms = nms  # 是否启用非极大值抑制

    def __call__(self, net, weight_path, data):
        """
        执行模型评估流程：加载模型、批量预测、计算指标、汇总结果
        参数:
            net: 模型网络结构
            weight_path: 模型权重文件路径
            data: 测试数据集，每个样本包含输入、目标标签和原始序列
        返回:
            包含精确率、召回率、AP等评估指标的元组
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # 忽略警告信息

            # 加载模型到CPU
            self.net = net.to('cpu')
            self.net.eval()  # 设置为评估模式
            self.net.load_state_dict(torch.load(weight_path, map_location='cpu'))  # 加载权重

            p_and_peakiou = torch.empty(2,0)
            target_num = 0
            rss = 0

            for i in range(self.num_analysis):
                inputs, target, sequence = data[i]  # 获取样本数据
                inputs = inputs.unsqueeze(0)  # 增加批次维度
                output = self.net(inputs)  # 模型预测

                # 调用分析工具计算单样本指标
                analysis = Analysis(nms=self.nms)
                p_and_peakiou_one,target_num_one,rss_one = analysis(output, target, sequence)
                p_and_peakiou = torch.cat((p_and_peakiou,p_and_peakiou_one),dim=-1)
                target_num = target_num + target_num_one
                rss = rss + rss_one

            # 对第一行排序，返回排序后的值和原始索引
            sorted_values, sorted_indices = torch.sort(p_and_peakiou[0, :], descending=True)

            # 按排序后的索引重新排列两行
            p_and_peakiou = p_and_peakiou[:, sorted_indices]

            iou_limit = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # COCO标准IoU阈值

            total_gt = target_num  # 所有样本的真实目标总数（已在循环中累加）
            num_preds = p_and_peakiou.shape[1]  # 所有样本的有效预测框总数
            ap_list = []  # 存储 10 个 IoU 阈值对应的 AP

            # 2. 遍历每个 IoU 阈值，计算对应 AP
            for iou_thresh in iou_limit:
                # 2.1 动态统计 TP/FP（按置信度降序，逐个判断）
                cumulative_tp = 0  # 累积 TP 数
                cumulative_fp = 0  # 累积 FP 数
                precisions = []  # 存储每个预测框对应的精确率
                recalls = []  # 存储每个预测框对应的召回率

                # 按置信度降序遍历所有预测框（p_and_peakiou已按第一行降序排序，直接遍历即可）
                for idx in range(num_preds):
                    conf = p_and_peakiou[0, idx].item()  # 当前预测框置信度
                    peak_iou = p_and_peakiou[1, idx].item()  # 当前预测框与GT的Peak IoU

                    # 判断当前预测框是 TP 还是 FP
                    if peak_iou >= iou_thresh and conf > 1e-9:  # IoU达标且置信度有效→TP
                        cumulative_tp += 1
                    else:  # IoU不达标或置信度无效→FP
                        cumulative_fp += 1

                    # 计算当前的精确率和召回率
                    total = cumulative_tp + cumulative_fp
                    precision = cumulative_tp / total if total > 0 else 0.0  # 避免除以0
                    recall = cumulative_tp / total_gt if total_gt > 0 else 0.0  # 避免除以0

                    precisions.append(precision)
                    recalls.append(recall)

                # 2.2 处理 PR 曲线（补全首尾+计算上包络线，消除锯齿）
                # 补全 PR 曲线首尾点（确保覆盖 [0,0] 和 [1,0]）
                recalls = np.concatenate(([0.0], recalls, [1.0]))
                precisions = np.concatenate(([0.0], precisions, [0.0]))

                # 计算上包络线（从后往前取最大精度，确保曲线单调不增）
                for j in range(len(precisions) - 2, -1, -1):
                    precisions[j] = np.maximum(precisions[j], precisions[j + 1])

                # 2.3 积分计算 AP（梯形积分法）
                # 找到召回率变化的点（避免重复计算相同召回率的区间）
                change_points = np.where(recalls[1:] != recalls[:-1])[0]
                ap = 0.0
                for j in change_points:
                    # 梯形面积 = 召回率区间长度 × （当前精度 + 下一个精度）/ 2
                    recall_diff = recalls[j + 1] - recalls[j]
                    precision_avg = (precisions[j] + precisions[j + 1]) / 2
                    ap += recall_diff * precision_avg

                ap_list.append(ap)  # 保存当前 IoU 阈值的 AP

            # 3. 计算最终指标
            # 3.1 提取各阈值 AP（AP50~AP95）
            ap50 = ap_list[0]  # IoU=0.5 对应的 AP
            ap55 = ap_list[1]  # IoU=0.55 对应的 AP
            ap60 = ap_list[2]  # IoU=0.6 对应的 AP
            ap65 = ap_list[3]  # IoU=0.65 对应的 AP
            ap70 = ap_list[4]  # IoU=0.7 对应的 AP
            ap75 = ap_list[5]  # IoU=0.75 对应的 AP
            ap80 = ap_list[6]  # IoU=0.8 对应的 AP
            ap85 = ap_list[7]  # IoU=0.85 对应的 AP
            ap90 = ap_list[8]  # IoU=0.9 对应的 AP
            ap95 = ap_list[9]  # IoU=0.95 对应的 AP

            # 3.2 计算 AP50-95（10个 AP 的平均值）
            ap50_95 = sum(ap_list) / len(ap_list)

            # 3.3 计算总体精确率和召回率（基于 IoU=0.5 阈值，参考标准评估）
            # 重新统计 IoU=0.5 时的总 TP/FP
            total_tp_50 = sum(
                1 for idx in range(num_preds) if p_and_peakiou[1, idx] >= 0.5 and p_and_peakiou[0, idx] > 1e-9)
            total_fp_50 = num_preds - total_tp_50
            overall_precision = total_tp_50 / (total_tp_50 + total_fp_50) if (total_tp_50 + total_fp_50) > 0 else 0.0
            overall_recall = total_tp_50 / total_gt if total_gt > 0 else 0.0

            # 4. 打印所有评估结果
            print("=" * 50)
            print("AP 计算结果（IoU 0.5~0.95）：")
            print(f"AP50: {ap50:.4f}, AP55: {ap55:.4f}, AP60: {ap60:.4f}, AP65: {ap65:.4f}, AP70: {ap70:.4f}")
            print(f"AP75: {ap75:.4f}, AP80: {ap80:.4f}, AP85: {ap85:.4f}, AP90: {ap90:.4f}, AP95: {ap95:.4f}")
            print(f"AP50: {ap50:.4f}")
            print(f"AP50-95（平均 AP）: {ap50_95:.4f}")
            print(f"总体精确率（IoU=0.5）: {overall_precision:.4f}")
            print(f"总体召回率（IoU=0.5）: {overall_recall:.4f}")
            print(f"平均重建误差（RSS）: {rss / self.num_analysis:.4f}")
            print("=" * 50)

            # 5. 返回结果（与原代码评估流程兼容）
            return [overall_precision], [overall_recall], ap50, ap50_95, ap_list, rss / self.num_analysis

if __name__ == "__main__":
    three_head = True
    dfl = True
    nms = True
    net = Net(three_head=three_head, dfl=three_head)
    weight_path = "params/net_best_True_True_True_True_none.pth"
    data = DataSet(100, data_type='validation')
    analysis = AnAlysis(nms=nms,num_analysis=100)
    analysis(net, weight_path, data)