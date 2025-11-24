import warnings
import torch
import torch.nn as nn
from nni.compression.utils.counter import count_flops_params


# -----------------------------------------------#
#             原始网络架构——改进前                  #
#          基于yolov8的骨架结构和检测头             #
#         检测头：p,left,right,prl,arv            #
# -----------------------------------------------#

class Conv(nn.Module):  # 卷积模块，包含Conv1d、BN和激活函数
    def __init__(self, c1, c2, k=1, s=1, p=0, bn=True, act='silu'):
        """
        初始化卷积模块
        参数:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            p: 填充
            bn: 是否使用BatchNorm
            act: 激活函数类型（'silu'、'sigmoid'、'softmax'）
        """
        super().__init__()
        self.applyBN = bn
        self.cv = nn.Conv1d(c1, c2, k, s, p)  # 1D卷积层
        self.bn = nn.BatchNorm1d(c2)  # 批归一化层
        self.act = self._get_activation_fn(act)  # 激活函数

    @staticmethod
    def _get_activation_fn(act):
        """根据类型获取激活函数"""
        if act == 'softmax':
            return nn.Softmax(dim=-2)
        elif act == 'sigmoid':
            return nn.Sigmoid()
        else:  # 默认使用SiLU激活函数
            return nn.SiLU()

    def forward(self, x):
        """前向传播：卷积 -> 批归一化（可选）-> 激活函数"""
        return self.act(self.bn(self.cv(x))) if self.applyBN else self.act(self.cv(x))


class Bottleneck(nn.Module):  # 瓶颈模块，用于构建深层网络
    def __init__(self, c1, c2, shortcut=False, k=3, s=1, p=1, e=0.5):
        """
        初始化瓶颈模块
        参数:
            c1: 输入通道数
            c2: 输出通道数
            shortcut: 是否使用残差连接
            k: 卷积核大小
            s: 步长
            p: 填充
            e: 隐藏层通道数缩放因子
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数（通过缩放因子计算）
        self.cv1 = Conv(c1, c_, k, s, p)  # 第一个卷积层（降维）
        self.cv2 = Conv(c_, c2, k, s, p)  # 第二个卷积层（升维）
        self.add = shortcut and c1 == c2  # 残差连接条件：启用且输入输出通道数相同

    def forward(self, x):
        """前向传播：若启用残差连接则返回x + 卷积结果，否则直接返回卷积结果"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):  # 改进的CSP模块，用于特征提取
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        """
        初始化C2f模块
        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck模块数量
            shortcut: 是否使用残差连接
            e: 隐藏层通道数缩放因子
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 初始卷积层（将通道数翻倍）
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 最终卷积层（融合特征）
        # 堆叠n个Bottleneck模块
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, k=3, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # 分割特征为两部分
        y.extend(m(y[-1]) for m in self.m)  # 对第二部分应用Bottleneck并拼接
        return self.cv2(torch.cat(y, 1))  # 融合所有特征


class SPPF(nn.Module):  # 空间金字塔池化模块（快速版）
    def __init__(self, c1, c2, k=5):
        """
        初始化SPPF模块
        参数:
            c1: 输入通道数
            c2: 输出通道数
            k: 最大池化核大小
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏层通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 降维卷积
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 融合卷积（4个特征图拼接）
        self.m = nn.MaxPool1d(kernel_size=k, stride=1, padding=k // 2)  # 最大池化层

    def forward(self, x):
        """前向传播：通过多次池化获取多尺度特征"""
        y = [self.cv1(x)]  # 初始特征
        y.extend(self.m(y[-1]) for _ in range(3))  # 3次池化，获取不同尺度特征
        return self.cv2(torch.cat(y, 1))  # 拼接并融合特征


class Backbone(nn.Module):  # 骨干网络（特征提取部分）
    def __init__(self, three_channel=False):
        """
        初始化骨干网络
        参数:
            three_channel: 是否使用3通道输入（否则为1通道）
        """
        super(Backbone, self).__init__()
        c_in = 3 if three_channel else 1  # 输入通道数
        # 各阶段子模块（下采样+特征提取）
        self.sub_module1 = nn.Sequential(
            Conv(c_in, 16, 3, 2, 1)  # 下采样至1/2
        )
        self.sub_module2 = nn.Sequential(
            Conv(16, 32, 3, 2, 1),  # 下采样至1/4
            C2f(32, 32, 1)  # C2f模块强化特征
        )
        self.sub_module3 = nn.Sequential(
            Conv(32, 64, 3, 2, 1),  # 下采样至1/8
            C2f(64, 64, 2)  # 2个Bottleneck
        )
        self.sub_module4 = nn.Sequential(
            Conv(64, 128, 3, 2, 1),  # 下采样至1/16
            C2f(128, 128, 2)  # 2个Bottleneck
        )
        self.sub_module5 = nn.Sequential(
            Conv(128, 256, 3, 2, 1),  # 下采样至1/32
            C2f(256, 256, 1)  # 1个Bottleneck
        )
        self.SPPF = nn.Sequential(
            SPPF(256, 256)  # SPPF模块（当前未使用）
        )

    def forward(self, x):
        """前向传播：输出不同尺度的特征图"""
        x1 = self.sub_module1(x)  # 1/2尺度特征
        x2 = self.sub_module2(x1)  # 1/4尺度特征
        x3 = self.sub_module3(x2)  # 1/8尺度特征
        x4 = self.sub_module4(x3)  # 1/16尺度特征
        x5 = self.sub_module5(x4)  # 1/32尺度特征
        # x5 = self.SPPF(x5)  # 可选SPPF处理
        return x1, x2, x3, x4, x5


class Upsample(nn.Module):  # 上采样模块
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        """使用最近邻插值上采样（缩放因子2）"""
        return nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class Cat(nn.Module):  # 特征拼接模块
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x1, x2):
        """在通道维度拼接两个特征图"""
        return torch.cat((x1, x2), dim=1)


class Head_Detect_dfl(nn.Module):  # 带DFL（分布焦点损失）的检测头
    def __init__(self, c_in, c_out):
        """
        初始化检测头（带DFL）
        参数:
            c_in: 输入通道数
            c_out: 输出通道数
        """
        super(Head_Detect_dfl, self).__init__()
        self.cv = nn.Sequential(
            Conv(c_in, c_in, 3, 1, 1),  # 卷积增强特征
            nn.Conv1d(c_in, c_out, 1, 1, 0)  # 输出层
        )

    def forward(self, x):
        """前向传播：输出处理后的检测结果（带softmax）"""
        x = self.cv(x)
        n = (x.size(1) - 1) // 4  # 计算每个分支的通道数
        # 分割为置信度和4个坐标分支
        x_, x1, x2, x3, x4 = torch.split(x, [1, n, n, n, n], dim=1)
        x_ = nn.functional.sigmoid(x_)  # 置信度（sigmoid激活）
        # 坐标分支使用softmax（DFL特性）
        x1 = nn.functional.softmax(x1, dim=-2)
        x2 = nn.functional.softmax(x2, dim=-2)
        x3 = nn.functional.softmax(x3, dim=-2)
        x4 = nn.functional.softmax(x4, dim=-2)
        return torch.cat((x_, x1, x2, x3, x4), dim=1)  # 拼接结果


class Head_Detect(nn.Module):  # 基础检测头（不带DFL）
    def __init__(self, c_in, c_out):
        """
        初始化基础检测头
        参数:
            c_in: 输入通道数
            c_out: 输出通道数
        """
        super(Head_Detect, self).__init__()
        self.cv = nn.Sequential(
            Conv(c_in, c_in, 3, 1, 1),  # 卷积增强特征
            nn.Conv1d(c_in, c_out, 1, 1, 0)  # 输出层
        )

    def forward(self, x):
        """前向传播：输出处理后的检测结果（带sigmoid）"""
        x = self.cv(x)
        n = (x.size(1) - 1) // 4  # 计算每个分支的通道数
        # 分割为置信度和4个坐标分支
        x_, x1, x2, x3, x4 = torch.split(x, [1, n, n, n, n], dim=1)
        x_ = nn.functional.sigmoid(x_)  # 置信度（sigmoid激活）
        # 坐标分支使用sigmoid（直接回归）
        x1 = nn.functional.sigmoid(x1)
        x2 = nn.functional.sigmoid(x2)
        x3 = nn.functional.sigmoid(x3)
        x4 = nn.functional.sigmoid(x4)
        return torch.cat((x_, x1, x2, x3, x4), dim=1)  # 拼接结果


class Net(nn.Module):  # 完整网络（骨干+颈部+检测头）
    def __init__(self, three_channel=False, three_head=True, dfl=True):
        """
        初始化完整网络
        参数:
            three_channel: 是否3通道输入
            three_head: 是否使用3个检测头（多尺度检测）
            dfl: 是否使用DFL（分布焦点损失）
        """
        super(Net, self).__init__()
        self.backbone = Backbone(three_channel=three_channel)  # 骨干网络
        self.Upsample = Upsample()  # 上采样模块
        self.Concat = Cat()  # 特征拼接模块
        # 颈部模块（特征融合）
        self.C2f1 = C2f(384, 128, 1)
        self.C2f2 = C2f(192, 64, 1)
        self.Conv1 = Conv(64, 64, 3, 2, 1)  # 下采样卷积
        self.C2f3 = C2f(192, 128, 1)
        self.Conv2 = Conv(128, 128, 3, 2, 1)  # 下采样卷积
        self.C2f4 = C2f(384, 256, 1)

        self.three_head = three_head  # 是否多检测头
        # 根据配置初始化检测头
        if three_head:
            if dfl:
                self.Head_Detect1 = Head_Detect_dfl(64, 65)
                self.Head_Detect2 = Head_Detect_dfl(128, 65)
                self.Head_Detect3 = Head_Detect_dfl(256, 65)
            else:
                self.Head_Detect1 = Head_Detect(64, 5)
                self.Head_Detect2 = Head_Detect(128, 5)
                self.Head_Detect3 = Head_Detect(256, 5)
        else:
            if dfl:
                self.Head_Detect = Head_Detect_dfl(64, 65)
            else:
                self.Head_Detect = Head_Detect(64, 5)

    def forward(self, x):
        """前向传播：完整网络推理"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # 忽略警告
            # 获取骨干网络输出的特征图（取后三个尺度）
            _, _, x4, x6, x9 = self.backbone(x)
            # 颈部特征融合（上采样+拼接+C2f）
            x12 = self.C2f1(self.Concat(self.Upsample(x9), x6))
            x15 = self.C2f2(self.Concat(self.Upsample(x12), x4))

            if self.three_head:  # 多检测头模式
                x18 = self.C2f3(self.Concat(self.Conv1(x15), x12))
                x21 = self.C2f4(self.Concat(self.Conv2(x18), x9))
                # 三个检测头分别输出
                y1, y2, y3 = self.Head_Detect1(x15), self.Head_Detect2(x18), self.Head_Detect3(x21)
                # 拼接并调整维度顺序
                y = torch.cat((y1, y2, y3), dim=-1).permute(0, 2, 1)
            else:  # 单检测头模式
                y = self.Head_Detect(x15).permute(0, 2, 1)
        return y


if __name__ == "__main__":
    # 测试网络各配置的输出形状
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 8种网络配置（通道数、检测头数量、是否DFL的组合）
    param = {
        1: {'three_head': False, 'dfl': False, 'three_channel': False},
        2: {'three_head': False, 'dfl': False, 'three_channel': True},
        3: {'three_head': False, 'dfl': True, 'three_channel': False},
        4: {'three_head': False, 'dfl': True, 'three_channel': True},
        5: {'three_head': True, 'dfl': False, 'three_channel': False},
        6: {'three_head': True, 'dfl': False, 'three_channel': True},
        7: {'three_head': True, 'dfl': True, 'three_channel': False},
        8: {'three_head': True, 'dfl': True, 'three_channel': True}
    }

    for i in range(8):
        selected_number = i + 1
        params = param[selected_number]
        net = Net(**params).to(device)
        x = torch.randn(10, 3, 1024).to(device) if i % 2 else torch.randn(10, 1, 1024).to(device)
        _, _, _ = count_flops_params(net, x)
        print(net(x).shape)
