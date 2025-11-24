import torch.nn
import warnings
from nni.compression.utils.counter import count_flops_params
from analysis import AnAlysis
from Net import Net
from loss import Loss_Calculate
from dataset import *
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import time
import os

# 设置环境变量（禁用TensorFlow的oneDNN优化，可能与PyTorch冲突）
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def set_seed(seed):
    """设置随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class Train:
    """
    模型训练类，负责网络训练、验证和测试的完整流程
    支持多种模型配置组合的对比实验
    """

    def __init__(self, epochs=100, peak_iou=True, nms=True, three_head=True, dfl=True, bce_or_mse='bce',
                 num_train=50000, num_test=20000, num_validation=10000):
        """
        初始化训练器
        参数:
            epochs: 训练轮数
            peak_iou: 是否使用Peak IoU损失
            nms: 是否使用非极大值抑制
            three_head: 是否使用三检测头结构
            dfl: 是否使用分布焦点损失
            bce_or_mse: 分类损失类型（BCE或MSE）
            num_train: 训练集大小
            num_test: 测试集大小
            num_validation: 验证集大小
        """
        # 设置设备（GPU或CPU）
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载数据集
        self.train_set = DataSet(num_samples=num_train, data_type='train')
        self.test_set = DataSet(num_samples=num_test, data_type='test')
        self.validation_set = DataSet(num_samples=num_validation, data_type='validation')
        self.num_test = num_test
        self.num_validation = num_validation

        # 创建数据加载器
        self.train_dataloader = DataLoader(self.train_set, batch_size=32, shuffle=True, num_workers=8)
        self.test_dataloader = DataLoader(self.test_set, batch_size=32, shuffle=False)
        self.validation_dataloader = DataLoader(self.validation_set, batch_size=32, shuffle=False)

        # 根据参数组合确定模型权重保存路径
        if peak_iou:
            bce_or_mse = 'none'  # 使用Peak IoU时不使用BCE/MSE
        self.best_weight_path = f"params/net_best_{peak_iou}_{nms}_{three_head}_{dfl}_{bce_or_mse}.pth"
        self.new_weight_path = f"params/net_new_{peak_iou}_{nms}_{three_head}_{dfl}_{bce_or_mse}.pth"

        # 初始化模型并移至设备
        self.net = Net(three_head=three_head, dfl=dfl).to(self.device)

        # 计算模型的FLOPs和参数量（用于性能分析）
        x = torch.randn(2, 1, 1024)
        count_flops_params(self.net, x)

        # 加载预训练权重（如果存在）
        if os.path.exists(self.best_weight_path):
            pass  # 注释掉加载操作，避免覆盖已有训练

        # 初始化优化器（使用随机梯度下降）
        self.net_optim = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        # 初始化损失函数
        self.loss_fn = Loss_Calculate(peak_iou=peak_iou, bce_or_mse=bce_or_mse)

        # 训练配置
        self.epochs = epochs
        self.peak_iou = peak_iou
        self.nms = nms
        self.file_name = f"results/results_{peak_iou}_{nms}_{three_head}_{dfl}_{bce_or_mse}.txt"

    def __call__(self):
        """执行完整的训练、测试和验证流程"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # 忽略警告

            # 初始化最佳指标
            best_map50, best_map50_95, best_rss = 0, 0, 0
            best_ap = []
            best_pre = []
            best_rec = []
            net_loss_train = 0

            # 开始训练循环
            for epoch in range(self.epochs):
                self.net.train()  # 设置为训练模式
                self.net = self.net.to(self.device)  # 确保模型在正确设备上

                # 训练一个轮次
                for step, (sequences, targets, _) in tqdm(enumerate(self.train_dataloader),
                                                          desc="第{}轮训练".format(epoch + 1),
                                                          total=len(self.train_dataloader)):
                    # 将数据移至设备
                    sequences, targets = torch.tensor(sequences).to(self.device), torch.tensor(targets).to(self.device)

                    # 前向传播
                    outputs = self.net(sequences)
                    loss = self.loss_fn(outputs, targets)

                    # 反向传播和优化
                    self.net_optim.zero_grad()
                    loss.backward()
                    self.net_optim.step()

                    net_loss_train += loss.item()

                # 计算平均训练损失
                net_loss_train = net_loss_train / len(self.train_dataloader)
                print("网络损失为：{}".format(net_loss_train))
                time.sleep(1)  # 短暂暂停，便于查看输出

                # 保存当前模型权重
                torch.save(self.net.state_dict(), self.new_weight_path)

                # 在测试集上评估模型
                with torch.no_grad():
                    self.net.eval()  # 设置为评估模式
                    print("第{}轮测试".format(epoch + 1))
                    analysis = AnAlysis(nms=self.nms, num_analysis=self.num_test)
                    pre, rec, map50, map50_95, ap, rss = analysis(self.net, self.new_weight_path, self.test_set)

                # 记录本轮训练结果到文件
                with open(self.file_name, 'a') as file:
                    file.write(f'第{epoch + 1}次\n')
                    file.write(f'train_loss为{net_loss_train}\n')
                    file.write(f"pre: {pre}, rec: {rec}\n")
                    file.write(f"ap: {ap}\n")
                    file.write(f"map50: {map50}, map50_95: {map50_95}\n")
                    file.write(f"rss: {rss}\n")

                # 如果当前模型在测试集上的性能更好，则保存为最佳模型
                if map50_95 >= best_map50_95:
                    torch.save(self.net.state_dict(), self.best_weight_path)
                    best_map50, best_map50_95, best_rss = map50, map50_95, rss
                    best_ap = ap[:]
                    best_pre = pre[:]
                    best_rec = rec[:]

            # 记录最佳测试结果
            with open(self.file_name, 'a') as file:
                file.write('最好的测试结果\n')
                file.write(f"best_pre: {best_pre}, best_rec: {best_rec}\n")
                file.write(f"best_ap: {best_ap}\n")
                file.write(f"best_map50: {best_map50}, best_map50_95: {best_map50_95}\n")
                file.write(f"best_rss: {best_rss}\n")

            with torch.no_grad():
                self.net.eval()  # 设置为评估模式
                # 在验证集上评估最佳模型
                analysis = AnAlysis(nms=self.nms, num_analysis=self.num_validation)
                print('验证结果')
                pre, rec, map50, map50_95, ap, rss = analysis(self.net, self.best_weight_path, self.validation_set)

            # 记录验证结果
            with open(self.file_name, 'a') as file:
                file.write('验证结果为\n')
                file.write(f"pre: {pre}, rec: {rec}\n")
                file.write(f"ap: {ap}\n")
                file.write(f"map50: {map50}, map50_95: {map50_95}\n")
                file.write(f"rss: {rss}\n")


if __name__ == "__main__":
    """主函数：执行多组对比实验"""
    set_seed(0)  # 设置随机种子，确保实验可复现

    # 定义多组实验参数组合，用于对比不同模型配置的性能
    param = {
        1: {'peak_iou': False, 'nms': False, 'three_head': False, 'dfl': False, 'bce_or_mse': 'mse'},
        2: {'peak_iou': False, 'nms': False, 'three_head': False, 'dfl': False, 'bce_or_mse': 'bce'},
        3: {'peak_iou': True, 'nms': False, 'three_head': False, 'dfl': False, 'bce_or_mse': 'none'},
        4: {'peak_iou': True, 'nms': True, 'three_head': False, 'dfl': False, 'bce_or_mse': 'none'},
        5: {'peak_iou': True, 'nms': True, 'three_head': False, 'dfl': True, 'bce_or_mse': 'none'},
        6: {'peak_iou': True, 'nms': True, 'three_head': True, 'dfl': False, 'bce_or_mse': 'none'},
        7: {'peak_iou': True, 'nms': True, 'three_head': True, 'dfl': True, 'bce_or_mse': 'none'},
    }

    # 依次执行每组实验
    for i in range(7):
        selected_number = i + 1
        params = param[selected_number]
        train = Train(**params)  # 使用参数解包初始化训练器
        train()  # 执行训练和评估