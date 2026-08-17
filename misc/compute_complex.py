import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

def compute_model_complexity(model, input1_shape, input2_shape):
    """
    计算模型的时间复杂度（FLOPs）和空间复杂度（参数数量和特征图占用内存）

    Args:
        model: PyTorch 构建的深度学习模型
        input1_shape: 第一个输入的形状 (C, H, W) 或 (N, C, H, W)
        input2_shape: 第二个输入的形状 (C, H, W) 或 (N, C, H, W)

    Returns:
        时间复杂度和空间复杂度的详细信息
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 构造两个输入张量
    input1 = torch.randn(input1_shape).to(device)
    input2 = torch.randn(input2_shape).to(device)

    # 使用 torchinfo 计算模型摘要，包括 FLOPs 和参数数量
    model_info = summary(
        model,
        input_data=[input1, input2],
        verbose=0,
        col_names=["output_size", "num_params", "mult_adds"],
        depth=3  # 控制输出的层级深度
    )

    # 总参数数量
    total_params = model_info.total_params

    # 总浮点操作次数（FLOPs）
    total_flops = model_info.total_mult_adds

    # 计算所有中间特征图的总内存占用
    total_memory = sum([
        torch.tensor(layer.output_size).prod().item() if layer.output_size else 0
        for layer in model_info.summary_list
    ]) * 4  # 每个浮点数占用4字节（32位）

    # 返回复杂度结果
    return {
        "Total Parameters": total_params,
        "Total FLOPs": total_flops,
        "Total Feature Map Memory (bytes)": total_memory,
    }

# 示例双输入模型
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 64 * 64 * 2, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))

        x2 = F.relu(self.conv1(x2))
        x2 = F.relu(self.conv2(x2))

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

# 测试
if __name__ == "__main__":
    model = ExampleModel()
    input1_shape = (1, 3, 64, 64)  # 批量大小 1
    input2_shape = (1, 3, 64, 64)

    complexity = compute_model_complexity(model, input1_shape, input2_shape)

    print("模型复杂度:")
    for k, v in complexity.items():
        print(f"{k}: {v}")
