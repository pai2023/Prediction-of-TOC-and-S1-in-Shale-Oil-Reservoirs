import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

# 定义 CNN 模型类（与你训练时的模型结构保持一致）
class OptimizedCNN(nn.Module):
    def __init__(self, input_size, conv_filters, kernel_size, num_conv_layers, output_size, conv_dropout_rate=0.2, fc_dropout_rate=0.4):
        super(OptimizedCNN, self).__init__()
        layers = []

        # 动态创建多个卷积层，并添加 Dropout
        for i in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels=1 if i == 0 else conv_filters,
                                    out_channels=conv_filters,
                                    kernel_size=kernel_size,
                                    padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))  # 最大池化
            layers.append(nn.Dropout(conv_dropout_rate))  # 卷积层的 Dropout

        self.conv = nn.Sequential(*layers)

        # 添加全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_filters * (input_size // (2 ** num_conv_layers)), 100),
            nn.ReLU(),
            nn.Dropout(fc_dropout_rate),  # 全连接层的 Dropout
            nn.Linear(100, output_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 加载保存的张量数据（测试集数据）
X_test = torch.load("X_test_well_log_data.pt")  # 已经处理好的测试集输入张量
Y_test = torch.load("Y_test_labels.pt")  # 已经处理好的测试集标签张量

# 创建 DataLoader
test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型参数（与你训练时保持一致）
input_size = X_test.size(1)  # 输入特征的维度
conv_filters = 64  # 卷积滤波器的数量
kernel_size = 5  # 卷积核大小
num_conv_layers = 3  # 卷积层的数量
conv_dropout_rate = 0.2
fc_dropout_rate = 0.4
output_size = 1  # 输出为回归问题的标量值

# 实例化模型并加载保存的权重
model = OptimizedCNN(input_size, conv_filters, kernel_size, num_conv_layers, output_size, conv_dropout_rate, fc_dropout_rate)
model_path = r"C:\Users\User\Desktop\test\CNN-BiLSTM\best_model.pth"  # 修改为你CNN模型的路径
model.load_state_dict(torch.load(model_path))

# 使用加载的模型进行预测
model.eval()
y_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.unsqueeze(1)  # 添加一个通道维度，确保输入形状为 [batch_size, 1, input_length]
        outputs = model(X_batch)
        y_preds.extend(outputs.numpy())

# 将预测值转换为NumPy数组并展平
y_preds_np = np.array(y_preds).flatten()

# 创建一个 DataFrame 保存实际值和预测值
df_results = pd.DataFrame({
    'True Values': Y_test.cpu().numpy().flatten(),
    'Predicted Values': y_preds_np
})

# 保存到 Excel 文件
df_results.to_excel('cnn_test_predictions_output.xlsx', index=False)
print("CNN模型测试集预测值已成功保存到 Excel 文件!")
