import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的深度神经网络类，包含层归一化
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)  # 第一层的层归一化
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)  # 第二层的层归一化
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = torch.relu(x)  # 使用ReLU激活函数
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
input_dim = 10
hidden_dim = 20
output_dim = 1
model = SimpleModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建随机输入数据和目标数据用于训练
torch.manual_seed(0)
data = torch.randn(100, input_dim)
target = torch.randn(100, output_dim)

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# 查看模型的层归一化后的输出
with torch.no_grad():
    test_input = torch.randn(1, input_dim)
    test_output = model(test_input)
    print("Test input:", test_input)
    print("Model output after LayerNorm and activation:", test_output)




# 定义带有残差连接的简单神经网络
class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # 输入经过第一层全连接层和ReLU激活
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        # 加入残差连接，并通过层归一化
        out = self.layer_norm(out + residual)
        return torch.relu(out)

# 构建多层残差网络
class DeepResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks):
        super(DeepResidualNetwork, self).__init__()
        self.blocks = nn.ModuleList([ResidualBlock(input_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.fc_out(x)

# 初始化模型、损失函数和优化器
input_dim = 10
output_dim = 1
num_blocks = 5  # 使用5个残差块
model = DeepResidualNetwork(input_dim, output_dim, num_blocks)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建随机输入数据和目标数据用于训练
torch.manual_seed(0)
data = torch.randn(100, input_dim)
target = torch.randn(100, output_dim)

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# 测试残差连接对模型输出的影响
with torch.no_grad():
    test_input = torch.randn(1, input_dim)
    test_output = model(test_input)
    print("Test input:", test_input)
    print("Model output with residual connection:", test_output)

