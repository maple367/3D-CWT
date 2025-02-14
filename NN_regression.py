# %%
'''加载数据'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
all_data = np.load('mesh_data_set_3hole_lessscale.npz',allow_pickle=True)['arr_0']
# %%
df = pd.DataFrame(all_data, columns=['eps_array', 'Q', 'SE'])
df = df[(df['SE'] <= 1.0) & (df['SE'] >= 0.0) & (df['Q'] >= 0.0)]
df['eps_array'] = df['eps_array'].apply(lambda x: (x).astype(np.float32))
df['SE'] = df['SE'].astype(np.float32)
df['Q'] = df['Q'].astype(np.float32)
train_df = df.sample(frac=0.9, random_state=233)
test_df = df.drop(train_df.index)
# %%
import torch.cuda
from torch import nn    #导入神经网络模块
from torch.utils.data import DataLoader, Dataset  #数据包管理工具
from torchvision.transforms import ToTensor  #数据转换，张量

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        初始化数据集
        :param data: 输入数据 (numpy 或其他格式)
        :param labels: 标签数据
        """
        self.data = data
        self.labels = labels
    
    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回一个样本和对应标签
        :param idx: 索引
        """
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Separate the complex number into real and imaginary parts
        #real_sample = np.real(sample)
        #imag_sample = np.imag(sample)
        
        # Stack the real and imaginary parts into a single feature vector
        #sample = np.stack((real_sample, imag_sample), axis=-1)
        
        # # For labels, separate into real and imaginary parts as well
        # real_label = np.real(label)
        # imag_label = np.imag(label)
        # label = np.stack((real_label, imag_label), axis=-1)
        
        return sample, label
    
# %%
'''数据集'''
training_data = CustomDataset(train_df['eps_array'].values, train_df['SE'].values)
test_data = CustomDataset(test_df['eps_array'].values, test_df['SE'].values)

'''
创建数据DataLoader（数据加载器）
bath_size:将数据集分成多份，每一份为bath_size个数据
优点：可以减少内存的使用，提高训练的速度
'''
batch_size = 64
train_dataloader = DataLoader(training_data,batch_size=batch_size,drop_last=True)  #64张图片为一个包
test_dataloader = DataLoader(test_data,batch_size=batch_size)

for X,Y in train_dataloader:  #X表示打包好的每一个数据包
    print(f'Shape of X:{X.shape} {X.dtype}')
    print(f'Shape of Y:{Y.shape} {Y.dtype}')
    break

'''判断当前设备是否支持GPU'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}.')
# %%
'''创建神经网络模型'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(32 * 32, 128)  # Adjusted for real and imaginary parts
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)  # Output

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.hidden3(x)
        x = torch.relu(x)
        x = self.out(x)  # Output real and imaginary parts
        return x

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # 卷积层1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 卷积层2
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 1)  # 输出层

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))  # 卷积 + ReLU激活
        x = torch.relu(self.conv2(x))  # 卷积 + ReLU激活
        x = x.view(x.size(0), -1)  # 展平为1维
        x = torch.relu(self.fc1(x))  # 全连接层 + ReLU激活
        x = torch.sigmoid(self.fc2(x))  # 输出层，使用sigmoid确保输出范围为[0, 1]
        return x
    

class Transformer_Model(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=3):
        super(Transformer_Model, self).__init__()
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)  # Transformer 编码器层
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  # 多层编码器
        self.fc1 = nn.Linear(d_model * 32, 128)  # 展平后连接全连接层
        self.fc2 = nn.Linear(128, 1)  # 输出层

    def forward(self, x):
        batch_size = x.size(0)
        
        # Transformer 期望输入为 (seq_len, batch_size, input_dim)
        x = x.transpose(0, 1)  # 转置为 (32, batch_size, 32)
        
        # 输入到 Transformer 编码器
        x = self.transformer(x)
        
        # 展平结果，(batch_size, d_model * seq_len)
        x = x.view(batch_size, -1)
        
        # 全连接层处理
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 输出层
        return x


# %%
# 神经网络
model = Transformer_Model().to(device)
print(model)
# 损失函数和优化器
loss_fn = nn.MSELoss()  # Mean Squared Error for real and imaginary parts
# def my_loss(output, target):
#     if abs(output - target) > 0.1:
#         return abs(output - target)-0.1
#     else:
#         return 0

loss_fn = my_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# 训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    return loss.item()

# 测试函数
def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()  # 测试模式
    test_loss, correct = 0, 0
    pred_list = []
    y_list = []
    with torch.no_grad():
        for x, y in dataloader:
            y_list += list(y.cpu().numpy())
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_list += list(pred.cpu().numpy())
            test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f'Test result: Avg loss: {test_loss}')
    # plt.plot([min(np.min(y_list),np.min(pred_list)), max(np.max(y_list),np.max(pred_list))], [min(np.min(y_list),np.min(pred_list)), max(np.max(y_list),np.max(pred_list))])
    # plt.scatter(y_list, pred_list)
    # plt.xlabel('True')
    # plt.ylabel('Predicted')
    # #plt.xscale('log')
    # #plt.yscale('log')
    # plt.show()
    # plt.close()
    return test_loss

# 训练和测试循环
epochs = 300
train_losses = []
test_losses = []

# %%
# 运行epochs次
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_losses += [train(train_dataloader, model, loss_fn, optimizer)]
    test_losses += [test(test_dataloader, model, loss_fn)]

# %%
import matplotlib.pyplot as plt
num_cut = 0
plt.plot(train_losses[num_cut:], label='train')
plt.plot(test_losses[num_cut:], label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
# %%
