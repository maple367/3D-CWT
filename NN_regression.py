# %%
'''加载数据'''
import numpy as np
import pandas as pd
all_data = np.load('mesh_data_set_3hole.npz',allow_pickle=True)['arr_0']
df = pd.DataFrame(all_data, columns=['eps_array', 'Q', 'SE'])
df = df[(df['SE'] <= 1.0) & (df['SE'] >= 0.0) & (df['Q'] >= 0.0)]
df['eps_array'] = df['eps_array'].apply(lambda x: (np.fft.fft2(df.values[0][0])/(32*32)).astype(np.complex64))
df['SE'] = df['SE'].astype(np.float32)
df['Q'] = df['Q'].astype(np.float32)
train_df = df.sample(frac=0.9, random_state=42)
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
        real_sample = np.real(sample)
        imag_sample = np.imag(sample)
        
        # Stack the real and imaginary parts into a single feature vector
        sample = np.stack((real_sample, imag_sample), axis=-1)
        
        # # For labels, separate into real and imaginary parts as well
        # real_label = np.real(label)
        # imag_label = np.imag(label)
        # label = np.stack((real_label, imag_label), axis=-1)
        
        return sample, label
    
# %%
'''数据集'''
training_data = CustomDataset(train_df['eps_array'].values, np.log(train_df['Q'].values))
test_data = CustomDataset(test_df['eps_array'].values, np.log(test_df['Q'].values))

'''
创建数据DataLoader（数据加载器）
bath_size:将数据集分成多份，每一份为bath_size个数据
优点：可以减少内存的使用，提高训练的速度
'''
batch_size = 128
train_dataloader = DataLoader(training_data,batch_size=batch_size,drop_last=True)  #64张图片为一个包
test_dataloader = DataLoader(test_data,batch_size=batch_size)
for X,Y in train_dataloader:  #X表示打包好的每一个数据包
    print(f'Shape of X:{X.shape} {X.dtype}')
    print(f'Shape of Y:{Y.shape} {Y.dtype}')
    break
# %%
'''判断当前设备是否支持GPU'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using  {device}  device')


# %%
'''创建神经网络模型'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(32 * 32 * 2, 2048)  # Adjusted for real and imaginary parts
        self.hidden2 = nn.Linear(2048, 1024)
        self.hidden3 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, 1)  # Output

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

# %%
# 神经网络
model = NeuralNetwork().to(device)
print(model)
# 损失函数和优化器
loss_fn = nn.MSELoss()  # Mean Squared Error for real and imaginary parts
optimizer = torch.optim.Adam(model.parameters(), lr=0.00015)

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
        print(f'Loss: {loss.item()}')
    return loss.item()

# 测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
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
    import matplotlib.pyplot as plt
    plt.plot([min(np.min(y_list),np.min(pred_list)), max(np.max(y_list),np.max(pred_list))], [min(np.min(y_list),np.min(pred_list)), max(np.max(y_list),np.max(pred_list))])
    plt.scatter(y_list, pred_list)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.close()
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
