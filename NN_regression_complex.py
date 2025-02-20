# %%
'''加载数据'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
all_data = np.load('mesh_data_set_3hole_lessscale.npz',allow_pickle=True)['arr_0']
# %%
df = pd.DataFrame(all_data, columns=['eps_array', 'Q', 'SE'])

df = df[(df['SE'] <= 1.0) & (df['SE'] >= 0.0) & (df['Q'] >= 0.0)]
# df['eps_array'] = df['eps_array'].apply(lambda x: x.astype(np.float32))
df['eps_array'] = df['eps_array'].apply(lambda x: np.fft.fftshift(np.fft.fft2(x)).astype(np.complex64))
df['SE'] = df['SE'].astype(np.float32)
df['Q'] = df['Q'].astype(np.float32)

train_df = df.sample(frac=0.9, random_state=233)
test_df = df.drop(train_df.index)
# %%
import torch.cuda
from torch import nn    #导入神经网络模块
from torch.utils.data import DataLoader, Dataset  #数据包管理工具
from torchvision.transforms import ToTensor  #数据转换，张量

def apply_complex(fr:nn.Module, fi:nn.Module, input:torch.Tensor, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

def complex_relu(input):
    return torch.relu(input.real).type(torch.complex64)+1j*torch.relu(input.imag).type(torch.complex64)

def complex_dropout(input, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part, 
    mask = torch.ones(*input.shape, dtype = torch.float32)
    mask = nn.functional.dropout(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    return mask*input

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
        # real_part = np.real(sample)
        # imaginary_part = np.imag(sample)
        # sample = np.stack((real_part, imaginary_part), axis=0)
        label = self.labels[idx]
        
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
class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = ComplexLinear(32 * 32, 1024)
        self.fc2 = ComplexLinear(1024, 512)
        self.fc3 = ComplexLinear(512, 256)
        self.fc4 = ComplexLinear(256, 64)
        self.fc_out = ComplexLinear(64, 1)
        self.activation = complex_relu

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc_out(x)
        x = x.abs()
        return x


# %%
# 神经网络
model = SimpleFC().to(device)
# 损失函数和优化器

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.Softshrink(0.05)

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        loss = self.loss_fn(error).mean()
        return loss

# loss_fn = nn.MSELoss()  # Mean Squared Error
# loss_fn = nn.CrossEntropyLoss()
loss_fn = CustomLoss()

# 损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)

# 训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y.unsqueeze(1))  # 回归问题需要对标签进行调整，确保是1维
        loss.backward()
        optimizer.step()
    return loss.item()


from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_distribution(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    # Plotting the distribution of true vs predicted values
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Perfect Fit")  # Diagonal line
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.show()

# 测试函数
def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    pred_list = []
    y_list = []
    with torch.no_grad():
        for x, y in dataloader:
            y_list += list(y.cpu().numpy())
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_list += list(pred.cpu().numpy())
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()

        test_loss /= num_batches
        print(f'Test result: Avg loss: {test_loss}')
    
    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(np.mean((np.array(pred_list) - np.array(y_list)) ** 2))
    print(f"RMSE: {rmse}")
    plot_distribution(y_list, pred_list)
    return test_loss


# 训练和测试循环
epochs = 30
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
fig, ax = plt.subplots()
ax.plot(train_losses[num_cut:], label='train')
ax.plot(test_losses[num_cut:], label='test', color='orange')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()
# %%
