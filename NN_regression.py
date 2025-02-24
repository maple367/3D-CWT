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
df['eps_array'] = df['eps_array'].apply(lambda x: ((np.fft.fft2(x)/1024).astype(np.complex64)))
df['SE'] = df['SE'].astype(np.float32)
df['Q'] = df['Q'].astype(np.float32)

train_df = df.sample(frac=0.9, random_state=233)
test_df = df.drop(train_df.index)
# %%
import torch.cuda
from torch import nn    #导入神经网络模块
from torch.utils.data import DataLoader, Dataset  #数据包管理工具
from torchvision.transforms import ToTensor  #数据转换，张量

# visualization
from sklearn.metrics import confusion_matrix, r2_score, explained_variance_score
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_distribution(y_true, y_pred, kde=False):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    plt.figure(figsize=(8, 6))
    if kde:
        sns.kdeplot(x=y_true, y=y_pred, fill=True, cbar=True, thresh=0.05)
    # Plotting the distribution of true vs predicted values
    plt.scatter(y_true, y_pred, alpha=0.15)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Perfect Fit")  # Diagonal line
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.show()

def plot_error_distribution(y_true, y_pred, bins=50, density=True, cumulative=True, alpha=0.6, color='g', **kwargs):
    error = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 6))
    plt.hist(error, bins=bins, density=density, cumulative=cumulative, alpha=alpha, color=color, **kwargs)
    plt.xlabel('Error')
    plt.ylabel('Cumulative Frequency')
    plt.title('Error Distribution')
    plt.show()


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        initialize dataset
        :param data: input data (numpy or other format)
        :param labels: label data
        """
        self.data = data
        self.labels = labels
    
    def __len__(self):
        """
        return the size of the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        return the sample and label at the given index
        :param idx: index of the sample
        """

        sample = self.data[idx]
        real_part = np.real(sample)
        imaginary_part = np.imag(sample)
        sample = np.stack((real_part, imaginary_part), axis=-1)
        label = self.labels[idx]
        
        return sample, label


# NN model
class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 32 * 32, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        # self.activation = nn.Softplus()
        self.activation = nn.PReLU()
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, 32, 1))

    def forward(self, x):
        x = x + self.pos_embedding.repeat(1,1,1,2)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc_out(x)
        return x
    
# training function
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y.unsqueeze(1))  # regression problem needs to adjust the label to ensure it is 1D
        loss.backward()
        optimizer.step()
    print(f'Train Avg loss: {loss.item()}')
    return loss.item()

# test function
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
    
    print(f'Test Avg loss: {test_loss}')
    # calculate scores
    print(f'R^2: {r2_score(y_list, pred_list)}')
    print(f'Explained Variance: {explained_variance_score(y_list, pred_list)}')
    plot_distribution(y_list, pred_list)
    return test_loss

# %%
# data set
training_data = CustomDataset(train_df['eps_array'].values, train_df['SE'].values)
test_data = CustomDataset(test_df['eps_array'].values, test_df['SE'].values)

# DataLoader

batch_size = 64
train_dataloader = DataLoader(training_data,batch_size=batch_size,drop_last=True)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

for X,Y in train_dataloader:  # X is the input, Y is the label
    print(f'Shape of X:{X.shape} {X.dtype}')
    print(f'Shape of Y:{Y.shape} {Y.dtype}')
    break

# gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}.')
# %%

model = SimpleFC().to(device)
# loss_fn = nn.MSELoss()  # Mean Squared Error
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.SmoothL1Loss(beta=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)

# initialize the iteration of epochs
epochs = 30
train_losses = []
test_losses = []

# %%
# run the training and testing loop
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

model.eval()
test_loss = 0
pred_list = []
y_list = []
with torch.no_grad():
    for x, y in test_dataloader:
        y_list += list(y.cpu().numpy())
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred_list += list(pred.cpu().numpy())
        test_loss += loss_fn(pred, y.unsqueeze(1)).item()
pred_list = np.array(pred_list).flatten()
y_list = np.array(y_list).flatten()
plot_error_distribution(y_list, pred_list, cumulative=True)
# %%
