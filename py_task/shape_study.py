# %%
'''加载数据'''
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
import torch.cuda
from torch import nn    #导入神经网络模块
# gpu or cpu
device = 'cpu'
print(f'Using {device}.')
from torch.utils.data import DataLoader, Dataset  #数据包管理工具
from torchvision.transforms import ToTensor  #数据转换，张量
from sklearn.metrics import confusion_matrix, r2_score, explained_variance_score
import seaborn as sns
#---------------------------------------------------#
#   set seed
#---------------------------------------------------#
def seed_everything(seed=233):
    global seed_number
    seed_number = seed
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

# %%
all_data = np.load('mesh_data_set_3hole_lessscale.npz',allow_pickle=True)['arr_0']
# %%
df = pd.DataFrame(all_data, columns=['eps_array', 'Q', 'SE'])

df = df[(df['SE'] <= 1.0) & (df['SE'] >= 0.0) & (df['Q'] >= 0.0)]
df['eps_array'] = df['eps_array'].apply(lambda x: ((np.fft.fftshift(np.fft.fft2(x)/1024)).astype(np.complex64)))
df['SE'] = df['SE'].astype(np.float32)
df['Q'] = df['Q'].astype(np.float32)

train_df = df.sample(frac=0.95, random_state=seed_number)
test_df = df.drop(train_df.index)

# %%
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        initialize dataset
        :param data: input data (numpy or other format)
        :param labels: label data
        """
        self._data_ = data
        self.labels = labels
        self._process_data()
        
    def _process_data(self):
        self.data = []
        for data in self._data_:
            flited = []
            for i in range(32):
                for j in range(32):
                    if i <= 16:
                        if j >= 32 - i:
                            flited.append(data[i][j])
                    else:
                        if j >= 33 - i:
                            flited.append(data[i][j])
            data = np.array(flited)
            real_part = np.real(data)
            imaginary_part = np.imag(data)
            data = np.stack((real_part, imaginary_part), axis=-1)
            self.data.append(data)
    
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
        label = self.labels[idx]
        return sample, label

# NN model
class FC4SE(nn.Module):
    def __init__(self):
        super(FC4SE, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(962, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)
        self.activation = nn.PReLU()
        self.pos_embedding = nn.Parameter(torch.randn(1, 481, 1))

    def forward(self, x):
        x = x + self.pos_embedding.repeat(1,1,2)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc_out(x)
        x = torch.sigmoid(x) # enable only for SE
        return x


class FC4Q(nn.Module):
    def __init__(self):
        super(FC4Q, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(962, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)
        self.activation = nn.PReLU()
        self.pos_embedding = nn.Parameter(torch.randn(1, 481, 1))

    def forward(self, x):
        x = x + self.pos_embedding.repeat(1,1,2)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc_out(x)
        return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=5, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


SE_model = FC4SE().to(device)
SE_model.load_state_dict(torch.load('SE_model.pth'))
SE_model.eval()
Q_model = FC4Q().to(device)
Q_model.load_state_dict(torch.load('Q_model.pth'))
Q_model.eval()
SE_data = CustomDataset(test_df['eps_array'].values, test_df['SE'].values)
Q_data = CustomDataset(test_df['eps_array'].values, np.log10(test_df['Q'].values))
SE_dataloader = DataLoader(SE_data, batch_size=1)
Q_dataloader = DataLoader(Q_data, batch_size=1)

# %%
model = SE_model
test_data = SE_dataloader
def f(x):
    with torch.no_grad():
        x = torch.tensor(x).reshape(-1,481,2)
        pred = model(x)
        y = pred.numpy()
    return y.flatten()

X = []
Y = []
for x, y in test_data:
    X.append(x.flatten())
    Y.append(y.item())
X = np.array(X)
Y = np.array(Y)
# %%
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# X = shap.kmeans(X, 1000)
X = shap.sample(X, 500, random_state=seed_number)
explainer = shap.KernelExplainer(model=f, data=X, link="identity")
columns_name = []
for i in range(32):
    for j in range(32):
        if i <= 16:
            if j >= 32 - i:
                columns_name.append(rf'$\xi_{{{i-16},{j-16}}}^R$')
                columns_name.append(rf'$\xi_{{{i-16},{j-16}}}^I$')
        else:
            if j >= 33 - i:
                columns_name.append(rf'$\xi_{{{i-16},{j-16}}}^R$')
                columns_name.append(rf'$\xi_{{{i-16},{j-16}}}^I$')
X = pd.DataFrame(X, columns=columns_name)
shap_values = explainer(X)
# %%
# np.savez('shap_values_SE.npz', shap_values=shap_values)
plt.figure(dpi=300)
shap.plots.violin(shap_values, plot_type="layered_violin", show=False, max_display=15)
ax = plt.gca()
collections = ax.collections # List of all collections (e.g., scatter points)
for c in collections:
    c.set_edgecolor('none') # Set edge color for each collection
plt.xlabel('SHAP value of $SEE$')
plt.show()
# %%
