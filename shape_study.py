import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def seed_everything(seed=233):
    global seed_number
    seed_number = seed
    import random
    random.seed(seed)
    np.random.seed(seed)
seed_everything()


# def f(X):
#     return np.sin(X[:,0]) + np.abs(X[:,1]) + X[:,2]**2 + X[:,3] + X[:,4]**3 + X[:,5]**4
# X = (np.random.rand(100, 13)-0.5)*3
# y = f(X)
# # explainer = shap.KernelExplainer(model=f, data=X, link="identity")
# explainer = shap.explainers.Exact(f, X)
# X = pd.DataFrame(X, columns=[fr'$\xi_{i}$' for i in range(X.shape[1])])
# shap_values = explainer(X)
# shap.plots.violin(shap_values, plot_type="layered_violin", show=False, plot_size=1.0, max_display=10)
# plt.show()

# %%
'''加载数据'''
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch.cuda
from torch import nn    #导入神经网络模块
# gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

df = pd.DataFrame(all_data, columns=['eps_array', 'Q', 'SE'])

df = df[(df['SE'] <= 1.0) & (df['SE'] >= 0.0) & (df['Q'] >= 0.0)]
df['eps_array'] = df['eps_array'].apply(lambda x: ((np.fft.fft2(x)/1024).astype(np.complex64)))
df['SE'] = df['SE'].astype(np.float32)
df['Q'] = df['Q'].astype(np.float32)

train_df = df.sample(frac=0.95, random_state=seed_number)
test_df = df.drop(train_df.index)

# %%
class FC4SE(nn.Module):
    def __init__(self):
        super(FC4SE, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 32 * 32, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)
        # self.dropout = nn.Dropout(0.2)
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
        x = torch.sigmoid(x) # enable only for SE
        return x

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

test_data = CustomDataset(test_df['eps_array'].values, test_df['SE'].values)
test_dataloader = DataLoader(test_data, batch_size=64)
model = torch.load('SE_model.pth', weights_only=False)
model.to('cpu')
model.eval()

# %%
def f(x):
    with torch.no_grad():
        x = torch.tensor(x).reshape(-1,32,32,2)
        pred = model(x)
        y = pred.numpy()
    return y.flatten()

X = []
for x, y in test_data:
    X.append(x.flatten())
X = np.array(X)
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# X = shap.kmeans(X, 1000)
X = X[:10]
explainer = shap.KernelExplainer(model=f, data=X, link="identity")
columns_name = []
for i in range(32):
    for j in range(32):
        if i > 16:
            i = i-32
        if j > 16:
            j = j-32
        columns_name.append(rf'$\xi_{{{i},{j}}}^R$')
        columns_name.append(rf'$\xi_{{{i},{j}}}^I$')
X = pd.DataFrame(X, columns=columns_name)
shap_values = explainer(X)
# %%
shap.plots.violin(shap_values, plot_type="layered_violin", show=False, plot_size=1.0, max_display=10)
plt.show()
# %%
