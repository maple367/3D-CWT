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
# %%
df = pd.DataFrame(all_data, columns=['eps_array', 'Q', 'SE'])

df = df[(df['SE'] <= 1.0) & (df['SE'] >= 0.0) & (df['Q'] >= 0.0)]
df['eps_array'] = df['eps_array'].apply(lambda x: ((np.fft.fftshift(np.fft.fft2(x)/1024)).astype(np.complex64)))
df['SE'] = df['SE'].astype(np.float32)
df['Q'] = df['Q'].astype(np.float32)

train_df = df.sample(frac=0.95, random_state=seed_number)
test_df = df.drop(train_df.index)
# %%
# visualization
import os
if not os.path.exists('./nn_opt'):
    os.makedirs('./nn_opt')

def plot_distribution(y_true, y_pred, kde=False, prefix=''):
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
    plt.savefig(f'./nn_opt/{prefix}_true_vs_pred.png')
    plt.clf()

def plot_error_distribution(y_true, y_pred, bins=50, density=True, cumulative=True, alpha=0.6, color='g', prefix='', **kwargs):
    error = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 6))
    plt.hist(error, bins=bins, density=density, cumulative=cumulative, alpha=alpha, color=color, **kwargs)
    plt.xlabel('Error')
    plt.ylabel('Cumulative Frequency')
    plt.title('Error Distribution')
    plt.savefig(f'./nn_opt/{prefix}_error_distribution.png')
    plt.clf()

def plot_learning_curve(train_losses, test_losses, prefix=''):
    num_cut = 0
    fig, ax = plt.subplots()
    ax.plot(train_losses[num_cut:], label='train')
    ax.plot(test_losses[num_cut:], label='test', color='orange')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.savefig(f'./nn_opt/{prefix}_learning_curve.png')
    plt.clf()

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
                    if i < 16:
                        if j >= 32 - i:
                            flited.append(data[i][j])
                    else:
                        if j >= 31 - i:
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
        self.fc1 = nn.Linear(32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)
        self.activation = nn.PReLU()
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, 1))

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
        self.fc1 = nn.Linear(32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)
        self.activation = nn.PReLU()
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, 1))

    def forward(self, x):
        x = x + self.pos_embedding.repeat(1,1,2)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc_out(x)
        return x


def evaluate_model(model, dataloader, mark=''):
    model.eval()
    pred_list = []
    y_list = []
    with torch.no_grad():
        for x, y in dataloader:
            y_list += list(y.cpu().numpy())
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_list += list(pred.cpu().numpy())
    pred_list = np.array(pred_list).flatten()
    y_list = np.array(y_list).flatten()
    plot_distribution(y_list, pred_list, prefix=mark)
    plot_error_distribution(y_list, pred_list, cumulative=True, prefix=mark)
    r2 = r2_score(y_list, pred_list)
    pearson_r = pearsonr(y_list, pred_list)
    print(f'R^2: {r2}')
    print(f'Pearson Correlation: {pearson_r}')
    return r2

def generate_dataloader(training_data, test_data, batch_size=64):
    train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X,Y in train_dataloader:  # X is the input, Y is the label
        print(f'Shape of X:{X.shape} {X.dtype}')
        print(f'Shape of Y:{Y.shape} {Y.dtype}')
        break
    return train_dataloader, test_dataloader

# %%
import optuna
class Objective:
    def __init__(self, func='SE'):
        if func == 'SE':
            self.func = self._run_SE_
            self.train_data = CustomDataset(train_df['eps_array'].values, train_df['SE'].values)
            self.test_data = CustomDataset(test_df['eps_array'].values, test_df['SE'].values)
        elif func == 'Q':
            self.func = self._run_Q_
            self.train_data = CustomDataset(train_df['eps_array'].values, np.log10(train_df['Q'].values))
            self.test_data = CustomDataset(test_df['eps_array'].values, np.log10(test_df['Q'].values))
        else:
            raise ValueError('Invalid function.')
        self.train_dataloader, self.test_dataloader = generate_dataloader(self.train_data,
                                                                          self.test_data,
                                                                          batch_size=64)

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        beta = trial.suggest_float('beta', 0.01, 0.2)
        lr = trial.suggest_float('lr', 0.00001, 0.001)
        print(f'trial: beta={beta}, lr={lr}')
        result = self.func(beta, lr, trial)
        print('result:', result)
        return result
    
    def run(self, beta=0.1, lr=0.0005, trial=None):
        return self.func(beta, lr, trial)
    
    def _run_SE_(self, beta=0.1, lr=0.0005, trial=None):
        c_time = time.strftime('%y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        model = FC4SE().to(device)
        r2 = self._run_learn_(model, self.train_dataloader, self.test_dataloader, beta, lr, trial=trial, mark=f'SE_{c_time}')
        return r2
    
    def _run_Q_(self, beta=0.1, lr=0.0005, trial=None):
        c_time = time.strftime('%y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        model = FC4Q().to(device)
        r2 = self._run_learn_(model, self.train_dataloader, self.test_dataloader, beta, lr, trial=trial, mark='Q_{c_time}')
        return r2
    
    def _run_learn_(self, model, train_dataloader, test_dataloader, beta, lr, trial=None, mark=''):
        print(model)
        self.model = model
        loss_fn = nn.SmoothL1Loss(beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses, test_losses = self._epoch_run_(model, train_dataloader, test_dataloader, loss_fn, optimizer, trial=trial)
        plot_learning_curve(train_losses, test_losses, prefix=mark)
        r2 = evaluate_model(model, test_dataloader, mark=mark)
        return r2
        
    def _epoch_run_(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=100, trial=None):
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            print(f'##### Epoch {epoch + 1}/{epochs} #####')
            train_losses += [self._train_(train_dataloader, model, loss_fn, optimizer)]
            test_losses += [self._test_(test_dataloader, model, loss_fn, trial, epoch)]
        return train_losses, test_losses
    
    def _train_(self, dataloader, model, loss_fn, optimizer):
        num_batches = len(dataloader)
        model.train()
        train_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= num_batches
        print(f'Train Avg loss: {train_loss}')
        return train_loss
    
    def _test_(self, dataloader, model, loss_fn, trial=None, epoch=None):
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
        pred_list = np.array(pred_list).flatten()
        y_list = np.array(y_list).flatten()
        r2 = r2_score(y_list, pred_list)
        pearson_r = pearsonr(y_list, pred_list)
        print(f'R^2: {r2}')
        print(f'Pearson Correlation: {pearson_r}')
        if trial is not None:
            if r2 < -1 and epoch > 20:
                raise optuna.TrialPruned()
        return test_loss

# If you just want to run the training process, you can use the following code.
# r2 = Objective('SE').run() # or Objective('Q').run()
# print(f'R^2: {r2}')

# Execute an optimization by using an `Objective` instance.
sampler = optuna.samplers.TPESampler(seed=seed_number)
study = optuna.create_study(sampler=sampler, direction='maximize')
objective = Objective('SE')
study.optimize(objective, n_trials=500, n_jobs=1)
'''
SE
trial: beta=0.09178184385648053, lr=2.1059675996980374e-05
R^2: 0.5606043934822083
Pearson Correlation: PearsonRResult(statistic=0.7690863364241467, pvalue=0.0)
'''
# objective_SE = Objective('SE')
# objective_SE.run(beta=0.20792354901416032, lr=0.00013650540364085198)
# torch.save(objective_SE.model, 'SE_model.pth')
'''
trial: beta=0.10824711361774697, lr=6.526571864860588e-05
R^2: 0.7575156092643738
Pearson Correlation: PearsonRResult(statistic=0.8733898155500965, pvalue=0.0)
'''
# objective_Q = Objective('Q')
# objective_Q.run(beta=0.16523405042720546, lr=0.00012158075688545713)
# torch.save(objective_Q.model, 'Q_model.pth')