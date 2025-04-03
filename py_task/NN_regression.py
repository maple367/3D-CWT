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

def plot_distribution(y_true, y_pred, kde=False, prefix='', clf=True):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    plt.figure(figsize=(8, 6))
    if kde:
        sns.kdeplot(x=y_true, y=y_pred, fill=True, cbar=True, thresh=0.05)
    # Plotting the distribution of true vs predicted values
    plt.scatter(y_true, y_pred, alpha=0.15, label='Test data')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Perfect fit")  # Diagonal line
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.savefig(f'./nn_opt/{prefix}_true_vs_pred.png')
    print(f'True vs Predicted values saved as ./nn_opt/{prefix}_true_vs_pred.png')
    if clf:
        plt.clf()

def plot_error_distribution(y_true, y_pred, bins=50, density=True, cumulative=True, alpha=0.6, color='g', prefix='', clf=True, **kwargs):
    error = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 6))
    plt.hist(error, bins=bins, density=density, cumulative=cumulative, alpha=alpha, color=color, **kwargs)
    plt.xlabel('Error')
    plt.ylabel('Cumulative Frequency')
    plt.title('Error Distribution')
    plt.savefig(f'./nn_opt/{prefix}_error_distribution.png')
    print(f'Error distribution saved as ./nn_opt/{prefix}_error_distribution.png')
    if clf:
        plt.clf()

def plot_learning_curve(train_losses, test_losses, prefix='', clf=True,):
    num_cut = 0
    fig, ax = plt.subplots()
    ax.plot(train_losses[num_cut:], label='Train loss')
    ax.plot(test_losses[num_cut:], label='Test loss', color='orange')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.savefig(f'./nn_opt/{prefix}_learning_curve.png')
    print(f'Learning curve saved as ./nn_opt/{prefix}_learning_curve.png')
    if clf:
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


def evaluate_model(model, dataloader, mark='',clf=True):
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
    plot_distribution(y_list, pred_list, prefix=mark, clf=clf)
    plot_error_distribution(y_list, pred_list, cumulative=True, prefix=mark, clf=clf)
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
        self.early_stopping = EarlyStopping('./nn_opt/SE')
        r2 = self._run_learn_(model, self.train_dataloader, self.test_dataloader, beta, lr, trial=trial, mark=f'SE_{c_time}')
        return r2
    
    def _run_Q_(self, beta=0.1, lr=0.0005, trial=None):
        c_time = time.strftime('%y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        model = FC4Q().to(device)
        self.early_stopping = EarlyStopping('./nn_opt/Q')
        r2 = self._run_learn_(model, self.train_dataloader, self.test_dataloader, beta, lr, trial=trial, mark=f'Q_{c_time}')
        return r2
    
    def _run_learn_(self, model, train_dataloader, test_dataloader, beta, lr, trial=None, mark=''):
        print(model)
        self.model = model
        loss_fn = nn.SmoothL1Loss(beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_losses, self.test_losses = self._epoch_run_(model, train_dataloader, test_dataloader, loss_fn, optimizer, trial=trial)
        plot_learning_curve(self.train_losses, self.test_losses, prefix=mark)
        r2 = evaluate_model(model, test_dataloader, mark=mark)
        return r2
        
    def _epoch_run_(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=100, trial=None):
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            print(f'##### Epoch {epoch + 1}/{epochs} #####')
            train_losses += [self._train_(train_dataloader, model, loss_fn, optimizer)]
            test_losses += [self._test_(test_dataloader, model, loss_fn, trial, epoch)]
                # 早停止
            self.early_stopping(test_losses[-1], model)
            #达到早停止条件时，early_stop会被置为True
            if self.early_stopping.early_stop:
                print("Early stopping")
                break #跳出迭代，结束训练
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

# %%
# If you just want to run the training process, you can use the following code.
# r2 = Objective('SE').run() # or Objective('Q').run()
# print(f'R^2: {r2}')

# Execute an optimization by using an `Objective` instance.
# sampler = optuna.samplers.TPESampler(seed=seed_number)
# study = optuna.create_study(sampler=sampler, direction='maximize')
# objective = Objective('Q')
# study.optimize(objective, n_trials=200, n_jobs=1)
'''
SE
beta=0.12, lr=2e-05
R^2: 0.6100635528564453
'''
# objective_SE = Objective('SE')
# r2_ls = []
# for beta in np.linspace(0.05, 0.15, 11):
#     for lr in np.linspace(1e-5, 1e-4, 10):
#         r2 = objective_SE.run(beta=beta, lr=lr)
#         r2_ls.append(r2)
#         if r2 >= max(r2_ls):
#             print('-------------------')
#             print(f'beta={beta}, lr={lr}')
#             print(f'R^2: {r2}')
#             torch.save(objective_SE.model.state_dict(), 'SE_model.pth')
#             np.save('SE_model.class', objective_SE)
#             print('-------------------')
'''
Q
beta=0.020000000000000004, lr=0.00011999999999999999
R^2: 0.7778209447860718
'''
# objective_Q = Objective('Q')
# r2_ls = []
# for beta in np.linspace(0.01, 0.1, 10):
#     for lr in np.linspace(5e-5, 15e-5, 11):
#         r2 = objective_Q.run(beta=beta, lr=lr)
#         r2_ls.append(r2)
#         if r2 >= max(r2_ls):
#             print('-------------------')
#             print(f'beta={beta}, lr={lr}')
#             print(f'R^2: {r2}')
#             torch.save(objective_Q.model.state_dict(), 'Q_model.pth')
#             np.save('Q_model.class', objective_Q)
#             print('-------------------')

# %%
# plot results
SE_model = FC4SE().to(device)
SE_model.load_state_dict(torch.load('SE_model.pth'))
SE_model.eval()
Q_model = FC4Q().to(device)
Q_model.load_state_dict(torch.load('Q_model.pth'))
Q_model.eval()
SE_data = CustomDataset(test_df['eps_array'].values, test_df['SE'].values)
Q_data = CustomDataset(test_df['eps_array'].values, np.log10(test_df['Q'].values))
SE_dataloader = DataLoader(SE_data, batch_size=64)
Q_dataloader = DataLoader(Q_data, batch_size=64)
# evaluate_model(SE_model, SE_dataloader, mark='SE', clf=False)
# evaluate_model(Q_model, Q_dataloader, mark='Q')

# %%
pred_list = []
y_list = []
with torch.no_grad():
    for x, y in SE_dataloader:
        y_list += list(y.cpu().numpy())
        x, y = x.to(device), y.to(device)
        pred = SE_model(x)
        pred_list += list(pred.cpu().numpy())
pred_list = np.array(pred_list).flatten()
y_list = np.array(y_list).flatten()
rmse = np.sqrt(np.mean((y_list - pred_list)**2))
print(f'RMSE: {rmse}')
plot_distribution(y_list, pred_list, prefix='SE', clf=False)
plt.title('')
plt.xlabel('$SEE_{CWT}$')
plt.ylabel('$SEE_{NN}$')
plt.tight_layout()
plt.show()
plot_error_distribution(y_list, pred_list, cumulative=True, prefix='SE', clf=False)
plt.title('')
plt.xlabel('Error ($\\left|SEE_{CWT}-SEE_{NN}\\right|$)')
plt.ylabel('Cumulative Frequency')
plt.tight_layout()
plt.show()
# %%
objective_SE = np.load('SE_model.class.npy', allow_pickle=True).item()
objective_Q = np.load('Q_model.class.npy', allow_pickle=True).item()

# %%
plot_learning_curve(objective_SE.train_losses, objective_SE.test_losses, clf=False)
plt.title('')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()
# %%
