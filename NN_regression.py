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
# %%
df = pd.DataFrame(all_data, columns=['eps_array', 'Q', 'SE'])

df = df[(df['SE'] <= 1.0) & (df['SE'] >= 0.0) & (df['Q'] >= 0.0)]
df['eps_array'] = df['eps_array'].apply(lambda x: ((np.fft.fft2(x)/1024).astype(np.complex64)))
df['SE'] = df['SE'].astype(np.float32)
df['Q'] = df['Q'].astype(np.float32)

train_df = df.sample(frac=0.95, random_state=seed_number)
test_df = df.drop(train_df.index)
# %%
# visualization
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


class FC4Q(nn.Module):
    def __init__(self):
        super(FC4Q, self).__init__()
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
        return x


def plot_learning_curve(train_losses, test_losses):
    import matplotlib.pyplot as plt
    num_cut = 0
    fig, ax = plt.subplots()
    ax.plot(train_losses[num_cut:], label='train')
    ax.plot(test_losses[num_cut:], label='test', color='orange')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.show()

def evaluate_model(model, dataloader):
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
    plot_distribution(y_list, pred_list)
    plot_error_distribution(y_list, pred_list, cumulative=True)
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
        self.func = self._run_SE_ if func == 'SE' else self._run_Q_

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        beta = trial.suggest_float('beta', 0.01, 1.0)
        lr = trial.suggest_float('lr', 0.0001, 0.01)
        print(f'trial: beta={beta}, lr={lr}')
        result = self.func(beta, lr, trial)
        print('result:', result)
        return 1 - result
    
    def run(self, beta=0.1, lr=0.0005, trial=None):
        return self.func(beta, lr, trial)
    
    def _run_SE_(self, beta=0.1, lr=0.0005, trial=None):
        train_dataloader, test_dataloader = generate_dataloader(CustomDataset(train_df['eps_array'].values, train_df['SE'].values),
                                                                CustomDataset(test_df['eps_array'].values, test_df['SE'].values),
                                                                batch_size=64)
        model = FC4SE().to(device)
        r2 = self._run_learn_(model, train_dataloader, test_dataloader, beta, lr, trial=trial)
        return r2
    
    def _run_Q_(self, beta=0.1, lr=0.0005, trial=None):
        train_dataloader, test_dataloader = generate_dataloader(CustomDataset(train_df['eps_array'].values, np.log10(1/train_df['Q'].values)),
                                                                CustomDataset(test_df['eps_array'].values, np.log10(1/test_df['Q'].values)),
                                                                batch_size=64)
        model = FC4Q().to(device)
        r2 = self._run_learn_(model, train_dataloader, test_dataloader, beta, lr, trial=trial)
        return r2
    
    def _run_learn_(self, model, train_dataloader, test_dataloader, beta, lr, trial=None):
        print(model)
        loss_fn = nn.SmoothL1Loss(beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses, test_losses = self._epoch_run_(model, train_dataloader, test_dataloader, loss_fn, optimizer, trial=trial)
        r2 = evaluate_model(model, test_dataloader)
        return r2
        
    def _epoch_run_(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=100, trial=None):
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            print(f'##### Epoch {epoch + 1}/{epochs} #####')
            train_losses += [self._train_(train_dataloader, model, loss_fn, optimizer)]
            test_losses += [self._test_(test_dataloader, model, loss_fn, trial, epoch)]
        plot_learning_curve(train_losses, test_losses)
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
        print(f'Train Avg loss: {loss}')
        return loss.item
    
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
            trial.report(r2, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return test_loss

# If you just want to run the training process, you can use the following code.
# r2 = Objective('SE').run() # or Objective('Q').run()
# print(f'R^2: {r2}')

# Execute an optimization by using an `Objective` instance.
sampler = optuna.samplers.TPESampler(seed=seed_number)
study = optuna.create_study(sampler=sampler, pruner=optuna.pruners.MedianPruner(), direction='maximize')
study.optimize(Objective('SE'), n_trials=100, n_jobs=1)
