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

import utils
import numpy as np
from model import AlxGaAs, rect_lattice, model_parameters, Model, CWT_solver, SGM_solver
def run_simu(eps_array, sgm_solver:SGM_solver):
    Al_x =         [0.0,  0.0,  0.4,   0.191, 0.45]
    t_list =       [0.35, 0.08, 0.025, 0.116, 2.11]
    is_phc =       [True, False,False, False, False]
    is_no_doping = [False,False,False, True,  False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            mat_list.append(rect_lattice.eps_mesh(eps_array))
        else:
            mat_list.append(AlxGaAs(Al_x[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[17.7, -3.23, 8.28, 2.00]}
    paras = model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/0.98) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = Model(paras)
    # pcsel_model.plot()
    cwt_solver = CWT_solver(pcsel_model)
    cwt_solver.core_num = 40 # Because the limitation of Windows, the core_num should be smaller than 61
    cwt_solver.run(10, parallel=True)
    res = cwt_solver.save_dict
    model_size = int(200/cwt_solver.a) # 200 um
    i_eigs_inf = np.argmin(np.real(res['eigen_values']))
    try:
        sgm_solver.run(pcsel_model, res['eigen_values'][i_eigs_inf], model_size, 20)
        Q = np.max(res['beta0'].real/(2*sgm_solver.eigen_values.imag))
        i_eigs = np.argmax(res['beta0'].real/(2*sgm_solver.eigen_values.imag))
        SE = 1-sgm_solver.P_edge/sgm_solver.P_stim
        SE = SE[i_eigs]
    except:
        # bad input parameter, the model is not converge
        Q = np.nan
        SE = np.nan
    pcsel_model.save()
    data_set = {'Q': Q, 'SE': SE, 'uuid': paras.uuid, 't11': pcsel_model.tmm.t_11, 'time_cost': cwt_solver._pre_cal_time}
    return data_set

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, theta):
    x = x-x0
    y = y-y0
    x_theta = x*np.cos(theta) + y*np.sin(theta)
    y_theta = -x*np.sin(theta) + y*np.cos(theta)
    return np.exp(-(x_theta**2/(2*sigma_x**2) + y_theta**2/(2*sigma_y**2)))
        
def generate_sample_array(x_size, y_size, num_holes, x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s):
    XX, YY = np.meshgrid(np.linspace(0, 1, x_size), np.linspace(0, 1, y_size))
    z = 0.0
    for i in range(num_holes):
        x0 = x0_s[i]
        y0 = y0_s[i]
        sigma_x = sigma_x_s[i]
        sigma_y = sigma_y_s[i]
        theta = theta_s[i]
        z += gaussian_2d(XX, YY, x0, y0, sigma_x, sigma_y, theta)
    return z

if __name__ == '__main__':

    import mph
    client = mph.start(cores=8)
    GaAs_eps = AlxGaAs(0).epsilon
    sgm_solver = SGM_solver(client)

    def run_validation(x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s, FF):
        num_holes = 3
        # x0_s = np.random.rand(num_holes)*0.2+np.array([0.15, 0.15, 0.65])
        # y0_s = np.random.rand(num_holes)*0.2+np.array([0.15, 0.65, 0.65])
        # sigma_x_s = np.random.rand(num_holes)*0.1+0.05
        # sigma_y_s = np.random.rand(num_holes)*0.1+0.05
        # theta_s = np.random.rand(num_holes)*2*np.pi
        # FF = np.random.rand()*0.1+0.25
        eps_sample = generate_sample_array(32*10, 32*10, num_holes, x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s)
        eps_thresh = np.percentile(eps_sample, (1-FF)*100)
        eps_array = np.where(eps_sample<eps_thresh, GaAs_eps, 1.0)
        eps_array = eps_array.reshape(32,10,32,10)
        eps_array = eps_array.mean(axis=(1,3))
        res = run_simu(eps_array, sgm_solver)
        return res
    
    # gpu or cpu
    device = 'cpu'
    print(f'Using {device}.')

    SE_model = FC4SE().to(device)
    SE_model.load_state_dict(torch.load('SE_model.pth'))
    SE_model.eval()
    Q_model = FC4Q().to(device)
    Q_model.load_state_dict(torch.load('Q_model.pth'))
    Q_model.eval()

    def f_SE(x):
        with torch.no_grad():
            x = torch.tensor(x).reshape(-1,481,2)
            pred = SE_model(x)
            y = pred.numpy()
        return y.flatten()

    def f_Q(x):
        with torch.no_grad():
            x = torch.tensor(x).reshape(-1,481,2)
            pred = Q_model(x)
            y = pred.numpy()
        return y.flatten()

    def process_data(eps_array):
        eps_array = (np.fft.fftshift(np.fft.fft2(eps_array)/1024)).astype(np.complex64)
        flited = []
        for i in range(32):
            for j in range(32):
                if i <= 16:
                    if j >= 32 - i:
                        flited.append(eps_array[i][j])
                else:
                    if j >= 33 - i:
                        flited.append(eps_array[i][j])
        eps_array = np.array(flited)
        real_part = np.real(eps_array)
        imaginary_part = np.imag(eps_array)
        data = np.stack((real_part, imaginary_part), axis=-1)
        return data

    def run_prediction(x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s, FF):
        num_holes = 3
        eps_sample = generate_sample_array(32*10, 32*10, num_holes, x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s)
        eps_thresh = np.percentile(eps_sample, (1-FF)*100)
        eps_array = np.where(eps_sample<eps_thresh, GaAs_eps, 1.0)
        eps_array = eps_array.reshape(32,10,32,10)
        eps_array = eps_array.mean(axis=(1,3))
        data = process_data(eps_array)
        SE_pred = f_SE(data)
        Q_pred = 10**f_Q(data)
        return SE_pred, Q_pred
    
    def preview(x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s, FF):
        num_holes = 3
        eps_sample = generate_sample_array(32*10, 32*10, num_holes, x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s)
        eps_thresh = np.percentile(eps_sample, (1-FF)*100)
        eps_array = np.where(eps_sample<eps_thresh, GaAs_eps, 1.0)
        eps_array = eps_array.reshape(32,10,32,10)
        eps_array = eps_array.mean(axis=(1,3))
        return eps_array

    def objective_function(SEE, Q, Target=1.0, lambda_penalty=5e-5):
        """
        Objective function for optimizing photonic crystal design.
        
        Parameters:
            SEE (float): Surface-emitting efficiency (should be maximized).
            Q (float): Quality factor (should be greater than 10^4).
            lambda_penalty (float): Weight for penalty on Q constraint.
            
        Returns:
            float: Objective function value (to be minimized).
        """
        penalty = max(0, 1e4 - Q)  # Apply penalty if Q < 1e4
        loss = abs(Target-SEE) + lambda_penalty * penalty
        return loss

    def opt_fun(x):
        SE, Q = run_prediction(x[0:3], x[3:6], x[6:9], x[9:12], x[12:15], x[15])
        return objective_function(SE, Q, Target=1.0)


    # i_iter = 0
    # while i_iter <= 1000:
    #     num_holes = 3
    #     x0_s = np.random.rand(num_holes)*0.2+np.array([0.15, 0.15, 0.65])
    #     y0_s = np.random.rand(num_holes)*0.2+np.array([0.15, 0.65, 0.65])
    #     sigma_x_s = np.random.rand(num_holes)*0.1+0.05
    #     sigma_y_s = np.random.rand(num_holes)*0.1+0.05
    #     theta_s = np.random.rand(num_holes)*2*np.pi
    #     FF = np.random.rand()*0.1+0.25
    #     SE, Q = run_prediction(x0_s, y0_s, sigma_x_s, sigma_y_s, theta_s, FF)
    #     print(SE, Q)
    #     objective_function(SE, Q)
    #     i_iter += 1

    # %%
    from scipy.optimize import direct, differential_evolution, dual_annealing, shgo, basinhopping
    bounds = []
    bounds += [(0.15, 0.35)]+[(0.15, 0.35)]+[(0.65, 0.85)]
    bounds += [(0.15, 0.35)]+[(0.65, 0.85)]+[(0.65, 0.85)]
    bounds += [(0.05, 0.15)]*6
    bounds += [(0.0, 2*np.pi)]*3
    bounds += [(0.25, 0.35)]
    time_start = time.time()
    direct_res = direct(opt_fun, bounds, maxiter=1000, vol_tol=1e-64)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    print(direct_res)

    x = direct_res.x
    params = [x[0:3], x[3:6], x[6:9], x[9:12], x[12:15], x[15]]
    prediction_res = run_prediction(*params)
    print(prediction_res)
    # validation_res = run_validation(*params)
    # print(validation_res)

    # %%
    plt.imshow(np.real(preview(*params)), cmap='Greys')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('preview.png')
    plt.show()