import model
import utils
import model.rect_lattice
from model import AlxGaAs, user_defined_material
import numpy as np
import matplotlib.pyplot as plt

def run_simu(FF,eps0, eps1, eps2, eps3, eps4, eps5, t0, t1, t2, t3, t4, t5 ,c1,c2,c3,c4,plot=True):
    rel_r = np.sqrt(FF/np.pi)
    # eps_ls = np.array([3.499, 3.349, 3.307, 3.265,3.4121,3.205])**2
    # t_list = [0.15, 0.15, 0.05, 0.03,0.154,2.5]
    grad_eps = np.linspace(eps0, 3.349**2, 7)[1:-1]
    grad_t = np.linspace(0.02, 0.02, 7)[1:-1]
    eps_ls = np.array([1.6**2, eps0, eps1, *grad_eps, eps2, eps3, eps4, eps5])
    t_list = [0.2, t0, t1, *grad_t, t2, t3, t4, t5]
    is_phc = [False, True, True, *[True]*5, True, True, False, False]
    is_no_doping = [False, False, False, *[False]*5, False, False, True, False]
    mat_list = []
    for i in range(len(is_phc)):
        if is_phc[i]:
            mat_list.append((model.rect_lattice.eps_circle(rel_r, user_defined_material(eps_ls[i]))))
        else:
            mat_list.append(user_defined_material(eps_ls[i]))
    doping_para = {'is_no_doping':is_no_doping,'coeff':[c1, c2, c3, c4]}
    paras = model.model_parameters((t_list, mat_list, doping_para), surface_grating=True, k0=2*np.pi/1.31) # input tuple (t_list, eps_list, index where is the active layer)
    pcsel_model = model.Model(paras)
    if plot: plot_model(pcsel_model)
    return pcsel_model

def plot_model(input_model:model.Model):
    z_mesh = np.linspace(input_model.tmm.z_boundary[0], input_model.tmm.z_boundary[-1]+0.5, 5000)
    E_profile_s = input_model.e_normlized_intensity(z=z_mesh)
    dopings = input_model.doping(z=z_mesh)
    eps_s = input_model.eps_profile(z=z_mesh)
    E_profile_s = E_profile_s / np.max(np.abs(E_profile_s)) * (np.max(np.abs(input_model.paras.avg_epsilons)) - np.min(np.abs(input_model.paras.avg_epsilons))) + np.min(np.abs(input_model.paras.avg_epsilons))
    a_const = input_model.paras.cellsize_x
    x_mesh = np.linspace(0, a_const, 500)
    y_mesh = np.linspace(0, a_const, 500)
    z_points = np.array([(input_model.phc_boundary_l[-1]+input_model.phc_boundary_r[-1])/2,]) # must be a vector
    XX, YY = np.meshgrid(x_mesh, y_mesh)
    eps_mesh_phc = input_model.eps_profile(XX, YY, z_points)[0]

    color1, color2, fontsize1, fontsize2, fontname = 'mediumblue', 'firebrick', 13, 18, 'serif'
    fig, ax0 = plt.subplots(figsize=(7,5))
    fig.subplots_adjust(left=0.12, right=0.86)
    ax1 = plt.twinx()
    ax0.plot(z_mesh, dopings, color=color1)
    ax0.tick_params(axis='y', colors=color1, labelsize=10)
    ax1.plot(z_mesh, eps_s, linestyle='--', color=color2)
    ax1.plot(z_mesh, E_profile_s, linestyle='--')
    ax1.fill_between(z_mesh, np.min(E_profile_s), E_profile_s, where=input_model.is_in_phc(z_mesh), alpha=0.4, hatch='//', color='orange')
    ax1.tick_params(axis='y', colors=color2, labelsize=10)
    ax0.set_xlabel(r'z ($\mu m$)', fontsize=fontsize1, fontname=fontname)
    ax0.set_ylabel(r'Doping ($\mu m^{-3}$)', fontsize=fontsize1, fontname=fontname, color=color1)
    ax0.set_yscale('symlog', linthresh=np.min(dopings[dopings!=0.0]))
    ax1.set_ylabel(r'$\epsilon_r$ and Normalized $|E|^2$', fontsize=fontsize1, fontname=fontname, color=color2)
    plt.title('', fontsize=fontsize2, fontname=fontname)

    ax2 = ax0.inset_axes([0.65, 0.10, 0.24, 0.24])
    im = ax2.imshow(np.real(eps_mesh_phc), cmap='Greys')
    ax2.set_xticks([])
    ax2.set_yticks([])
    cb = fig.colorbar(im, cax=ax2.inset_axes([0, 1.05, 1, 0.2]), orientation='horizontal', label='Epsilon')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    plt.show()
    plt.close()

def start_solver(cores=8):
    import mph
    client = mph.start(cores=cores)
    semi_solver = 'None'
    # semi_solver = model.SEMI_solver(client)
    sgm_solver = model.SGM_solver(client)
    return semi_solver, sgm_solver

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


class PSO():
    """
    PSO to find the maximum of a function
    """
    def __init__(self, func, dimension, time, size, low, up, v_low, v_high):
        # 初始化
        self.func = func  # 适应函数
        self.dimension = dimension  # 变量个数
        self.time = time  # 迭代的代数
        self.size = size  # 种群大小
        self.bound = []  # 变量的约束范围
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.dimension))[0]  # 全局最优的位置
        self.curval_p_best = np.zeros((self.size))  # 每个粒子最优的位置值
        self.curval_g_best = np.zeros((1))[0]  # 目前全局最优的值

        # 初始化第0代初始全局最优解
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]  # 储存最优的个体
            self.curval_p_best[i] = self.fitness(self.p_best[i])
            # 做出修改
            if self.curval_p_best[i] > temp:
                self.g_best = self.p_best[i]
                temp = self.curval_p_best[i]
        self.curval_g_best = self.fitness(self.g_best)

    def fitness(self, x):
        """
        个体适应值计算
        """
        y = self.func(x)
        # print(y)
        return y

    def update(self, size):
        c1 = 2.0  # 学习因子
        c2 = 2.0
        w = 0.8  # 自身权重因子
        for i in range(size):
            # 更新速度(核心公式)
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            fitness_xi = self.fitness(self.x[i])
            if fitness_xi > self.curval_p_best[i]:
                self.curval_p_best[i] = fitness_xi
                self.p_best[i] = self.x[i]
            if fitness_xi > self.curval_g_best:
                self.curval_g_best = fitness_xi
                self.g_best = self.x[i]

    def pso(self):
        best = []
        self.final_best = np.array([3.499**2,3.349**2,0.15,0.3])
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            print('当前最佳位置：{}'.format(self.final_best))
            temp = self.fitness(self.final_best)
            print('当前的最佳适应度：{}'.format(temp))
            best.append(temp)
        t = [i for i in range(self.time)]
        plt.figure()
        plt.grid(axis='both')
        plt.plot(t, best, color='red', marker='.', ms=10)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次数")  # X轴标签
        plt.ylabel(u"适应度")  # Y轴标签
        plt.title(u"迭代过程")  # 标题
        plt.show()


# if __name__ == '__main__':
#     time = 50
#     size = 100
#     dimension = 5
#     v_low = -1
#     v_high = 1
#     low = [1, 1, 1, 1, 1]
#     up = [25, 25, 25, 25, 25]
#     pso = PSO(dimension, time, size, low, up, v_low, v_high)
#     pso.pso()

if __name__ == '__main__':
    ### README ###
    ### Don't run any sentence out of this block, otherwise it will be called by the child process and cause error. ###
    import multiprocessing as mp
    mp.freeze_support()
    eps0, eps1, t0, dt = [12.243001,11.86747719,0.3,0.3]
    res_final_best = run_simu(0.19, eps0, eps1, 3.307**2, 3.265**2, 3.4121**2, 3.205**2, t0, dt-t0, 0.05, 0.03, 0.154, 2.5, 17.7, -3.23, 8.28, 2.00, plot=True)
    cwt_solver = model.CWT_solver(res_final_best)
    cwt_solver.run()