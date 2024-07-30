# XQ, Noel, 2024.7.3
import numpy as np
np.set_printoptions(linewidth=np.inf)
from scipy.stats import ortho_group
import time

class explore_core():
    def __init__(self, bb, core_step, re_check_y='none', _stop_=False, name='C:'):
        self.name = name
        self.bb = bb
        self.core_step = np.array(core_step)
        self.min_step = self.core_step/10
        self.re_check_y = re_check_y
        self.ite = 0
        self._stop_ = _stop_
        self.local_grow_rate = 'none'
        self.global_grow_rate = 'none'

    def __explore_init__(self):
        self.epoch = 0
        self.cont_flag = 1
        self.step = self.core_step.copy()
        if self.ite == 0: self.dim = len(self.x)
        self.fabri_tol_fac = 1

    def v_n(self, x, n=5):
        try: return float(str('%.'+str(n)+'g') % x)
        except: return [float(str('%.'+str(n)+'g') % i) for i in x]

    def __DR1D1_in__(self):
        self.direction = []
        for self.dire_vec in self.ortho_matrix:
            self.weight, self.state = 1, 1
            while 1:
                self.dx = self.dire_vec*self.step*self.weight
                self.new_x = self.x.copy()+self.dx
                self.new_y = self.bb(self.new_x)

                # self.print_info = self.ite, self.v_n(self.new_y), self.v_n(self.new_x), self.v_n(self.step), self.v_n(self.dx), self.weight, self.state, self.local_grow_rate, self.global_grow_rate
                self.print_info = self.ite, self.v_n(self.new_y,n=3), self.v_n(self.new_x,n=4)
                self.ite += 1
                print('\033[33m',self.name+' ',self.print_info,'\033[0m',sep='') if self.new_y > self.y else print(self.name, self.print_info)

                if self.new_y > self.y:
                    self.x, self.y = self.new_x.copy(), self.new_y
                    if abs(self.weight) == 1:
                        if self.state == 0: break
                        else: self.direction.append([self.state])
                    else: self.step += abs(self.dx/self.weight)/3
                    self.weight *= 2
                else:
                    if abs(self.weight) == 1:
                        if self.state == 0: break
                        elif self.state == 1:
                            self.state = -1
                            self.weight *= -1
                        else:
                            self.step -= abs(self.dx)*0.618
                            self.direction.append([1])
                            break
                    else: self.weight, self.state = round(np.sign(self.weight)), 0

    def __stop_criteria__(self):
        if (self._stop_ == True) and (True not in (self.step >= self.core_step)):
            self.local_grow_rate = (self.y-self.y_clone)/(self.y-self.init_y)
            self.global_grow_rate = (self.y-self.init_y)/(self.std_y-self.init_y) if self.std_y != -np.inf else 1
            if self.global_grow_rate < 0.5 or self.y == -np.inf:
                print(self.name, 'bad initial point')
                self.cont_flag = 0
            elif self.local_grow_rate <= 0.03:
                print(self.name, 'converged end')
                self.cont_flag = 0
            elif self.local_grow_rate <= 0.1:
                if self.global_grow_rate > 0.75: pass
                else:
                    print(self.name, 'no hope to surpass')
                    self.cont_flag = 0

    def DR1D1_out(self, init_x='none', init_y='none', scan_num=1, std_y=-np.inf):
        self.x = init_x
        self.init_y = self.bb(self.x) if init_y == 'none' else init_y
        self.y = self.init_y
        self.std_y = std_y
        self.__explore_init__()
        print(self.name, "initial:", self.y, self.x)
        while 1:
            print(self.name, "epoch", self.epoch)
            self.ortho_matrix = ortho_group.rvs(self.dim)
            self.step = np.maximum(self.step, self.min_step)
            for j in range(scan_num):
                print(self.name, "scan num", j)
                self.y_clone = self.y
                self.__DR1D1_in__()
                self.ortho_matrix *= self.direction
                self.__stop_criteria__()
                if self.cont_flag == 0: break
            if self.cont_flag == 0: break
            self.epoch += 1
        print(self.name, "end:", self.y, self.x)
        try: _ = self.re_check_y[0]
        except:
            self.fabri_tol_fac = self.re_check_y(self.x, self.y)
            print(self.name, "re_check_y:", self.fabri_tol_fac)
        return self.y*self.fabri_tol_fac

class explore_inter():
    def __init__(self, innerlayer, inter_step, _stop_=False, name='I:'):
        self.name = name
        self.innerlayer, self.bb = innerlayer, innerlayer.DR1D1_out
        self.inter_step = np.array(inter_step)
        self.min_step = self.inter_step/10
        self._stop_ = _stop_
        self.ite = 0
        self.local_grow_rate = 'none'
        self.global_grow_rate = 'none'

    def __explore_init__(self):
        self.epoch = 0
        self.cont_flag = 1
        self.step = self.inter_step.copy()
        if self.ite == 0: self.dim = len(self.x)
        self.x_ls = []

    def v_n(self, x, n=5):
        try: return float(str('%.'+str(n)+'g') % x)
        except: return [float(str('%.'+str(n)+'g') % i) for i in x]

    def __DR1D1_in__(self):
        for self.dire_vec in self.ortho_matrix:
            self.weight, self.state = 1, 1
            while 1:
                self.dx = self.dire_vec*self.step*self.weight
                self.new_x = self.x.copy()+self.dx
                self.new_y = self.bb(init_x=self.new_x, std_y=self.y)
                self.new_x = self.innerlayer.x.copy()

                # self.print_info = self.ite, self.v_n(self.new_y), self.v_n(self.new_x), self.v_n(self.step), self.v_n(self.dx), self.weight, self.state, self.local_grow_rate, self.global_grow_rate
                self.print_info = self.ite, self.v_n(self.new_y,n=3), self.v_n(self.new_x,n=4)
                self.ite += 1
                print('\033[33m',self.name+' ',self.print_info,'\033[0m',sep='') if self.new_y > self.y else print(self.name, self.print_info)

                self.true_dire_vec = self.new_x-self.x
                self.true_dire_vec /= np.linalg.norm(self.true_dire_vec)
                dire_adj = 1
                for x_history in self.x_ls:
                    if True not in (abs(self.new_x-x_history) > self.inter_step/10):
                        dire_adj = 0
                        break
                self.x_ls.append(self.new_x)
                if dire_adj == 1:
                    self.dire_vec = self.true_dire_vec.copy()
                    print(self.name, "direction specially adjusted")

                if self.new_y > self.y:
                    self.x_ls = []
                    self.x, self.y = self.new_x, self.new_y
                    if abs(self.weight) == 1:
                        if self.state == 0: break
                        else: self.direction.append([self.state])
                    else: self.step += abs(self.true_dire_vec*self.step)/3
                    self.weight *= 2
                else:
                    if abs(self.weight) == 1:
                        if self.state == 0: break
                        elif self.state == 1:
                            self.state = -1
                            self.weight *= -1
                        else:
                            self.step -= abs(self.true_dire_vec*self.step)*0.618
                            self.direction.append([1])
                            break
                    else: self.weight, self.state = round(np.sign(self.weight)), 0

    def __stop_criteria__(self):
        if (self._stop_ == True) and (True not in (self.step >= self.core_step)):
            self.local_grow_rate = (self.y-self.y_clone)/(self.y-self.init_y)
            self.global_grow_rate = (self.y-self.init_y)/(self.std_y-self.init_y) if self.std_y != -np.inf else 1
            if self.global_grow_rate < 0.5 or self.y == -np.inf:
                print(self.name, 'bad initial point')
                self.cont_flag = 0
            elif self.local_grow_rate <= 0.02:
                print(self.name, 'converged end')
                self.cont_flag = 0
            elif self.local_grow_rate <= 0.1:
                if self.global_grow_rate > 0.75: pass
                else:
                    print(self.name, 'no hope to surpass')
                    self.cont_flag = 0

    def DR1D1_out(self, init_x='none', init_y='none', scan_num=1, std_y=-np.inf):
        if init_y == 'none':
            self.init_y = self.bb(init_x)
            self.x = self.innerlayer.x.copy()
        else: self.x, self.init_y = np.array(init_x), init_y
        self.y = self.init_y
        self.std_y = std_y
        self.__explore_init__()
        print(self.name, "initial:", self.y, self.x)
        while 1:
            print(self.name, "epoch", self.epoch)
            self.ortho_matrix = ortho_group.rvs(self.dim)
            self.step = np.maximum(self.step, self.min_step)
            for j in range(scan_num):
                print(self.name, "scan num", j)
                self.y_clone = self.y
                self.direction = []
                self.__DR1D1_in__()
                self.ortho_matrix *= self.direction
                self.__stop_criteria__()
                if self.cont_flag == 0: break
            if self.cont_flag == 0: break
            self.epoch += 1
        return self.y

class explore_hull():
    def __init__(self, innerlayer, hull_step, pre_check_x='none', name='H:'):
        self.name = name
        self.innerlayer, self.bb = innerlayer, innerlayer.DR1D1_out
        self.hull_step = np.array(hull_step)
        self.min_step = self.hull_step/10
        self.pre_check_x = pre_check_x
        self.ite = 0
        self.begin_time = time.time()

    def __explore_init__(self):
        self.epoch = 0
        self.cont_flag = 1
        self.step = self.hull_step.copy()
        if self.ite == 0: self.dim = len(self.x)
        self.x_ls = []

    def v_n(self, x, n=5):
        try: return float(str('%.'+str(n)+'g') % x)
        except: return [float(str('%.'+str(n)+'g') % i) for i in x]

    def __DR1D1_in__(self):
        for self.dire_vec in self.ortho_matrix:
            self.weight, self.state = 1, 1
            while 1:
                simu_or_not = 1
                try: _ = self.pre_check_x[0]
                except: simu_or_not = self.pre_check_x(self.new_x)
                if simu_or_not == 0:
                    print(self.name, "useless initial point:", self.new_x)
                    self.direction.append([-1])
                    break

                self.dx = self.dire_vec*self.step*self.weight
                self.new_x = self.x.copy()+self.dx
                self.new_y = self.bb(init_x=self.new_x, std_y=self.y)
                self.new_x = self.innerlayer.x.copy()

                # self.print_info = self.ite, self.v_n(self.new_y), self.v_n(self.new_x), self.v_n(self.step), self.v_n(self.dx), self.weight, self.state
                self.print_info = self.ite, self.v_n(self.new_y,n=3), self.v_n(self.new_x,n=4)
                self.ite += 1
                print('\033[33m',self.name+' ',self.print_info,'\033[0m',sep='') if self.new_y > self.y else print(self.name, self.print_info)

                self.true_dire_vec = self.new_x-self.x
                self.true_dire_vec /= np.linalg.norm(self.true_dire_vec)
                dire_adj = 1
                for x_history in self.x_ls:
                    if True not in (abs(self.new_x-x_history) > self.hull_step/10):
                        dire_adj = 0
                        break
                self.x_ls.append(self.new_x)
                if dire_adj == 1:
                    self.dire_vec = self.true_dire_vec.copy()
                    print(self.name, "direction specially adjusted")

                if self.new_y > self.y:
                    self.x_ls = []
                    self.x, self.y = self.new_x, self.new_y
                    if abs(self.weight) == 1:
                        if self.state == 0: break
                        else: self.direction.append([self.state])
                    else: self.step += abs(self.true_dire_vec*self.step)/3
                    self.weight *= 2
                else:
                    if abs(self.weight) == 1:
                        if self.state == 0: break
                        elif self.state == 1:
                            self.state = -1
                            self.weight *= -1
                        else:
                            self.step -= abs(self.true_dire_vec*self.step)*0.618
                            self.direction.append([1])
                            break
                    else: self.weight, self.state = round(np.sign(self.weight)), 0

    def __stop_criteria__(self):
        self.time_consumed = self.begin_time - time.time()
        if self.time_consumed > 86400*365*10000:
            print('100 century gone, end')
            self.cont_flag = 0
        else: print('\033[94mtime consumed:', self.time_consumed, '\033[0m')

    def DR1D1_out(self, init_x='none', init_y='none', scan_num=1, std_y=-np.inf):
        if init_y == 'none':
            self.y = self.bb(init_x)
            self.x = self.innerlayer.x.copy()
        else: self.x, self.y = np.array(init_x), init_y
        self.std_y = std_y
        self.__explore_init__()
        print(self.name, "initial:", self.y, self.x)
        while 1:
            print(self.name, "epoch", self.epoch)
            self.ortho_matrix = ortho_group.rvs(self.dim)
            self.step = np.maximum(self.step, self.min_step)
            for j in range(scan_num):
                print(self.name, "scan num", j)
                self.y_clone = self.y
                self.direction = []
                self.__DR1D1_in__()
                self.ortho_matrix *= self.direction
                self.__stop_criteria__()
                if self.cont_flag == 0: break
            if self.cont_flag == 0: break
            self.epoch += 1
        return self.y

class sandwich():
    def __init__(self, bb, core_step, init_x, init_y='none', pre_check_x='none', re_check_y='none'):
        # bb: input parameters, output 1 score, required
        # core_step: initial step at the core layer of the algorithm, required
        # init_x: initial parameters, required
        # init_y: the score of init_x, not required
        # pre_check_x: input parameters, output 1 if the parameters worth simulation otherwise 0, not required
        # re_check_y: input parameters, output 0~1 and multiply it directly with the score to generate the final score, not required
        init_x, core_step = np.array(init_x), np.array(core_step)
        self.core_layer = explore_core(bb=bb, core_step=core_step, re_check_y=re_check_y, _stop_=True)
        self.inter_layer = explore_inter(innerlayer=self.core_layer, inter_step=core_step*10, _stop_=True)
        self.hull_layer = explore_hull(innerlayer=self.inter_layer, hull_step=core_step*100, pre_check_x=pre_check_x)
        self.hull_layer.DR1D1_out(init_x=init_x, init_y=init_y)
