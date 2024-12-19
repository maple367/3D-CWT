import multiprocessing as mp
mp.freeze_support()
### README ###
### Don't run any sentence out of this block, otherwise it will be called by the child process and cause error. ###
import pandas as pd
import mph
import utils
import numpy as np
from model import SGM_solver
client = mph.start(cores=8)
sgm_solver = SGM_solver(client)
import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots(tight_layout=True)
fig.canvas.manager.set_window_title('result preview')
ax2 = ax.twinx()
ax.set_xlabel(r'size ($\mu m$)') 
ax.set_ylabel('Q')
ax.tick_params(axis='y', colors='r')
ax2.set_ylabel('$P_{rad}/P_{stim}$')
ax2.tick_params(axis='y', colors='b')

df = pd.read_csv('GaN_data_set.csv')
Q_list = []
SE_list = []
FF_list = []
size_list = []

line1, = ax.plot(size_list, Q_list, 'r', label='Q', marker='o')
line2, = ax2.plot(size_list, SE_list, 'b', label='$P_{rad}/P_{stim}$', marker='o')
plt.pause(0.1)
uuid = '20241219102059_6be3ff442b5e470b9bcdfe0c58aa6992'
for size in [200,250,300,350,400,450,500]:
    pcsel_model_loader = utils.Data(f'./history_res/{uuid}')
    pcsel_model_loader.load_model()
    try:
        res = pcsel_model_loader.res['cwt_res']
        model_size = int(size/res['a']) # 500 um
        i_eigs_inf = np.argmin(np.real(res['eigen_values']))
        sgm_solver.run(pcsel_model_loader, res['eigen_values'][i_eigs_inf], model_size, 20)
        Q = np.max(res['beta0'].real/(2*sgm_solver.eigen_values.imag))
        i_eigs = np.argmax(res['beta0'].real/(2*sgm_solver.eigen_values.imag))
        SE = sgm_solver.P_rad/sgm_solver.P_stim
        Q_list.append(Q)
        SE_list.append(SE[i_eigs])
        size_list.append(size)
        line1.set_data(size_list, Q_list)
        line2.set_data(size_list, SE_list)
        ax.relim()
        ax.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        plt.draw()
        plt.pause(0.1)
    except:
        continue

plt.ioff()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a = 0.182
FF=np.linspace(0.05,0.25,101)
radius = np.sqrt(FF/np.pi)*a
diameter = 2*radius
sidelength = np.sqrt(2*FF)*a
df = pd.DataFrame({'FF':FF, 'diameter':diameter, 'sidelength':sidelength})
ax = df.plot.line(x='FF', y=['diameter', 'sidelength'])
plt.show()
# %%
