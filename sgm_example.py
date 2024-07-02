# %%
import utils

# %%
data = utils.Data(r'history_res\0babe404bd5748eaa98b8a4ea0d9677c')
res = data.load_res()
C_mats = res['C_mats']
C_mat_sum = C_mats['1D'] + C_mats['rad']+C_mats['2D']
sgm_solver = utils.SGM(C_mat_sum, -0.06045577+0.000j, 200, 20)

# %%
sgm_solver.run(10)
sgm_solver.plot()
# %%
