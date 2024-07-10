# %%
import utils
import numpy as np

# %%
data = utils.Data(r'history_res\e0d13b0f52684955b9d0fe4a6f76ce3a')
res = data.load_res()
C_mat_sum = res['C_mat_sum']
sgm_solver = utils.SGM(C_mat_sum, -0.06045577+0.03j, 200, 3)

# %%
pass