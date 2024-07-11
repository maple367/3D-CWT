# %%
import utils
import numpy as np
import scipy.sparse

# %%
data = utils.Data(r'history_res\e0d13b0f52684955b9d0fe4a6f76ce3a')
res = data.load_res()
sgm_solver = utils.SGM(res, 0+0j, 200, 25)
# %%
sol = sgm_solver.run(k=50, show_plot=True)
print(sol[0])
# %%
sgm_solver.construct_sgm_mesh(1)
# %%
