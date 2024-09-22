# %%
import utils
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

# %%
data = utils.Data(r'history_res\4a34e82a9941473abb734c7be1a92d3d')
res = data.load_res()
sgm_solver = utils.SGM_2D(res, -0.23671718+0.00705658j, 400, 17)
# %%
sol = sgm_solver.run(k=50, show_plot=True)
# %%
for i in range(50):
    sgm_solver.construct_sgm_mesh(i,show_plot=True)
    plt.plot(sgm_solver._vec_raw_)
    plt.show()
# %%
