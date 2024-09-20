# %%
import utils
import numpy as np
import scipy.sparse

# %%
data = utils.Data(r'history_res\babcca1bbbaa4a8f94d7b0421d70f9fa')
res = data.load_res()
sgm_solver = utils.SGM(res, -0.24028528+0.00903809j, 400, 17)
import matplotlib.pyplot as plt
plt.imshow(np.real(sgm_solver.C.toarray()))
plt.colorbar()
plt.show()
# %%
sol = sgm_solver.run(k=100, show_plot=True)
# %%
for i in range(50):
    sgm_solver.construct_sgm_mesh(i,show_plot=True)
# %%
