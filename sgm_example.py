# %%
import utils
import numpy as np
import scipy.sparse

# %%
data = utils.Data(r'history_res\d1a315a0106e4d02b10b38a192538d00')
res = data.load_res()
sgm_solver = utils.SGM(res, 0-3.12255450e-07j, 50, 3)
import matplotlib.pyplot as plt
plt.imshow(np.real(sgm_solver.C.toarray()))
plt.colorbar()
plt.show()
# %%
sol = sgm_solver.run(k=6, show_plot=True)
# %%
sgm_solver.construct_sgm_mesh(2,show_plot=True)
# %%
