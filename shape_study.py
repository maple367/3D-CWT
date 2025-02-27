import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def seed_everything(seed=233):
    global seed_number
    seed_number = seed
    import random
    random.seed(seed)
    np.random.seed(seed)
seed_everything()


def f(X):
    return np.sin(X[:,0]) + np.abs(X[:,1]) + X[:,2]**2 + X[:,3] + X[:,4]**3 + X[:,5]**4
X = (np.random.rand(100, 13)-0.5)*3
y = f(X)
# explainer = shap.KernelExplainer(model=f, data=X, link="identity")
explainer = shap.explainers.Exact(f, X)
X = pd.DataFrame(X, columns=[fr'$\xi_{i}$' for i in range(X.shape[1])])
shap_values = explainer(X)
shap.plots.violin(shap_values, plot_type="layered_violin", show=False, plot_size=1.0, max_display=10)
plt.show()

# %%