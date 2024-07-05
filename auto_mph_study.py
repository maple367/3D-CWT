# %%
import mph
import numpy as np
client = mph.start(cores=8,version='6.1')

# %%
# parameters
model = client.load(r'D:\Desktop\DQW-1D.mph')

# %%
input_para = model/'parameters'/'input_para'
parameters_dict_show = model.parameters(True)
x_fraction = model/'functions'/'x_fraction'
x_fraction_expr = x_fraction.property('expr')
# %%
