import utils
import pandas as pd
import numpy as np

data_set_file = 'mesh_data_set_3hole_lessscale.csv'
df = pd.read_csv(data_set_file)
# df.plot.scatter(x='Q', y='SE', marker='o', s=3)
# plt.xscale('log')
# plt.show()
# %%
df = df.dropna()
i = 0
datas = []
for line in df.values:
    para = utils.Data(f'./history_res/{line[2]}').load_para()
    res = utils.Data(f'./history_res/{line[2]}').load_model()
    eps_array = para['materials'][0].eps_distribtion_array
    eps_array_s = [eps_array, eps_array.T, np.flip(eps_array,0), np.flip(eps_array.T,0), np.flip(eps_array,1), np.flip(eps_array.T,1), np.flip(eps_array), np.flip(eps_array.T)]
    for pseduo_eps_array in eps_array_s:
        datas.append({'eps_array': pseduo_eps_array, 'Q': line[0], 'SE': line[1]})
    i += 1
    print(i)
np.savez_compressed(f'{data_set_file.removesuffix('.csv')}.npz', pd.DataFrame(datas).sample(frac=1))