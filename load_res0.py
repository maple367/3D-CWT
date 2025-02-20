import utils
import pandas as pd
import numpy as np

data_set_file = 'mesh_data_set_3hole_lessscale.csv'
suffix = ''
df = pd.read_csv(data_set_file)
# df.plot.scatter(x='Q', y='SE', marker='o', s=3)
# plt.xscale('log')
# plt.show()
# %%
df = df.dropna()
i = 0
datas = []

trans_mat = np.zeros((32,32))
trans_mat[0] = 1
trans_mat[-1] = 1
trans_mat[:,0] = 1
trans_mat[:,-1] = 1

for line in df.values:
    para = utils.Data(f'./history_res/{line[2]}').load_para()
    # res = utils.Data(f'./history_res/{line[2]}').load_model()
    eps_array = para['materials'][0].eps_distribtion_array
    # eps_array_s = [eps_array]
    eps_array_s = [np.rot90(eps_array, k=i) for i in range(4)]
    eps_array_s = [*[np.fliplr(eps_array) for eps_array in eps_array_s], *eps_array_s]
    for pseduo_eps_array in eps_array_s:
        datas.append({'eps_array': pseduo_eps_array, 'Q': line[0], 'SE': line[1]})
    i += 1
    print(i)
np.savez_compressed(f'{data_set_file.removesuffix('.csv')}{suffix}.npz', pd.DataFrame(datas).sample(frac=1))
# %%
arrays = pd.DataFrame(datas)['eps_array'].values
array_tuples = {tuple(arr.flatten()) for arr in arrays}

# 判断是否有重复
print(f'Unique: {len(array_tuples)} in {len(arrays)}')
if len(array_tuples) == len(arrays):
    print("没有重复的array")
else:
    print("有重复的array")
