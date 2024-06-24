# %%
import matlab.engine
eng = matlab.engine.start_matlab()

# %%
a = []
for i in range(10000):
    future = eng.sqrt(4.0,background=True)
    a.append(future)
for future in a:
    ret = future.result()
    print(ret)
# %%
