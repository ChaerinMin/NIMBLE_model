import numpy as np
# textured_pkl = "./assets/NIMBLE_TEX_FUV.pkl"
textured_pkl = "utils/NIMBLE_model/assets/NIMBLE_TEX_FUV.pkl"
f_uv = np.load(textured_pkl, allow_pickle=True)
print(f_uv)