import numpy as np
import os
import pickle

esm_msa = dict()
files = os.listdir("./esm-msa-emb/")
for file_name in files:
    tid = file_name.split(".")[0]
    esm_msa[tid] = np.load(f"./esm-msa-emb/{file_name}",allow_pickle=True)


with open("../../esm-msa/ara_and_eff_esm-msa.pkl","wb") as f:
    pickle.dump(esm_msa,f)


