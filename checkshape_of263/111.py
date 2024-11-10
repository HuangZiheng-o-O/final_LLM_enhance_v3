
import numpy as np
import  torch
rawdata = torch.tensor(np.load(
    "/Users/huangziheng/PycharmProjects/final_LLM_enhance_v4/S-shape of walk_and_wave/raw/raw_sample0_repeat0_len128.npy"),
                       dtype=torch.float32)

print(rawdata.shape)