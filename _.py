import torch
import numpy as np

label = np.array([5.4], dtype=np.float32)[0]
label = label.astype(np.int32)
label = torch.tensor(label, dtype=torch.long)
print(label)
