import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import functional as F
import tqdm
from torch.optim import lr_scheduler
import pytorch_ssim
import matplotlib.pyplot as plt
import net
import time

test_time = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = net.FSnet(input_channel=3).eval().to(device)
test_tensor = torch.randn(1, 3, 256, 256).to(device)
total_time = 0
with torch.no_grad():
    for i in tqdm.tqdm(range(test_time)):
        start_time = time.time()
        output = model(test_tensor)
        end_time = time.time()
        total_time += end_time - start_time

print(total_time/test_time)