# import torch
# import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using:", device)

# frames = 1000
# height = 608
# width = 1024
# bin_y, bin_x = 32, 32

# # Generate synthetic data
# x = torch.rand(frames, height, width, device=device)

# start = time.time()
# binned = x.view(frames, height // bin_y, bin_y, width // bin_x, bin_x).mean(dim=(2, 4))
# end = time.time()

# print("Binned shape:", binned.shape)
# print(f"Time taken: {end - start:.4f} sec")


import time, torch, torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frames, H, W = 1000, 608, 1024
by, bx = 32, 32

x = torch.rand(frames, H, W, device=device, dtype=torch.float32)

# warmup
for _ in range(3):
    _ = x.unsqueeze(1)  # (N,1,H,W)
    _ = F.avg_pool2d(_, kernel_size=(by, bx), stride=(by, bx))

if device.type == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()
y = F.avg_pool2d(x.unsqueeze(1), kernel_size=(by, bx), stride=(by, bx)).squeeze(1)  # (N, H//by, W//bx)
if device.type == "cuda": torch.cuda.synchronize()
print("Binned shape:", tuple(y.shape), "time:", time.perf_counter() - t0)
