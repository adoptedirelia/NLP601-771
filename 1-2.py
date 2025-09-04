import torch
import torch.nn.functional as F
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from memory_profiler import memory_usage

class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.proj = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


SEQ_LENGTHS = [10, 100, 1000, 10000]
EMBED_DIM = 512
NUM_HEADS = 8
NUM_TRIALS = 5
DEVICE_TYPES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def profile_attention(L, device_type):
    device = torch.device(device_type)
    model = SelfAttention(EMBED_DIM, NUM_HEADS).to(device)
    x = torch.randn(1, L, EMBED_DIM).to(device)

    def run_model():
        with torch.no_grad():
            model(x)

    # Wall time
    start = time.time()
    run_model()
    elapsed = time.time() - start

    # Memory
    if device_type == 'cpu':
        mem_usage = memory_usage((run_model,), interval=0.001, timeout=1)
        peak_mem_mb = max(mem_usage) - min(mem_usage)
    else:
        torch.cuda.reset_peak_memory_stats()
        run_model()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # FLOPs
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU] if device_type == 'cpu'
                   else [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True
    ) as prof:
        run_model()

    total_flops = sum([e.flops for e in prof.key_averages() if e.flops is not None])
    return elapsed, peak_mem_mb, total_flops

results = []

for device_type in DEVICE_TYPES:
    for L in SEQ_LENGTHS:
        times, mems, flops_list = [], [], []

        for _ in range(NUM_TRIALS):
            try:
                t, m, f = profile_attention(L, device_type)
                times.append(t)
                mems.append(m)
                flops_list.append(f)
            except Exception as e:
                print(f"Error at L={L}, device={device_type}: {e}")
                times.append(np.nan)
                mems.append(np.nan)
                flops_list.append(np.nan)

        def sem(x): return np.std(x) / np.sqrt(len(x))

        results.append({
            "Device": device_type.upper(),
            "SeqLength": L,
            "WallTime": np.nanmean(times),
            "WallTimeSEM": sem(times),
            "MemoryMB": np.nanmean(mems),
            "MemoryMBSEM": sem(mems),
            "FLOPs": np.nanmean(flops_list),
            "FLOPsSEM": sem(flops_list)
        })


df = pd.DataFrame(results)


def plot_metric(df, metric, sem_col, ylabel):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="SeqLength", y=metric, hue="Device", marker="o", err_style="bars", errorbar=None)
    for device in df['Device'].unique():
        sub = df[df['Device'] == device]
        plt.errorbar(sub['SeqLength'], sub[metric], yerr=sub[sem_col], fmt='o', capsize=5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Sequence Length (log scale)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Sequence Length")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(f"{metric}.png", bbox_inches="tight")


plot_metric(df, "WallTime", "WallTimeSEM", "Wall Clock Time (s)")
plot_metric(df, "MemoryMB", "MemoryMBSEM", "Memory Usage (MB)")
plot_metric(df, "FLOPs", "FLOPsSEM", "FLOPs")
