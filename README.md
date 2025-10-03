
# BKSEF — Balanced Kernel-Size Estimation for CNNs

BKSEF is a practical method to choose convolution kernel sizes per block/stage by balancing three forces:

Information gain (larger receptive fields capture more context),

Expected accuracy gain (diminishing returns at larger kernels),

Compute cost (FLOPs/MACs grow with 
𝑘
2).

This repo contains reproducible case studies showing how BKSEF compares to a fixed-kernel baseline on:

CIFAR-10 (cloud-ish setting), and

GTSRB (edge setting). 


## Deployment

To deploy Case Study A — CIFAR-10 run (CPU-friendly (Windows))

```bash
python -u "Case Study A (CIFAR-10).py" --epochs 10 --batch_size 128 --no_cuda

```

What you’ll see:

BKSEF decisions per block, e.g. [Stage 1 Block 1] ... -> k*=7; detail={...}

Training logs each epoch: train loss/acc, test loss/acc, best test acc.

Latency at the end (avg ms/inference).

Case Study B — GTSRB (edge)
```bash
python -u "case_study_B_Edge case (GTSRB).py" --epochs 10 --force_cpu

```
What you’ll see:

BKSEF picks kernels (late stage may allow k=1 to cut MACs),

Per-epoch logs and final latency on CPU.