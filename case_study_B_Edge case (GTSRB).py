# case_study_B_one_shot_compare.py
# One-shot script: BKSEF-selected CNN vs fixed-k baseline on GTSRB (Edge scenario)
# Windows/CPU friendly defaults (workers=0, pin_memory=False). Use --force_cpu to pin to CPU.

import math, time, random, argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.inference_mode()
def latency_ms(model, input_shape, device, warmup=30, iters=200):
    model.eval()
    x = torch.randn(*input_shape, device=device)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(sum(times) / len(times))

def print_table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    sep = "-+-".join("-"*w for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))

# -----------------------------
# BKSEF core (Eq. 1)
# J(k) = λ1 * I'(k) + λ2 * A'(k) - λ3 * C'(k)
# I(k)=log(1+βk), A(k)=1-exp(-αk), C(k)=H*W*C_in*C_out*k^2 (MACs)
# normalization (′) is min–max within the candidate set for that layer
# -----------------------------
def information_gain(k: int, beta: float) -> float:
    return math.log(1.0 + beta * float(k))

def accuracy_gain(k: int, alpha: float) -> float:
    return 1.0 - math.exp(-alpha * float(k))

def compute_cost_macs(k: int, h: int, w: int, cin: int, cout: int) -> float:
    return float(h * w * cin * cout * (k ** 2))

def _minmax(xs: List[float]) -> List[float]:
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]

def normalize_terms(kset: List[int], h: int, w: int, cin: int, cout: int,
                    alpha: float, beta: float) -> Dict[int, Tuple[float, float, float]]:
    I = [information_gain(k, beta) for k in kset]
    A = [accuracy_gain(k, alpha) for k in kset]
    C = [compute_cost_macs(k, h, w, cin, cout) for k in kset]
    I_p, A_p, C_p = _minmax(I), _minmax(A), _minmax(C)
    return {k: (I_p[i], A_p[i], C_p[i]) for i, k in enumerate(kset)}

def J_score(Ip: float, Ap: float, Cp: float, lam1: float, lam2: float, lam3: float) -> float:
    return lam1 * Ip + lam2 * Ap - lam3 * Cp

def choose_k_for_layer(kset: List[int], h: int, w: int, cin: int, cout: int,
                       lam1: float, lam2: float, lam3: float, alpha: float, beta: float):
    stats = normalize_terms(kset, h, w, cin, cout, alpha, beta)
    best_k, best_J = None, -1e9
    detail = {}
    for k, (Ip, Ap, Cp) in stats.items():
        J = J_score(Ip, Ap, Cp, lam1, lam2, lam3)
        detail[k] = {"I'": Ip, "A'": Ap, "C'": Cp, "J": J}
        if J > best_J:
            best_J, best_k = J, k
    return best_k, detail

# -----------------------------
# Models: BKSEF CNN and Baseline CNN
# -----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, s: int = 1, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = k // 2  # same padding for odd k
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BKSEF_CNN(nn.Module):
    """
    3-stage CNN for edge: leaner channel sizes; stage-wise k is chosen via BKSEF (Eq. 1).
    """
    def __init__(self,
                 in_ch: int,
                 num_classes: int,
                 input_hw: Tuple[int,int],
                 stage_channels: List[int],
                 stage_blocks: List[int],
                 stage_strides: List[int],
                 k_candidates: List[List[int]],
                 lam: Tuple[float,float,float],
                 alpha: float,
                 beta: float,
                 print_decisions: bool = True):
        super().__init__()
        assert len(stage_channels) == len(stage_blocks) == len(stage_strides) == len(k_candidates)
        H, W = input_hw
        cin = in_ch
        layers = []
        for s_idx, (cout, B, stride, kset) in enumerate(zip(stage_channels, stage_blocks, stage_strides, k_candidates)):
            for b in range(B):
                s = stride if b == 0 else 1
                k_best, detail = choose_k_for_layer(kset, H, W, cin, cout, lam[0], lam[1], lam[2], alpha, beta)
                if print_decisions:
                    print(f"[Stage {s_idx+1} Block {b+1}] HxW={H}x{W} cin={cin} cout={cout} -> k*={k_best}; detail={detail}")
                layers.append(ConvBNReLU(cin, cout, k_best, s))
                cin = cout
                if s > 1:
                    H = (H + s - 1) // s
                    W = (W + s - 1) // s
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(cin, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class FixedKernelCNN(nn.Module):
    """
    Baseline: same channels/blocks/strides as BKSEF model, but use a fixed kernel (e.g., 3x3) everywhere.
    """
    def __init__(self,
                 in_ch: int,
                 num_classes: int,
                 input_hw: Tuple[int,int],
                 stage_channels: List[int],
                 stage_blocks: List[int],
                 stage_strides: List[int],
                 k_fixed: int = 3):
        super().__init__()
        assert len(stage_channels) == len(stage_blocks) == len(stage_strides)
        H, W = input_hw
        cin = in_ch
        layers = []
        for s_idx, (cout, B, stride) in enumerate(zip(stage_channels, stage_blocks, stage_strides)):
            for b in range(B):
                s = stride if b == 0 else 1
                layers.append(ConvBNReLU(cin, cout, k_fixed, s))
                cin = cout
                if s > 1:
                    H = (H + s - 1) // s
                    W = (W + s - 1) // s
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(cin, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# -----------------------------
# Data & Train/Eval
# -----------------------------
def loaders_gtsrb(batch=128, workers=0, pin=False):
    # standard 48x48 resize for edge experiments
    t_train = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ])
    t_test = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ])
    tr = torchvision.datasets.GTSRB("./data", split="train", transform=t_train, download=True)
    te = torchvision.datasets.GTSRB("./data", split="test",  transform=t_test,  download=True)
    return DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=pin), \
           DataLoader(te, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin)

def train_epoch(model, loader, opt, device):
    model.train()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * x.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot += x.size(0)
    return tot_loss / tot, tot_correct / tot

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        tot_loss += loss.item() * x.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot += x.size(0)
    return tot_loss / tot, tot_correct / tot

def train_and_eval(model, tr, te, device, epochs=5, lr=0.05, label="MODEL"):
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    best = 0.0
    for e in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, tr, opt, device)
        te_loss, te_acc = evaluate(model, te, device)
        best = max(best, te_acc)
        print(f"[{label}] Epoch {e:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | test {te_loss:.4f}/{te_acc:.4f} (best {best:.4f})")
    return best

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pin_memory", type=int, default=0)

    # BKSEF parameters (edge-leaning default puts more weight on cost)
    ap.add_argument("--lam", type=float, nargs=3, default=[0.35, 0.40, 0.25], help="λ1 λ2 λ3")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--beta", type=float, default=0.75)

    # Baseline kernel
    ap.add_argument("--baseline_k", type=int, default=3)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    tr, te = loaders_gtsrb(batch=args.batch_size, workers=args.workers, pin=bool(args.pin_memory))

    # ----------------- BKSEF model -----------------
    bksef_model = BKSEF_CNN(
        in_ch=3, num_classes=43, input_hw=(48,48),
        stage_channels=[32,64,128],
        stage_blocks=[2,2,2],
        stage_strides=[1,2,2],
        k_candidates=[[3,5,7],[3,5],[1,3,5]],  # allow k=1 in late stage to reduce MACs
        lam=tuple(args.lam),
        alpha=args.alpha,
        beta=args.beta,
        print_decisions=True
    ).to(device)

    print("\n=== Training BKSEF-selected model (Edge) ===")
    bksef_best = train_and_eval(bksef_model, tr, te, device, epochs=args.epochs, lr=args.lr, label="BKSEF")

    bksef_lat = latency_ms(bksef_model, (1,3,48,48), device, warmup=50, iters=200)
    print(f"[BKSEF] Inference latency: {bksef_lat:.3f} ms on {device}")

    # ----------------- Baseline model -----------------
    base_model = FixedKernelCNN(
        in_ch=3, num_classes=43, input_hw=(48,48),
        stage_channels=[32,64,128],
        stage_blocks=[2,2,2],
        stage_strides=[1,2,2],
        k_fixed=args.baseline_k
    ).to(device)

    print("\n=== Training BASELINE (fixed k) model ===")
    base_best = train_and_eval(base_model, tr, te, device, epochs=args.epochs, lr=args.lr, label=f"BASE-k={args.baseline_k}")

    base_lat = latency_ms(base_model, (1,3,48,48), device, warmup=50, iters=200)
    print(f"[BASELINE k={args.baseline_k}] Inference latency: {base_lat:.3f} ms on {device}")

    # ----------------- Summary -----------------
    print("\n=== Summary (GTSRB, Edge) ===")
    rows = [
        [f"BKSEF (λ={args.lam}, α={args.alpha}, β={args.beta})", f"{bksef_best:.4f}", f"{bksef_lat:.3f} ms"],
        [f"Baseline (k={args.baseline_k})",                    f"{base_best:.4f}",  f"{base_lat:.3f} ms"],
    ]
    print_table(rows, headers=["Model", "Best Test Acc", "Latency (1x3x48x48)"])

if __name__ == "__main__":
    main()
