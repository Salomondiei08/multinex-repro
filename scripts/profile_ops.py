"""Diagnose Multinex latency: top CUDA/CPU ops for one 400x600 forward (random weights; timing only)."""
import sys, os, torch
sys.path.insert(0, os.path.abspath('.'))
from basicsr.utils.options import parse
from basicsr.models import create_model
opt = parse(sys.argv[1], is_train=False); opt['dist'] = False
net = create_model(opt).net_g.cuda().eval()
x = torch.rand(1, 3, 400, 600, device='cuda')
with torch.no_grad():
    for _ in range(10): net(x)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        net(x); torch.cuda.synchronize()
print(p.key_averages().table(sort_by='cuda_time_total', row_limit=12))
print(p.key_averages().table(sort_by='cpu_time_total', row_limit=8))
