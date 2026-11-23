"""Per-image PSNR/SSIM/LPIPS on LOL-v1 test set, using the exact pipeline of
Multinex/Enhancement/test.py (same loader, padding, clamp, metric functions).
Self-check: means must equal the numbers printed by test.py (parsed from its log).
Run from ~/multinex_repro/Multinex."""
import os, sys, csv, re, argparse
sys.path.insert(0, os.path.abspath('Enhancement')); sys.path.insert(0, os.path.abspath('.'))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, lpips
from glob import glob
from natsort import natsorted
import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

ap = argparse.ArgumentParser()
ap.add_argument('--opt'); ap.add_argument('--weights'); ap.add_argument('--model_name')
ap.add_argument('--test_log'); ap.add_argument('--out_csv')
a = ap.parse_args()

opt = parse(a.opt, is_train=False); opt['dist'] = False
net = create_model(opt).net_g
ck = torch.load(a.weights)
try:
    net.load_state_dict(ck['params'])
except Exception:
    net.load_state_dict({'module.' + k: v for k, v in ck['params'].items()})
net = nn.DataParallel(net.cuda()).eval()
lp = lpips.LPIPS(net='alex').cuda()
to_lp = lambda im: torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1

ind, tgd = opt['datasets']['val']['dataroot_lq'], opt['datasets']['val']['dataroot_gt']
inp = natsorted(glob(os.path.join(ind, '*.png')) + glob(os.path.join(ind, '*.jpg')))
tgt = natsorted(glob(os.path.join(tgd, '*.png')) + glob(os.path.join(tgd, '*.jpg')))
assert len(inp) == len(tgt) == 15, (len(inp), len(tgt))
rows = []
with torch.inference_mode():
    for ip, tp in zip(inp, tgt):
        assert os.path.basename(ip) == os.path.basename(tp)
        img = np.float32(utils.load_img(ip)) / 255.; tar = np.float32(utils.load_img(tp)) / 255.
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
        _, _, h, w = x.shape; f = 2
        H, W = ((h + f) // f) * f, ((w + f) // f) * f
        x = F.pad(x, (0, W - w if w % f else 0, 0, H - h if h % f else 0), 'reflect')
        r = torch.clamp(net(x)[:, :, :h, :w], 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
        rows.append(dict(image=os.path.basename(ip), model=a.model_name,
                         psnr=float(utils.PSNR(r, tar, False)),
                         ssim=float(utils.calculate_ssim(r, tar, border=0, gtmean=False)),
                         lpips=float(lp(to_lp(r.astype(np.float32)), to_lp(tar.astype(np.float32))).item())))

log = open(a.test_log).read()
ref = {k: float(re.search(p, log).group(1)) for k, p in
       [('psnr', r'RGB  - PSNR: ([\d.]+)'), ('ssim', r'RGB  - SSIM: ([\d.]+)'), ('lpips', r'LPIPS \(alex\): ([\d.]+)')]}
for k in ref:
    m = float(np.mean([r[k] for r in rows]))
    print(f'{a.model_name} mean {k}: {m:.6f}  (test.py log: {ref[k]})')
    assert abs(m - ref[k]) < 2e-3, f'{k} mismatch vs test.py: {m} vs {ref[k]}'
with open(a.out_csv, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print('wrote', a.out_csv)
