"""Extra measurements for deliverable v2. Run from ~/multinex_repro/Multinex.
1. vram_mb: torch.cuda.max_memory_allocated for one forward (batch 1, fp32,
   pretrained LOL-v1 weights) at 600x400 and 1920x1080, per model.
2. "mine" substitute: no own photo was provided, so per benchmark dataset
   (DICM, LIME, MEF, NPE) the image with the MEDIAN Multinex NIQE in the README
   no-reference run is picked (not the best, to avoid cherry-picking). For it:
   NIQE/BRISQUE of the INPUT using test_noref.py's own functions, the Multinex
   scores read from that run's per_image.csv, and Multinex (dagger NoRef weights)
   latency at the image's own resolution: 10 warm-up + 50 timed forwards,
   torch.cuda.synchronize around each, time.perf_counter.
Writes ../logs/10_measure_v2.json."""
import os, sys, json, csv, time
sys.path.insert(0, os.path.abspath('Enhancement')); sys.path.insert(0, os.path.abspath('.'))
import numpy as np, torch
import test_noref as T  # main() is guarded by __name__, safe to import
from basicsr.utils.options import parse

dev = torch.device('cuda')
out = {'vram_mb': {}, 'mine': []}

def load(opt_path, w):
    opt = parse(opt_path, is_train=False); opt['dist'] = False
    return T.make_model(opt, w, dev, dp_world=False).eval()

# 1. VRAM
for name in ['Multinex', 'Multinex-Nano']:
    net = load(f'Options/{name}_LOL-v1.yaml', f'pretrained_weights/{name}_LOL-v1.pth')
    out['vram_mb'][name] = {}
    for tag, (h, w) in [('600x400', (400, 600)), ('1920x1080', (1080, 1920))]:
        x = torch.rand(1, 3, h, w, device=dev)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            net(x)
        torch.cuda.synchronize()
        out['vram_mb'][name][tag] = torch.cuda.max_memory_allocated() / 2**20
        print(name, tag, 'max_memory_allocated MB', out['vram_mb'][name][tag])
    del net; torch.cuda.empty_cache()

# 2. median-NIQE image per dataset
net = load('Options/Multinex_LOL-v2-synthetic.yaml', 'pretrained_weights/Multinex_dagger_NoRef.pth')
for ds in ['DICM', 'LIME', 'MEF', 'NPE']:
    rows = sorted(csv.DictReader(open(f'results/Multinex_NoRef/{ds}/per_image.csv')), key=lambda r: float(r['niqe']))
    r = rows[(len(rows) - 1) // 2]  # lower median
    path = os.path.join('data', ds, r['image'])
    rgb = T.imread_rgb(path)
    h, w = rgb.shape[:2]
    x = T.to_tensor(rgb.astype(np.float32) / 255.0).to(dev)
    niqe_in = T.compute_niqe_custom(rgb)
    brisque_in = T.compute_brisque_piq(x)
    xp, _ = T.pad_to_factor(x, 32)  # same padding as test_noref.py default (--pad_factor 32)
    times = []
    with torch.inference_mode():
        for i in range(60):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            net(xp)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            if i >= 10:
                times.append((t1 - t0) * 1000)
    assert len(times) == 50
    e = dict(file=f'{ds}/{r["image"]}', resolution=f'{w}x{h}', niqe_input=niqe_in,
             niqe_multinex=float(r['niqe']), brisque_input=brisque_in, brisque_multinex=float(r['brisque']),
             ms_multinex=float(np.mean(times)), ms_multinex_median=float(np.median(times)),
             rank_in_dataset=f'{(len(rows) - 1) // 2 + 1} of {len(rows)} by Multinex NIQE')
    out['mine'].append(e); print(e)
json.dump(out, open('../logs/10_measure_v2.json', 'w'), indent=2)
print('wrote ../logs/10_measure_v2.json')
