"""Steps 2-3 for the photos in ../my_inputs_resized. Run from ~/multinex_repro/Multinex
with CUDA_VISIBLE_DEVICES=<gpu> and PHYS_GPU=<same index> (for nvidia-smi).
NIQE/BRISQUE: test_noref.py's own functions (input) and the per_image.csv of the
test_noref.py runs (outputs). Main weights: LOL-v1; the dagger NoRef run is reported too.
Timing: LOL-v1 weights, input padded to 32 like test_noref.py, fp32, batch 1,
10 warm-up + 50 timed, torch.cuda.synchronize + perf_counter.
Writes ../my_photo/{my_photo_results.json, my_photo_input.png, my_photo_side_by_side.png, my_photo_crop.png}."""
import os, sys, csv, json, time, subprocess, shutil
sys.path.insert(0, os.path.abspath('Enhancement')); sys.path.insert(0, os.path.abspath('.'))
import numpy as np, torch, cv2
import test_noref as T
from basicsr.utils.options import parse

IN = '../my_inputs_resized'; OUT = '../my_photo'; os.makedirs(OUT, exist_ok=True)
GPU = os.environ['PHYS_GPU']; dev = torch.device('cuda')
R = 'results/my_photos'
runs = {'multinex': f'{R}/Multinex_LOL-v1', 'nano': f'{R}/Multinex-Nano_LOL-v1',
        'multinex_dagger': f'{R}/Multinex_NoRef', 'nano_dagger': f'{R}/Multinex-Nano_NoRef'}
W = {'multinex': ('Options/Multinex_LOL-v1.yaml', 'pretrained_weights/Multinex_LOL-v1.pth'),
     'nano': ('Options/Multinex-Nano_LOL-v1.yaml', 'pretrained_weights/Multinex-Nano_LOL-v1.pth')}
csvs = {k: {r['image']: r for r in csv.DictReader(open(f'{v}/my_inputs_resized/per_image.csv'))} for k, v in runs.items()}
outp = lambda k, f: f'{runs[k]}/my_inputs_resized/images/{f}'
meta = {m['file']: m for m in json.load(open('../logs/10_my_photos_meta.json'))}
files = sorted(meta)

def util():
    return int(subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).decode().strip())

nets = {k: T.make_model(dict(parse(o, is_train=False), dist=False), w, dev, dp_world=False).eval() for k, (o, w) in W.items()}
res = []
for f in files:
    rgb = T.imread_rgb(f'{IN}/{f}')
    x = T.to_tensor(rgb.astype(np.float32) / 255.0).to(dev)
    xp, _ = T.pad_to_factor(x, 32)
    ms, util_before, util_during = {}, util(), []
    for k, net in nets.items():
        times = []
        mon = subprocess.Popen(['nvidia-smi', '-i', GPU, '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-lms', '200'], stdout=subprocess.PIPE)
        with torch.inference_mode():
            for i in range(60):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                net(xp)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                if i >= 10: times.append((t1 - t0) * 1000)
        mon.terminate(); util_during += [int(v) for v in mon.communicate()[0].decode().split() if v.isdigit()]
        assert len(times) == 50
        ms[k] = float(np.mean(times))
    bright = lambda p: float(cv2.imread(p, cv2.IMREAD_COLOR).mean())
    m = meta[f]
    res.append(dict(
        file=f, resolution=m['resolution'], exif_date=m['exif_date'], exif_device=m['exif_device'],
        niqe=dict(input=T.compute_niqe_custom(rgb), multinex=float(csvs['multinex'][f]['niqe']), nano=float(csvs['nano'][f]['niqe'])),
        brisque=dict(input=T.compute_brisque_piq(x), multinex=float(csvs['multinex'][f]['brisque']), nano=float(csvs['nano'][f]['brisque'])),
        mean_brightness=dict(input=bright(f'{IN}/{f}'), multinex=bright(outp('multinex', f)), nano=bright(outp('nano', f))),
        ms=ms, gpu_util_pct=util_before,
        weights=dict(multinex=W['multinex'][1], nano=W['nano'][1]),
        gpu_util_note=f'gpu_util_pct = nvidia-smi on physical GPU {GPU} just before timing this photo (other users\' load); '
                      f'during timing (includes this run): min {min(util_during)}, max {max(util_during)}, n={len(util_during)} samples',
        dagger_noref=dict(weights='pretrained_weights/Multinex_dagger_NoRef.pth, pretrained_weights/Multinex-Nano_dagger_NoRef.pth',
                          niqe=dict(multinex=float(csvs['multinex_dagger'][f]['niqe']), nano=float(csvs['nano_dagger'][f]['niqe'])),
                          brisque=dict(multinex=float(csvs['multinex_dagger'][f]['brisque']), nano=float(csvs['nano_dagger'][f]['brisque'])),
                          mean_brightness=dict(multinex=bright(outp('multinex_dagger', f)), nano=bright(outp('nano_dagger', f)))),
        source=dict(page=m['source_page'], author=m['author'], license=m['license'], original=m['original'],
                    original_resolution=m['original_resolution'])))
    print(json.dumps(res[-1]))

# figures: photo with the lowest Multinex (LOL-v1) NIQE
best = min(res, key=lambda r: r['niqe']['multinex']); bf = best['file']
paths = [f'{IN}/{bf}', outp('multinex', bf), outp('nano', bf)]
def row(ims, out):  # same layout as build_deliver_v2.row: equal height, 10 px white gaps
    h = min(i.shape[0] for i in ims)
    ims = [i if i.shape[0] == h else cv2.resize(i, (round(i.shape[1] * h / i.shape[0]), h), interpolation=cv2.INTER_AREA) for i in ims]
    gap = np.full((h, 10, 3), 255, np.uint8); parts = []
    for i in ims: parts += [i, gap]
    cv2.imwrite(out, np.hstack(parts[:-1]))
ims = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
shutil.copy(paths[0], f'{OUT}/my_photo_input.png')
row(ims, f'{OUT}/my_photo_side_by_side.png')
# crop: 120x120 window (stride 20) on the input; keep windows whose Laplacian std is in the top
# quarter (textured), then take the darkest of them
g = cv2.cvtColor(ims[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
lap = cv2.Laplacian(g, cv2.CV_32F)
S = 120; H, Wd = g.shape; cand = []
for y in range(0, H - S + 1, 20):
    for x0 in range(0, Wd - S + 1, 20):
        cand.append((float(g[y:y+S, x0:x0+S].mean()), float(lap[y:y+S, x0:x0+S].std()), x0, y))
thr = np.percentile([c[1] for c in cand], 75)
mean_b, tex, cx, cy = min(c for c in cand if c[1] >= thr)
crops = [cv2.resize(i[cy:cy+S, cx:cx+S], (3 * S, 3 * S), interpolation=cv2.INTER_NEAREST) for i in ims]
row(crops, f'{OUT}/my_photo_crop.png')
for r in res:
    r['figure'] = r['file'] == bf
best['crop_box'] = dict(x=cx, y=cy, w=S, h=S, zoom=3, interpolation='nearest', coords='pixels of the 1200 px resized input',
                        rule='120x120 windows, stride 20; Laplacian std in top 25% of windows; lowest mean grey among them',
                        window_mean_grey=mean_b, window_laplacian_std=tex, laplacian_std_threshold=float(thr))
json.dump(res, open(f'{OUT}/my_photo_results.json', 'w'), indent=2, ensure_ascii=False)
print('best', bf, 'crop', cx, cy)
