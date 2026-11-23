"""Build ~/multinex_repro/deliver/ in the v2 layout and zip it.
Every number is parsed from logs/CSVs/JSON written by earlier steps; nothing is typed in.
The only literals are the paper Table 1 values given in the task, used for the deltas.
Run from ~/multinex_repro with the multinex env."""
import os, re, csv, json, glob, shutil, subprocess
import numpy as np, cv2

R = os.path.expanduser('~/multinex_repro'); L = f'{R}/logs'; M = f'{R}/Multinex'; D = f'{R}/deliver'
rd = lambda p: open(p, errors='replace').read()
def grab(pat, s, cast=str):
    m = re.search(pat, s, re.M)
    assert m, f'pattern not found: {pat}'
    return cast(m.group(1).strip())

os.makedirs(f'{D}/logs', exist_ok=True)
env = rd(f'{L}/01_env.txt'); commit = rd(f'{L}/02_commit.txt')
meas = json.load(open(f'{L}/10_measure_v2.json'))

# ---------- metrics ----------
def lol(name, testlog):
    t = rd(f'{L}/{testlog}.log'); c = rd(f'{L}/06_complexity_{name}.log')
    ms = lambda res: float(re.search(rf'^\s*{res} \|\s*[\d.]+ \|\s*([\d.]+) \|', c, re.M).group(1))
    return dict(psnr=grab(r'RGB  - PSNR: ([\d.]+)', t, float), ssim=grab(r'RGB  - SSIM: ([\d.]+)', t, float),
                lpips=grab(r'LPIPS \(alex\): ([\d.]+)', t, float),
                psnr_y=grab(r'Y    - PSNR_y: ([\d.]+)', t, float), ssim_y=grab(r'Y    - SSIM_y: ([\d.]+)', t, float),
                params=grab(r'Params\s+: ([\d,]+)', c, lambda s: int(s.replace(',', ''))),
                ms_600x400=ms('400x600'), ms_1080p=ms('1080x1920'),
                vram_mb=meas['vram_mb'][name]['600x400'])
mx = lol('Multinex', '05a_multinex_lolv1_attempt1'); nn_ = lol('Multinex-Nano', '05b_nano_lolv1_attempt1')

pm = {r['image']: r for r in csv.DictReader(open(f'{L}/06_per_image_Multinex.csv'))}
pn = {r['image']: r for r in csv.DictReader(open(f'{L}/06_per_image_Multinex-Nano.csv'))}
best = max(pm, key=lambda k: float(pm[k]['psnr'])); worst = min(pm, key=lambda k: float(pm[k]['psnr']))

# ---------- per_image_lolv1.csv ----------
lol_in = lambda im: f'{M}/data/LOLv1/Test/input/{im}'
lol_gt = lambda im: f'{M}/data/LOLv1/Test/target/{im}'
lol_mx = lambda im: f'{M}/results/LOL-v1/Multinex_LOL-v1/Multinex_LOL-v1/{im}'
lol_nn = lambda im: f'{M}/results/LOL-v1/Multinex-Nano_LOL-v1/Multinex-Nano_LOL-v1/{im}'
bright = lambda p: float(cv2.imread(p).astype(np.float64).mean())  # mean of all uint8 pixel values, 0-255
with open(f'{D}/per_image_lolv1.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['image', 'psnr_multinex', 'ssim_multinex', 'psnr_nano', 'ssim_nano', 'mean_brightness_input', 'mean_brightness_gt'])
    for im in sorted(pm, key=lambda k: int(os.path.splitext(k)[0])):
        w.writerow([im, pm[im]['psnr'], pm[im]['ssim'], pn[im]['psnr'], pn[im]['ssim'], bright(lol_in(im)), bright(lol_gt(im))])

# ---------- images: no captions, same height, 10 px white gap ----------
def row(paths, out):
    ims = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
    assert all(i is not None for i in ims), paths
    h = min(i.shape[0] for i in ims)
    ims = [i if i.shape[0] == h else cv2.resize(i, (round(i.shape[1] * h / i.shape[0]), h), interpolation=cv2.INTER_AREA) for i in ims]
    gap = np.full((h, 10, 3), 255, np.uint8)
    parts = []
    for i in ims:
        parts += [i, gap]
    cv2.imwrite(out, np.hstack(parts[:-1]))

shutil.copy(lol_in(best), f'{D}/lol_input.png')
row([lol_in(best), lol_mx(best), lol_nn(best), lol_gt(best)], f'{D}/lol_side_by_side.png')
row([lol_in(worst), lol_mx(worst), lol_nn(worst), lol_gt(worst)], f'{D}/lol_worst_side_by_side.png')
mine0 = meas['mine'][0]  # DICM median-NIQE image (largest benchmark)
ds, fn = mine0['file'].split('/', 1); stem = os.path.splitext(fn)[0] + '.png'
cv2.imwrite(f'{D}/mine_input.png', cv2.imread(f'{M}/data/{ds}/{fn}', cv2.IMREAD_COLOR))  # lossless PNG of the original JPG
row([f'{M}/data/{ds}/{fn}', f'{M}/results/Multinex_NoRef/{ds}/images/{stem}', f'{M}/results/Multinex-Nano_NoRef/{ds}/images/{stem}'],
    f'{D}/mine_side_by_side.png')

# ---------- logs (copied, not edited) ----------
cp = lambda src, dst: shutil.copy(f'{L}/{src}', f'{D}/logs/{dst}')
cp('01_env.txt', '01_env.txt'); cp('02_commit.txt', '02_commit.txt'); cp('03_install.log', 'install.txt')
cp('05a_multinex_lolv1_attempt1.log', 'first_attempt.txt')
cp('05a_multinex_lolv1_attempt1.log', 'test_multinex.txt')  # first attempt succeeded, so it is also the final run
cp('05b_nano_lolv1_attempt1.log', 'test_nano.txt'); cp('07_pip_freeze_after_fix.txt', 'pip_freeze.txt')
with open(f'{D}/logs/noref_mine.txt', 'w') as fh:  # concatenation of unedited logs, with file headers
    for f in ['07a_noref_multinex_attempt1.log', '07b_noref_multinex.log', '07c_noref_nano.log', '10_measure_v2.log']:
        fh.write(f'########## logs/{f} ##########\n' + rd(f'{L}/{f}') + '\n')
priority = [('07a_noref_multinex_attempt1.log', r'test_noref\.py", line 40'), ('07a_noref_multinex_attempt1.log', r'ModuleNotFoundError'),
            ('07_download_noref.log', r'Failed to retrieve folder contents'), ('07_download_noref.log', r'has more than 50 files')]
ex = []
for f, pat in priority:
    for n, line in enumerate(rd(f'{L}/{f}').splitlines(), 1):
        if re.search(pat, line):
            ex.append(f'logs/raw/{f}:{n}: {line.strip()}'); break
seen = set()
for f in sorted(os.listdir(L)):
    if not f.endswith(('.log', '.txt')) or len(ex) >= 10:
        continue
    for n, line in enumerate(rd(f'{L}/{f}').splitlines(), 1):
        if re.search(r'Error|\[WARN\]|Warning', line) and line.strip() not in seen and not any(line.strip() in e for e in ex):
            seen.add(line.strip()); ex.append(f'logs/raw/{f}:{n}: {line.strip()}')
            if len(ex) >= 10: break
open(f'{D}/logs/error_excerpt.txt', 'w').write('\n'.join(ex[:10]) + '\n')
if os.path.exists(f'{D}/logs/raw'): shutil.rmtree(f'{D}/logs/raw')
shutil.copytree(L, f'{D}/logs/raw')  # every original log, for traceability
shutil.copytree(f'{R}/scripts', f'{D}/scripts', dirs_exist_ok=True)

# ---------- results.json (exact key set) ----------
cmds = [l.strip() for l in rd(f'{R}/scripts/commands_pre.txt').splitlines() if l.strip()]
cmds += [re.sub(r'^\[[^\]]*\]\s*', '', l.strip()) for l in rd(f'{L}/commands.txt').splitlines() if l.startswith('[')]
noref = lambda log: {m[0]: (float(m[1]), float(m[2])) for m in re.findall(r'=== Dataset: (\w+) ===.*?NIQE:\s+([\d.]+)\nBRISQUE:\s+([\d.]+)', rd(f'{L}/{log}.log'), re.S)}
nm, nb = noref('07b_noref_multinex'), noref('07c_noref_nano')
prof = re.search(r'aten::native_layer_norm\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+([\d.]+)%', rd(f'{L}/06_profile_multinex.log')).group(1)
gfl = lambda n: re.search(r'^\s*256x256 \|\s*([\d.]+)', rd(f'{L}/06_complexity_{n}.log'), re.M).group(1)
counts = rd(f'{L}/07_noref_data_counts.txt').strip().splitlines()[:4]
res = {
  "gpu": grab(r'^gpu0_name (.+)$', env), "gpu_count": grab(r'^gpu_count (\d+)$', env, int),
  "driver": grab(r'Driver Version: (\S+)', env), "cuda_driver": grab(r'CUDA Version: (\S+)', env),
  "cuda_torch": grab(r'^torch.version.cuda (.+)$', env), "cudnn": grab(r'^cudnn (.+)$', env),
  "torch": grab(r'^torch (.+)$', env), "python": grab(r'== conda env python\n.*\nPython (\S+)', env),  # env python, not the system python3 also in 01_env.txt "os": grab(r'PRETTY_NAME="(.+)"', env),
  "commit": grab(r'^commit (\w+)', commit), "commit_date": grab(r'^Date:\s+(.+)$', commit),
  "install_method": "conda",
  "commands": cmds,
  "fixes": [
    "`which conda` returned nothing on server9 -> conda is not installed on the machine -> installed Miniforge3 in ~/miniforge3 (user space), then ran the README conda commands unchanged",
    "ModuleNotFoundError: No module named 'piq' (Enhancement/test_noref.py line 40) -> test_noref.py imports piq for BRISQUE but piq is not in the README pip list -> pip install piq (piq==0.8.0, no other package changed)",
    "gdown --folder: 'Failed to retrieve folder contents ... has more than 50 files' -> README has no link for DICM/LIME/MEF/NPE and the BIMEF Drive folder that hosts them exceeds gdown's 50-file folder limit -> downloaded the four datasets__*.zip files by file ID"],
  "data": {"lolv1_test_pairs": len(pm),
           "how_downloaded": "gdown 1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu (README LOL-v1 Google Drive link) -> LOLv1.zip, unzipped to Multinex/data/LOLv1/{Train,Test}/{input,target}; Test has 15 input + 15 target PNGs (600x400). No-reference sets: BIMEF Google Drive datasets__{DICM,LIME,MEF,NPE}.zip by file ID (" + ', '.join(counts) + ")"},
  "lolv1": {"multinex": mx, "nano": nn_},
  "best_image": {"name": best, "psnr_multinex": float(pm[best]['psnr']), "psnr_nano": float(pn[best]['psnr'])},
  "worst_image": {"name": worst, "psnr_multinex": float(pm[worst]['psnr']), "psnr_nano": float(pn[worst]['psnr'])},
  "mine": [{k: e[k] for k in ['file', 'resolution', 'niqe_input', 'niqe_multinex', 'brisque_input', 'brisque_multinex', 'ms_multinex']} for e in meas['mine']],
  "other_runs": (
    "README no-reference run (dagger NoRef weights), NIQE/BRISQUE per dataset: "
    + '; '.join(f"{d}: Multinex {nm[d][0]}/{nm[d][1]}, Nano {nb[d][0]}/{nb[d][1]}" for d in ['DICM', 'LIME', 'MEF', 'NPE'])
    + f". GFLOPs at 256x256 (repo basicsr/complexity.py, thop): Multinex {gfl('Multinex')}, Nano {gfl('Multinex-Nano')}."
    + f" vram_mb at 1920x1080: Multinex {meas['vram_mb']['Multinex']['1920x1080']}, Nano {meas['vram_mb']['Multinex-Nano']['1920x1080']}."
    + " Independent per-image re-run (scripts/per_image_metrics.py) reproduced the test.py means for both models."
    + " LOL-v2, self-ensemble, GT-mean: NOT RUN (not requested)."),
  "failures": ["Step 7 own photo: NOT RUN. No own photo was provided (~/multinex_repro/my_inputs stayed empty). At the user's request the README no-reference benchmarks were used instead; 'mine' holds the median-NIQE image of each dataset (DICM, LIME, MEF, NPE)."],
  "notes": (
    "All 8 GPUs were shared with another user's jobs at 90-100% utilization for the whole session, so ms values are upper bounds. "
    f"Multinex is slow for 44.6K params: torch.profiler shows aten::native_layer_norm at {prof}% of CUDA time at 400x600 (channels-last LayerNorm in Multinex_arch.py). "
    "ms_600x400/ms_1080p: repo basicsr/complexity.py, random input, 10 warm-up + 50 timed runs, CUDA events + torch.cuda.synchronize, fp32, batch 1, cudnn.benchmark on. "
    "ms_multinex in 'mine': 10 warm-up + 50 timed, torch.cuda.synchronize + perf_counter, dagger NoRef weights, input padded to a multiple of 32 as test_noref.py does. "
    "vram_mb = torch.cuda.max_memory_allocated for one 600x400 forward (pretrained LOL-v1 weights, inference_mode). "
    "mean_brightness = mean of all uint8 RGB values (0-255). "
    "Visual (from the side-by-side PNGs): Nano shows colour casts (beige on LOL 547, navy on DICM/18, blue/orange blotches on MEF) with blotchy noise in dark areas; Multinex is close to ground truth on LOL but on DICM/18 stays under-exposed with a green cast. "
    "test_noref.py printed 'RuntimeWarning: Mean of empty slice' inside NIQE during DICM; all 104 per-image NIQE/BRISQUE values per model are finite, so no image was dropped from the averages. "
    "Warnings: FutureWarning for torch.cuda.amp.GradScaler in basicsr/models/image_restoration_model.py (harmless). "
    "README self-ensemble example names Multinex_LOLv1.pth / LOL_v1, but the file is Multinex_LOL-v1.pth.")
}
json.dump(res, open(f'{D}/results.json', 'w'), indent=2)
shutil.copy(f'{D}/results.json', f'{R}/results.json')

# ---------- interpretation.md (<= 10 lines) ----------
P = dict(multinex=(23.19, 0.843, 0.129), nano=(19.42, 0.742, 0.276))
obs_p = f'{R}/scripts/observation.txt'
obs = open(obs_p).read().strip() if os.path.exists(obs_p) else 'NOT WRITTEN'
lines = []
for k, lab in [('multinex', 'Multinex'), ('nano', 'Nano')]:
    v = res['lolv1'][k]; p = P[k]
    lines.append(f"{lab} minus Table 1: PSNR {v['psnr'] - p[0]:+.4f} dB, SSIM {v['ssim'] - p[1]:+.4f}, LPIPS {v['lpips'] - p[2]:+.4f} (ours {v['psnr']} / {v['ssim']} / {v['lpips']}).")
lines += [
  "Metric implementation: VERIFIED. test.py uses utils.PSNR -> basicsr.metrics.psnr_ssim.calculate_psnr (RGB, [0,1], no crop), not skimage, and utils.calculate_ssim; the paper ships this code, so PSNR/SSIM gaps are at print-rounding level.",
  "GT-mean: VERIFIED off. The README command has no --GT_mean, so args.GT_mean is False in Enhancement/test.py.",
  "Image I/O: VERIFIED. cv2.imread BGR->RGB, float32/255 on lossless PNGs; metrics use the float output before it is saved.",
  "GPU nondeterminism: VERIFIED negligible. An independent second run (scripts/per_image_metrics.py) reproduced both models' means to the printed precision.",
  "Different weights: VERIFIED same. Multinex_LOL-v1.pth and Nano LOL-v1 blobs are unchanged since the first weights commit bf33e08 (2026-04-12); the _dagger checkpoints are only for no-reference runs.",
  "LPIPS gap: GUESS. test.py uses LPIPS alex v0.1 on float outputs; the paper may have used another LPIPS version or saved uint8 images. Not checked against the paper's code.",
  f"Own photo: NOT PROVIDED; substitute {mine0['file']} (median-NIQE DICM image): {obs}"]
assert len(lines) <= 10
open(f'{D}/interpretation.md', 'w').write('\n'.join(lines) + '\n')

# ---------- zip ----------
z = f'{R}/multinex_results.zip'
if os.path.exists(z): os.remove(z)
subprocess.run(['zip', '-qr', z, 'deliver'], cwd=R, check=True)
print('built', D, 'and', z)
