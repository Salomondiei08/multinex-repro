"""Build results.json, side-by-side figures, deliver/ and the zip.
Every number is parsed from logs/CSVs produced by earlier steps (no hand-edited values).
Run from ~/multinex_repro with the multinex env."""
import os, re, csv, json, glob, shutil, subprocess
import numpy as np, cv2

R = os.path.expanduser('~/multinex_repro'); L = f'{R}/logs'; M = f'{R}/Multinex'; D = f'{R}/deliver'
rd = lambda p: open(p).read()
grab = lambda pat, s, cast=str: cast(re.search(pat, s, re.M).group(1).strip())

env = rd(f'{L}/01_env.txt')
info = dict(gpu=grab(r'^gpu0_name (.+)$', env), driver=grab(r'== driver\n(.+)', env),
            cuda=grab(r'^torch.version.cuda (.+)$', env), torch=grab(r'^torch (.+)$', env),
            cudnn=grab(r'^cudnn (.+)$', env), python=grab(r'^Python (\S+)', env),
            nvcc=grab(r'release ([\d.]+)', env), os=grab(r'PRETTY_NAME="(.+)"', env),
            commit=grab(r'^commit (\w+)', rd(f'{L}/02_commit.txt')))
cmds = [l.strip() for l in rd(f'{L}/commands.txt').splitlines() if l.startswith('[')]
fixes = re.findall(r'^## \d+\. (.+)$', rd(f'{L}/fixes.md'), re.M)

def per_image(name):
    return list(csv.DictReader(open(f'{L}/06_per_image_{name}.csv')))

def model_block(name, testlog):
    t = rd(f'{L}/{testlog}.log'); c = rd(f'{L}/06_complexity_{name}.log')
    row = lambda res: re.search(rf'^\s*{res} \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+)', c, re.M).groups()
    r600, r1080, r256 = row('400x600'), row('1080x1920'), row('256x256')
    return dict(psnr=grab(r'RGB  - PSNR: ([\d.]+)', t, float), ssim=grab(r'RGB  - SSIM: ([\d.]+)', t, float),
                lpips=grab(r'LPIPS \(alex\): ([\d.]+)', t, float),
                params=grab(r'Params\s+: ([\d,]+)', c, lambda s: int(s.replace(',', ''))),
                ms_600x400=float(r600[1]), ms_1080p=float(r1080[1]),
                ms_600x400_median=float(r600[2]), ms_1080p_median=float(r1080[2]),
                ms_600x400_std=float(r600[5]), ms_1080p_std=float(r1080[5]),
                gflops_256x256=float(r256[0]), gflops_600x400=float(r600[0]), gflops_1080p=float(r1080[0]))

mx, nn_ = model_block('Multinex', '05a_multinex_lolv1_attempt1'), model_block('Multinex-Nano', '05b_nano_lolv1_attempt1')
pm, pn = per_image('Multinex'), per_image('Multinex-Nano')
best = max(pm, key=lambda r: float(r['psnr'])); worst = min(pm, key=lambda r: float(r['psnr']))
nano_by = {r['image']: r for r in pn}

# combined per-image CSV
os.makedirs(D, exist_ok=True)
with open(f'{D}/lolv1_per_image.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['image', 'multinex_psnr', 'multinex_ssim', 'multinex_lpips', 'nano_psnr', 'nano_ssim', 'nano_lpips'])
    for r in sorted(pm, key=lambda r: -float(r['psnr'])):
        n = nano_by[r['image']]; w.writerow([r['image'], r['psnr'], r['ssim'], r['lpips'], n['psnr'], n['ssim'], n['lpips']])

def label(img, text):
    bar = np.full((36, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])

def side_by_side(paths_labels, out, max_h=600):
    tiles = []
    for p, lab in paths_labels:
        im = cv2.imread(p) if p else None
        if im is None:
            im = np.zeros((400, 600, 3), np.uint8); lab += ' (missing)'
        if im.shape[0] > max_h:
            im = cv2.resize(im, (round(im.shape[1] * max_h / im.shape[0]), max_h), interpolation=cv2.INTER_AREA)
        tiles.append(label(im, lab))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    cv2.imwrite(out, np.hstack(sum([[t, np.full((h, 6, 3), 255, np.uint8)] for t in tiles], [])[:-1]))

fig = f'{D}/figures'; os.makedirs(fig, exist_ok=True)
for tag, r in [('best', best), ('worst', worst)]:
    im, n = r['image'], nano_by[r['image']]
    side_by_side([(f'{M}/data/LOLv1/Test/input/{im}', 'Input'),
                  (f'{M}/results/LOL-v1/Multinex_LOL-v1/Multinex_LOL-v1/{im}', f"Multinex {float(r['psnr']):.2f} dB"),
                  (f'{M}/results/LOL-v1/Multinex-Nano_LOL-v1/Multinex-Nano_LOL-v1/{im}', f"Nano {float(n['psnr']):.2f} dB"),
                  (f'{M}/data/LOLv1/Test/target/{im}', 'Ground truth')], f'{fig}/lolv1_{tag}_{os.path.splitext(im)[0]}.png')

# no-reference benchmarks (DICM, LIME, MEF, NPE) replace own photos at user request
def summ(log):
    t = rd(f'{L}/{log}.log')
    return {m[0]: dict(niqe=float(m[1]), brisque=float(m[2])) for m in
            re.findall(r'=== Dataset: (\w+) ===.*?NIQE:\s+([\d.]+)\nBRISQUE:\s+([\d.]+)', t, re.S)}
noref = dict(multinex=summ('07b_noref_multinex'), nano=summ('07c_noref_nano'))
for k in noref:
    v = list(noref[k].values())
    assert len(v) == 4, noref[k]
    noref[k]['mean_of_4'] = dict(niqe=float(np.mean([x['niqe'] for x in v])), brisque=float(np.mean([x['brisque'] for x in v])))
per = lambda rdir, ds: list(csv.DictReader(open(f'{M}/results/{rdir}/{ds}/per_image.csv')))
noref_examples = {}
for ds in ['DICM', 'LIME', 'MEF', 'NPE']:
    pmx = per('Multinex_NoRef', ds); pnn = {r['image']: r for r in per('Multinex-Nano_NoRef', ds)}
    ex = min(pmx, key=lambda r: float(r['niqe']))  # best Multinex NIQE in this dataset
    img = ex['image']; rel = os.path.splitext(img)[0] + '.png'
    cands = [q for q in glob.glob(f'{M}/data/{ds}/**/*', recursive=True)
             if os.path.relpath(q, f'{M}/data/{ds}') == img or os.path.basename(q) == os.path.basename(img)]
    noref_examples[ds] = dict(file=img, multinex_niqe=float(ex['niqe']), multinex_brisque=float(ex['brisque']),
                              nano_niqe=float(pnn[img]['niqe']), nano_brisque=float(pnn[img]['brisque']))
    side_by_side([(cands[0] if cands else None, f'{ds} input'),
                  (f'{M}/results/Multinex_NoRef/{ds}/images/{rel}', f"Multinex NIQE {float(ex['niqe']):.2f}"),
                  (f'{M}/results/Multinex-Nano_NoRef/{ds}/images/{rel}', f"Nano NIQE {float(pnn[img]['niqe']):.2f}"),
                 ], f'{fig}/noref_{ds}_{os.path.splitext(os.path.basename(img))[0]}.png')
    for rdir in ['Multinex_NoRef', 'Multinex-Nano_NoRef']:
        shutil.copy(f'{M}/results/{rdir}/{ds}/per_image.csv', f'{D}/noref_per_image_{rdir}_{ds}.csv')

res = dict(**info, commands=cmds, fixes=fixes,
           lolv1=dict(multinex=mx, nano=nn_),
           paper_table1=dict(multinex=dict(psnr=23.19, ssim=0.843, lpips=0.129, params_M=0.0446, gflops=2.50),
                             nano=dict(psnr=19.42, ssim=0.742, lpips=0.276, params_K=0.69)),
           best_image=dict(image=best['image'], ranked_by='Multinex PSNR', multinex_psnr=float(best['psnr']), nano_psnr=float(nano_by[best['image']]['psnr'])),
           worst_image=dict(image=worst['image'], ranked_by='Multinex PSNR', multinex_psnr=float(worst['psnr']), nano_psnr=float(nano_by[worst['image']]['psnr'])),
           mine=None, noref=noref, noref_examples=noref_examples,
           paper_noref_table_note='Repo assets/NoRef_table.png shows pre-dagger numbers; README says the dagger checkpoints match the updated arXiv table, which is not in the repo. Not directly comparable.',
           notes=['Timing: repo basicsr/complexity.py, warmup=10, runs=50, CUDA events + torch.cuda.synchronize, random input, batch 1, fp32, cudnn.benchmark=True.',
                  'GPU was shared with another user (90-100% util on all 8 GPUs); latencies are upper bounds. See logs/06_complexity_*.log.',
                  'Resolutions in complexity logs are HxW: 400x600 = 600x400 image, 1080x1920 = 1080p.',
                  'Own photos not provided; replaced at user request by the no-reference benchmarks DICM (69), LIME (10), MEF (17), NPE (8) from the BIMEF Google Drive folder, run with the unchanged README no-reference commands (*_dagger_NoRef.pth weights).'])
json.dump(res, open(f'{R}/results.json', 'w'), indent=2)
shutil.copy(f'{R}/results.json', D)
if os.path.exists(f'{D}/logs'): shutil.rmtree(f'{D}/logs')
shutil.copytree(L, f'{D}/logs')
shutil.copytree(f'{R}/scripts', f'{D}/scripts', dirs_exist_ok=True)
if os.path.exists(f'{R}/multinex_results.zip'): os.remove(f'{R}/multinex_results.zip')
subprocess.run(['zip', '-qr', f'{R}/multinex_results.zip', 'deliver'], cwd=R, check=True)
print(json.dumps({k: res[k] for k in ['best_image', 'worst_image', 'noref', 'noref_examples']}, indent=1))
