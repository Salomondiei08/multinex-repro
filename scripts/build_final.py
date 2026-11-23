"""Step 4: add the photo results to ~/multinex_repro/deliver, update results.json
(only 'mine' and 'failures'), write README_submission.md, zip. Run from ~/multinex_repro."""
import os, re, json, shutil, glob, zipfile
D = 'deliver'
for f in ['my_photo_input.png', 'my_photo_side_by_side.png', 'my_photo_crop.png', 'my_photo_results.json', 'my_photo_notes.md']:
    shutil.copy(f'my_photo/{f}', D)
for f in glob.glob('logs/10_*'):
    shutil.copy(f, f'{D}/logs/')
for f in ['prep_my_photos.py', 'measure_my_photos.py', 'build_final.py']:
    shutil.copy(f'scripts/{f}', f'{D}/scripts/')
for f in ['lol_input.png', 'lol_side_by_side.png', 'lol_worst_side_by_side.png', 'mine_input.png', 'mine_side_by_side.png', 'per_image_lolv1.csv']:
    assert os.path.getsize(f'{D}/{f}') > 0, f
assert os.listdir(f'{D}/logs')

R = json.load(open(f'{D}/results.json'))
mp = json.load(open('my_photo/my_photo_results.json'))
R['mine'] = mp
R['failures'] = [x for x in R['failures'] if not x.startswith('Step 7 own photo: NOT RUN')]
json.dump(R, open(f'{D}/results.json', 'w'), indent=2, ensure_ascii=False)
shutil.copy(f'{D}/results.json', 'results.json')

L, n = R['lolv1'], len(mp)
OS = re.search(r'PRETTY_NAME="([^"]+)"', open('logs/01_env.txt').read()).group(1)  # results.json has no os key
mean = lambda k, m: sum(r[k][m] for r in mp) / n
nq = {m: mean('niqe', m) for m in ['input', 'multinex', 'nano']}
bq = {m: mean('brisque', m) for m in ['input', 'multinex', 'nano']}
ms = {m: mean('ms', m) for m in ['multinex', 'nano']}
cmds = '\n'.join(R['commands'] + open('logs/10_commands.txt').read().split('\n'))
ls = sorted(os.listdir(D))
readme = f"""# Multinex reproduction / Multinex 재현

Repo: https://github.com/albrateanu/Multinex, commit `{R['commit'][:7]}` ({R['commit']}).

## Environment / 환경
GPU {R['gpu']} x{R['gpu_count']} (shared with other jobs / 다른 작업과 공유), driver {R['driver']},
CUDA {R['cuda_torch']} (torch build; driver supports {R['cuda_driver']}), cuDNN {R['cudnn']},
PyTorch {R['torch']}, Python {R['python']}, {OS}. Install: {R['install_method']} + README packages + piq.

## Main results / 주요 결과
LOL-v1 test set ({R['data']['lolv1_test_pairs']} pairs / 쌍), README test.py:

| Model | PSNR | SSIM | LPIPS | Paper (PSNR/SSIM/LPIPS) | Params | ms 600x400 | ms 1080p |
|---|---|---|---|---|---|---|---|
| Multinex | {L['multinex']['psnr']} | {L['multinex']['ssim']} | {L['multinex']['lpips']} | 23.19 / 0.843 / 0.129 | {L['multinex']['params']} | {L['multinex']['ms_600x400']} | {L['multinex']['ms_1080p']} |
| Multinex-Nano | {L['nano']['psnr']} | {L['nano']['ssim']} | {L['nano']['lpips']} | 19.42 / 0.742 / 0.276 | {L['nano']['params']} | {L['nano']['ms_600x400']} | {L['nano']['ms_1080p']} |

Photos ({n} phone night photos from Wikimedia Commons, long side 1200 px, LOL-v1 weights), mean over photos:
사진 ({n}장, Wikimedia Commons의 휴대폰 야간 사진, 긴 변 1200 px, LOL-v1 가중치), 평균:
NIQE input {nq['input']:.4f}, Multinex {nq['multinex']:.4f}, Nano {nq['nano']:.4f};
BRISQUE input {bq['input']:.4f}, Multinex {bq['multinex']:.4f}, Nano {bq['nano']:.4f};
ms Multinex {ms['multinex']:.2f}, Nano {ms['nano']:.3f} (lower is better / 낮을수록 좋음).
Benchmark no-reference results (DICM/LIME/MEF/NPE) are in results.json `other_runs`.

Notes: timings are upper bounds on shared GPUs (photo timing ran on an idle GPU, see gpu_util_pct).
The photos are not our own: they are free-licensed photos (CC0 / CC BY / CC BY-SA), with author and
license listed per photo in my_photo_results.json `source` and logs/10_my_photos.txt.
참고: 공유 GPU이므로 시간은 상한값입니다. 사진은 직접 찍은 것이 아니라 자유 라이선스 사진이며,
작가와 라이선스는 my_photo_results.json의 `source`와 logs/10_my_photos.txt에 있습니다.

## Commands / 실행 명령
```
{cmds.strip()}
```

## Contents / 폴더 내용
""" + '\n'.join(f'{i+1}. `{f}`' for i, f in enumerate(ls + ['README_submission.md'] if 'README_submission.md' not in ls else ls)) + """

`logs/` holds the original logs, `logs/raw/` all step logs, `scripts/` our helper scripts.
`logs/`는 원본 로그, `logs/raw/`는 모든 단계 로그, `scripts/`는 보조 스크립트입니다.
"""
open(f'{D}/README_submission.md', 'w').write(readme)

def zipit(skip_raw):
    with zipfile.ZipFile('multinex_results_final.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(D):
            if skip_raw and root.startswith(f'{D}/logs/raw'): continue
            for f in fs: z.write(os.path.join(root, f))
zipit(False)
if os.path.getsize('multinex_results_final.zip') > 20 * 2**20:
    zipit(True); print('over 20 MB: rebuilt without logs/raw')
print('zip bytes', os.path.getsize('multinex_results_final.zip'), f"= {os.path.getsize('multinex_results_final.zip')/2**20:.2f} MB")
