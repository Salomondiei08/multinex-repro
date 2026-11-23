# Multinex reproduction (course project)

소스 코드, 실행 로그 및 결과 이미지는 이 GitHub 저장소에서 확인하실 수 있습니다: `scripts/` (소스 코드), `logs/` (실행 로그), `results/` (결과 이미지).

Live demo: https://multinex-demo.vercel.app

Reproduction of Multinex (CVPR 2026) from the official repo
https://github.com/albrateanu/Multinex at commit `e15790a963935569fff1ba6a8b0d33a791765749`.
This repo holds only our scripts, logs and results. It does not contain the model code,
weights or datasets; get those from the official repo and the README links.

## Environment
NVIDIA RTX A6000 (8x, shared), driver 580.178.04, Ubuntu 24.04.4, conda Python 3.9.23,
PyTorch 2.7.1+cu118 (CUDA 11.8, cuDNN 90100). The README packages plus `piq` 0.8.0.

## How to reproduce
1. Install as in `logs/03_install.sh`. Every command we ran is in `logs/commands.txt`.
2. Download LOL-v1 (`logs/04_download.log`) and the DICM/LIME/MEF/NPE sets (`logs/fixes.md` section 6).
3. Run the scripts in `scripts/` from inside the cloned `Multinex/` folder:
   `per_image_metrics.py` (per-image PSNR/SSIM), `profile_ops.py` (op profile),
   `measure_v2.py` (VRAM, no-reference picks, timing), `build_deliver_v2.py` (builds the deliverable).

## Main results (LOL-v1, 15 test pairs)
| Model | PSNR | SSIM | LPIPS | Paper PSNR/SSIM/LPIPS | Params | ms 600x400 |
|---|---|---|---|---|---|---|
| Multinex | 23.1828 | 0.8426 | 0.124836 | 23.19 / 0.843 / 0.129 | 44,620 | 190.974 |
| Multinex-Nano | 19.4172 | 0.7422 | 0.276419 | 19.42 / 0.742 / 0.276 | 696 | 1.01 |

Timings are upper bounds because the GPUs were shared with other jobs. Full numbers are in
`results/results.json`, and reasons for the small gaps are in `results/interpretation.md`.

## Layout
`logs/` holds the raw logs of every step, `fixes.md` and `commands.txt`.
`scripts/` holds our helper scripts. `results/` holds results.json, the per-image CSV,
the figures and interpretation.md.

## Photo test (step 7)
No own photos were taken. Five free-licensed phone night photos from Wikimedia Commons
(`logs/10_my_photos.txt` lists author, license and EXIF) were resized to 1200 px and run
through both models. Run with `scripts/prep_my_photos.py`, the four `test_noref.py` commands in
`logs/10_commands.txt`, and `scripts/measure_my_photos.py`. Results are in
`results/my_photo_results.json`, `results/my_photo_notes.md` and the `results/my_photo_*.png` figures.
`results/README_submission.md` is the short Korean/English submission summary.
