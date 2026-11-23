# Multinex reproduction / Multinex 재현

Repo: https://github.com/albrateanu/Multinex, commit `e15790a` (e15790a963935569fff1ba6a8b0d33a791765749).

## Environment / 환경
GPU NVIDIA RTX A6000 x8 (shared with other jobs / 다른 작업과 공유), driver 580.178.04,
CUDA 11.8 (torch build; driver supports 13.0), cuDNN 90100,
PyTorch 2.7.1+cu118, Python 3.9.23, Ubuntu 24.04.4 LTS. Install: conda + README packages + piq.

## Main results / 주요 결과
LOL-v1 test set (15 pairs / 쌍), README test.py:

| Model | PSNR | SSIM | LPIPS | Paper (PSNR/SSIM/LPIPS) | Params | ms 600x400 | ms 1080p |
|---|---|---|---|---|---|---|---|
| Multinex | 23.1828 | 0.8426 | 0.124836 | 23.19 / 0.843 / 0.129 | 44620 | 190.974 | 1593.531 |
| Multinex-Nano | 19.4172 | 0.7422 | 0.276419 | 19.42 / 0.742 / 0.276 | 696 | 1.01 | 15.341 |

Photos (5 phone night photos from Wikimedia Commons, long side 1200 px, LOL-v1 weights), mean over photos:
사진 (5장, Wikimedia Commons의 휴대폰 야간 사진, 긴 변 1200 px, LOL-v1 가중치), 평균:
NIQE input 4.6910, Multinex 3.6306, Nano 3.3938;
BRISQUE input 31.1830, Multinex 34.2491, Nano 24.9779;
ms Multinex 420.92, Nano 4.308 (lower is better / 낮을수록 좋음).
Benchmark no-reference results (DICM/LIME/MEF/NPE) are in results.json `other_runs`.

Notes: timings are upper bounds on shared GPUs (photo timing ran on an idle GPU, see gpu_util_pct).
The photos are not our own: they are free-licensed photos (CC0 / CC BY / CC BY-SA), with author and
license listed per photo in my_photo_results.json `source` and logs/10_my_photos.txt.
참고: 공유 GPU이므로 시간은 상한값입니다. 사진은 직접 찍은 것이 아니라 자유 라이선스 사진이며,
작가와 라이선스는 my_photo_results.json의 `source`와 logs/10_my_photos.txt에 있습니다.

## Commands / 실행 명령
```
cd ~/multinex_repro && git clone https://github.com/albrateanu/Multinex.git
cd ~/multinex_repro/Multinex && git checkout e15790a963935569fff1ba6a8b0d33a791765749 && git log -1 > ../logs/02_commit.txt
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh && bash /tmp/mf.sh -b -p ~/miniforge3
conda create -n multinex python=3.9 -y
conda activate multinex
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm pytorch_msssim
cd ~/multinex_repro/Multinex && pip install -e .
cd ~/multinex_repro/data_raw && gdown 1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu
cd ~/multinex_repro && unzip -q -o data_raw/LOLv1.zip -d Multinex/data -x "*.DS_Store"
python Enhancement/test.py --opt Options/Multinex_LOL-v1.yaml --weights pretrained_weights/Multinex_LOL-v1.pth --dataset LOL-v1
python Enhancement/test.py --opt Options/Multinex-Nano_LOL-v1.yaml --weights pretrained_weights/Multinex-Nano_LOL-v1.pth --dataset LOL-v1
python ../scripts/per_image_metrics.py --opt Options/Multinex_LOL-v1.yaml --weights pretrained_weights/Multinex_LOL-v1.pth --model_name Multinex --test_log ../logs/05a_multinex_lolv1_attempt1.log --out_csv ../logs/06_per_image_Multinex.csv
python ../scripts/per_image_metrics.py --opt Options/Multinex-Nano_LOL-v1.yaml --weights pretrained_weights/Multinex-Nano_LOL-v1.pth --model_name Multinex-Nano --test_log ../logs/05b_nano_lolv1_attempt1.log --out_csv ../logs/06_per_image_Multinex-Nano.csv
CUDA_VISIBLE_DEVICES=6 python -m basicsr.complexity --opt Options/Multinex_LOL-v1.yaml --device cuda --warmup 10 --runs 50 --resolutions 400x600,1080x1920,256x256
CUDA_VISIBLE_DEVICES=6 python -m basicsr.complexity --opt Options/Multinex-Nano_LOL-v1.yaml --device cuda --warmup 10 --runs 50 --resolutions 400x600,1080x1920,256x256
CUDA_VISIBLE_DEVICES=6 python ../scripts/profile_ops.py Options/Multinex_LOL-v1.yaml
python Enhancement/test_noref.py --opt Options/Multinex_LOL-v2-synthetic.yaml --weights pretrained_weights/Multinex_dagger_NoRef.pth --input_dirs ../my_inputs --result_dir results/Multinex_NoRef
pip install piq
cd data_raw/bimef && gdown --folder https://drive.google.com/drive/folders/0B_FjaR958nw_djVQanJqeEhUM1k
cd data_raw/bimef && gdown 0B_FjaR958nw_eHhvUUN6MzBCQXc -O datasets__DICM.zip
cd data_raw/bimef && gdown 0B_FjaR958nw_QUpMeFlmYW5MUVE -O datasets__LIME.zip
cd data_raw/bimef && gdown 0B_FjaR958nw_NmdONjU1eDJsVW8 -O datasets__MEF.zip
cd data_raw/bimef && gdown 0B_FjaR958nw_VC1ueFBVWXhkUTA -O datasets__NPE.zip
python Enhancement/test_noref.py --opt Options/Multinex_LOL-v2-synthetic.yaml --weights pretrained_weights/Multinex_dagger_NoRef.pth --input_dirs data/DICM data/MEF data/NPE data/LIME --result_dir results/Multinex_NoRef
python Enhancement/test_noref.py --opt Options/Multinex-Nano_LOL-v2-synthetic.yaml --weights pretrained_weights/Multinex-Nano_dagger_NoRef.pth --input_dirs data/DICM data/MEF data/NPE data/LIME --result_dir results/Multinex-Nano_NoRef
CUDA_VISIBLE_DEVICES=6 python ../scripts/measure_v2.py
cd ~/multinex_repro/Multinex && python ../scripts/prep_my_photos.py
python Enhancement/test_noref.py --opt Options/Multinex_LOL-v1.yaml --weights pretrained_weights/Multinex_LOL-v1.pth --input_dirs ../my_inputs_resized --result_dir results/my_photos/Multinex_LOL-v1 --gpus 7
python Enhancement/test_noref.py --opt Options/Multinex-Nano_LOL-v1.yaml --weights pretrained_weights/Multinex-Nano_LOL-v1.pth --input_dirs ../my_inputs_resized --result_dir results/my_photos/Multinex-Nano_LOL-v1 --gpus 7
python Enhancement/test_noref.py --opt Options/Multinex_LOL-v2-synthetic.yaml --weights pretrained_weights/Multinex_dagger_NoRef.pth --input_dirs ../my_inputs_resized --result_dir results/my_photos/Multinex_NoRef --gpus 7
python Enhancement/test_noref.py --opt Options/Multinex-Nano_LOL-v2-synthetic.yaml --weights pretrained_weights/Multinex-Nano_dagger_NoRef.pth --input_dirs ../my_inputs_resized --result_dir results/my_photos/Multinex-Nano_NoRef --gpus 7
CUDA_VISIBLE_DEVICES=7 PHYS_GPU=7 python ../scripts/measure_my_photos.py
```

## Contents / 폴더 내용
1. `README_submission.md`
2. `interpretation.md`
3. `logs`
4. `lol_input.png`
5. `lol_side_by_side.png`
6. `lol_worst_side_by_side.png`
7. `mine_input.png`
8. `mine_side_by_side.png`
9. `my_photo_crop.png`
10. `my_photo_input.png`
11. `my_photo_notes.md`
12. `my_photo_results.json`
13. `my_photo_side_by_side.png`
14. `per_image_lolv1.csv`
15. `results.json`
16. `scripts`

`logs/` holds the original logs, `logs/raw/` all step logs, `scripts/` our helper scripts.
`logs/`는 원본 로그, `logs/raw/`는 모든 단계 로그, `scripts/`는 보조 스크립트입니다.
