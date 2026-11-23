# Fixes and deviations from the README

All runs at commit e15790a963935569fff1ba6a8b0d33a791765749, conda env `multinex` (Python 3.9.23).

## 1. No conda on the server: installed Miniforge
server9 (Ubuntu 24.04) had no conda. Installed Miniforge3 into `~/miniforge3`
(user space, no sudo). Miniforge was chosen over Miniconda because it uses the
conda-forge channel and does not require accepting the
Anaconda channel Terms of Service prompt in a non-interactive script.
The README commands themselves (`conda create -n multinex python=3.9 -y`, ...)
were run unchanged. Log: `logs/03_install.log`.

## 2. CUDA: cu118 used as in the README (no fallback needed)
Driver 580.178.04 supports CUDA 11.8 wheels, so `torch 2.7.1+cu118` was
installed exactly as written. The cu121 fallback was not needed.

## 3. LOL-v1 test (Multinex and Multinex-Nano): no fix needed
Both README `Enhancement/test.py` commands ran on the first attempt with only
the README packages. `lpips` is already in the README pip list (section 1.3),
so it was not missing. Logs: `logs/05a_multinex_lolv1_attempt1.log`,
`logs/05b_nano_lolv1_attempt1.log`.

## 4. No-reference script: missing `piq`
`Enhancement/test_noref.py` line 40 does `from piq import brisque`, but `piq`
is not in the README pip list. First attempt failed with:

    ModuleNotFoundError: No module named 'piq'

Fix: `pip install piq` (installed piq==0.8.0). `pip freeze` diff before/after
shows only `+ piq==0.8.0`; torch and other packages unchanged.
Logs: `logs/07a_noref_multinex_attempt1.log`, `logs/07_fix_install_piq.log`.

## 5. Notes (not fixes)
GPU selection: `test.py` sets `CUDA_VISIBLE_DEVICES` from `--gpus` (default "0"),
so the README commands ran on physical GPU 0. Metrics do not depend on the GPU.
All 8 RTX A6000 GPUs were shared with another user's jobs at 90-100 % utilization
for the whole session; timing numbers are therefore upper bounds, not clean
latency. nvidia-smi snapshots are in `logs/06_complexity_*.log`.

Multinex latency: a torch.profiler run (`logs/06_profile_multinex.log`) shows
`aten::native_layer_norm` takes 73 % of CUDA time at 400x600 (24 calls, about
7 ms each). These are the channels-last `LayerNormalization` blocks in
`basicsr/models/archs/Multinex_arch.py`. Model code was not changed.

README typo (not used): the self-ensemble example references
`pretrained_weights/Multinex_LOLv1.pth` and `--dataset LOL_v1`, but the shipped
file is `Multinex_LOL-v1.pth`.

## 6. Step 7 data: own photos replaced by standard no-reference benchmarks
No own photos were provided. At the user's request, step 7 uses the four
no-reference datasets named in the README command (DICM, MEF, NPE, LIME). The
README gives no download link for them. Source: the public BIMEF Google Drive
folder (github.com/baidut/BIMEF), files datasets__{DICM,LIME,MEF,NPE}.zip.
`gdown --folder` failed because the folder has more than 50 files
(log: `logs/07_download_noref.log`), so the four zips were downloaded by file ID
(`logs/07_download_noref_files.log`). Counts: DICM 69, LIME 10, MEF 17, NPE 8.
SHA-256 in `logs/07_noref_data_counts.txt`. The README no-reference commands
were then run unchanged (`logs/07b_noref_multinex.log`, `logs/07c_noref_nano.log`).

The repo's `assets/NoRef_table.png` (copied to `logs/`) shows pre-dagger numbers.
The README says the `_dagger` checkpoints correspond to the updated arXiv table,
which is not in the repo, so the comparison with the repo table is indicative only.
