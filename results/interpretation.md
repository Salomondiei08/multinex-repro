Multinex minus Table 1: PSNR -0.0072 dB, SSIM -0.0004, LPIPS -0.0042 (ours 23.1828 / 0.8426 / 0.124836).
Nano minus Table 1: PSNR -0.0028 dB, SSIM +0.0002, LPIPS +0.0004 (ours 19.4172 / 0.7422 / 0.276419).
Metric implementation: VERIFIED. test.py uses utils.PSNR -> basicsr.metrics.psnr_ssim.calculate_psnr (RGB, [0,1], no crop), not skimage, and utils.calculate_ssim; the paper ships this code, so PSNR/SSIM gaps are at print-rounding level.
GT-mean: VERIFIED off. The README command has no --GT_mean, so args.GT_mean is False in Enhancement/test.py.
Image I/O: VERIFIED. cv2.imread BGR->RGB, float32/255 on lossless PNGs; metrics use the float output before it is saved.
GPU nondeterminism: VERIFIED negligible. An independent second run (scripts/per_image_metrics.py) reproduced both models' means to the printed precision.
Different weights: VERIFIED same. Multinex_LOL-v1.pth and Nano LOL-v1 blobs are unchanged since the first weights commit bf33e08 (2026-04-12); the _dagger checkpoints are only for no-reference runs.
LPIPS gap: GUESS. test.py uses LPIPS alex v0.1 on float outputs; the paper may have used another LPIPS version or saved uint8 images. Not checked against the paper's code.
Own photo: NOT PROVIDED; substitute DICM/18.JPG (median-NIQE DICM image): Multinex brightens the lit pier but leaves sky and water near black (still under-exposed) with a green cast; Nano lifts the shadows into a navy-blue cast with blotchy noise.
