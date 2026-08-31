# VF-AdvJPEG Public Release

This directory is the public code release for VF-AdvJPEG.

It reproduces the revised IEEE SPL manuscript's CPU-only headline surface. The scientific question is whether the proxy-teacher discrepancy is structured enough to be fitted offline as a reusable calibration artifact, replacing repeated online sampling.

- Table I source data and corrected Static Q90 retention
- Fig. 1 JPEG-aware mechanism schematic
- supporting efficiency-retention plot
- CPU pairwise source data for the 12 ordered Pet backbone pairs
- CPU-only `/data` contract for CIFAR-10, ImageNet-1k subset, RobustBench checkpoints, transformer checkpoints, DeepRobust, RobustBench, and AutoAttack inputs when the expanded routes are enabled

This release intentionally excludes manuscript, LaTeX, PDF, arXiv, and submission-packaging files.

## Directory layout

- `src/vf_advjpeg`: runtime package subset only
- `scripts/run_public_release.py`: single entrypoint used by `run.sh`
- `configs/public_release.yaml`: release reproduction config
- `assets/pet37_ei_cpu_splits.json`: tracked split manifest
- `assets/canonical_expected_metrics.json`: regression baseline for the revised manuscript tables and figures
- `assets/paper_source_data`: lightweight CSV/JSON expected fixtures used after the cold-start run for regression checks
- `assets/paper_source_data/calibration/vf_calibration_manifest.json`: calibration metadata used in headline metrics
- `assets/data_assets_manifest.json`: expected `/data` contents and checkpoint checksums
- `assets/environment_lock.json`: starter environment lock, overlay package lock, and runtime policy

## Expected starter environment

- `PyTorch (2.4.0, CUDA 12.4.0, Mambaforge24.5.0-0, Python3.12.4, Ubuntu22.04)`
- The reproducible run is still CPU-only and sets `CUDA_VISIBLE_DEVICES=""`.

## Expected `/data`

- `/data/oxford-iiit-pet/images`
- `/data/oxford-iiit-pet/annotations`
- `/data/checkpoints/resnet18_pet37.pt`
- `/data/checkpoints/mobilenet_v2_pet37.pt`
- `/data/checkpoints/densenet121_pet37.pt`
- `/data/checkpoints/vgg16_pet37.pt`
- `/data/perceptual/alexnet-owt-7be5be79.pth`
- optional CPU-only expanded inputs under `/data/cifar10`, `/data/hf_mirror`, `/data/checkpoints/imagenet`, `/data/checkpoints/torch_home`, `/data/checkpoints/robustbench`, and `/data/third_party`

All generated outputs are written to `/results`.
