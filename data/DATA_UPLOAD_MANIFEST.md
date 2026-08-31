# Data Upload Manifest

Upload these runtime inputs to the Code Ocean `/data` pane. Do not place them in `/code`.

- `/data/oxford-iiit-pet/images`
- `/data/oxford-iiit-pet/annotations`
- `/data/checkpoints/resnet18_pet37.pt`
- `/data/checkpoints/mobilenet_v2_pet37.pt`
- `/data/checkpoints/densenet121_pet37.pt`
- `/data/checkpoints/vgg16_pet37.pt`
- `/data/perceptual/alexnet-owt-7be5be79.pth`
- `/data/cifar10/cifar-10-python.tar.gz` or `/data/cifar10/cifar-10-batches-py/` for CPU-only CIFAR-10 routes
- `/data/hf_mirror/imagenet1k_val_1k` for CPU-only ImageNet subset routes
- `/data/checkpoints/cifar/CIFAR10_ResNet18_epoch_20.pt`
- `/data/checkpoints/imagenet/deit_tiny_patch16_224.fb_in1k/model.safetensors`
- `/data/checkpoints/imagenet/deit_small_patch16_224.fb_in1k/model.safetensors`
- `/data/checkpoints/torch_home/hub/checkpoints/vit_b_16-c867db91.pth`
- `/data/checkpoints/torch_home/hub/checkpoints/swin_t-704ceda3.pth`
- `/data/checkpoints/torch_home/hub/checkpoints/swin_v2_t-b137f0e2.pth`
- `/data/checkpoints/robustbench/cifar10/Linf/Wong2020Fast.pt`
- `/data/checkpoints/robustbench/cifar10/Linf/Rice2020Overfitting.pt`
- `/data/checkpoints/robustbench/cifar10/Linf/Engstrom2019Robustness.pt`
- `/data/checkpoints/robustbench/imagenet/Linf/Salman2020Do_R18.pt`
- `/data/checkpoints/robustbench/imagenet/Linf/Mo2022When_ViT-B.pt`
- `/data/checkpoints/robustbench/imagenet/Linf/Mo2022When_Swin-B.pt`
- `/data/checkpoints/robustbench/imagenet/Linf/Engstrom2019Robustness.pt`
- `/data/third_party/DeepRobust`
- `/data/third_party/RobustBench`
- `/data/third_party/AutoAttack`
- `/data/THIRD_PARTY_NOTICES.txt` (recommended)

The capsule run starts in `/code`, reads only from `/data`, and writes downloadable outputs only to `/results`.

The lightweight paper-source CSV/JSON files under `/code/assets/paper_source_data` are expected fixtures for regression checks.
The primary CPU-only reproduction path reads `/data` inputs and writes generated outputs under `/results`.
