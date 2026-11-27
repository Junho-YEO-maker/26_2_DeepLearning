# denoising_SEM

**복원한 SEM 이미지로 치수를 재도 되는가?**

빠른 저선량 SEM 스캔을 딥러닝으로 복원할 때, 이미지 품질 지표(PSNR/SSIM)가 좋은 모델이 측정(CD/LER)에도 좋은가?

## Models (4종)

| Model | Architecture | Property |
|-------|-------------|----------|
| DnCNN | ResNet blocks x8 | 평균 성질 → 흐릿하지만 편향 적음 |
| GAN | U-Net + PatchGAN | 선명하지만 거칠기를 지어낼 수 있음 |
| CycleGAN | ResNet blocks x6 | 쌍 없이 학습 가능 |
| Diffusion | U-Net + DDPM | 다중 샘플로 불확실성 추정 가능 |

## Quick Start

```bash
python -m data.synthetic_sem --output data/synthetic_sem_dataset
python train.py --model dncnn --noise medium --epochs 100
python evaluate.py --noise medium
```
