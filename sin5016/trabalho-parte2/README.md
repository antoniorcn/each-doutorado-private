# Face Recognition Trainer

`train_faces.py` implements a configurable PyTorch training loop tailored to the CelebA-style dataset.

## Usage

```bash
python train_faces.py path/to/celeba/root \
  --height 160 \
  --width 120 \
  --batch-size 64 \
  --epochs 25 \
  --lr 0.0005 \
  --num-workers 8 \
  --num-classes 2000 \
  --log-file logs/celeba.log \
  --model-out saved/celeba_face.pt \
  --use-cuda
```

Arguments:

- `dataset_dir`: root directory containing one subfolder per identity (ImageFolder layout).
- `--height` / `--width`: resize to a fixed height and center-crop to the desired width.
- `--batch-size`, `--epochs`, `--lr`, `--weight-decay`, `--dropout`, `--num-workers`: Hyperparameters exposed on the CLI.
- `--num-classes`: number of face classes. Defaults to 2000.
- `--log-file` / `--model-out`: paths where logs and model checkpoints are saved.
- `--seed`: randomness seed for reproducibility.
- `--gpu`: prefer GPU when available.

Metrics (accuracy + macro F1) and epoch losses are logged to both console and the provided log file.

Example:
```bash
python train_faces.py --image-dir C:/git/dados/sin5016/Celeb_A/img_align_celeba --input-file  C:/git/dados/sin5016/celeb_a_labels.csv --height 160 --width 120 --batch-size 256 --epochs 25 --lr 0.0005 --num-workers 8 --num-classes 3113 --log-file logs/celeba.log --model-out saved/celeba_face.pt --train-split 0.7 --val-split 0.15 --test-split 0.15 --gpu
```