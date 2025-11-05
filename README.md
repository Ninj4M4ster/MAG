# MAG
Metal artifact generator for CT images for course on advanced neural networks

## Venv creation

```bash
python3.12 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
```

## Pipeline usage

Before starting any pipeline, run following command to start tensorboard logging:

```bash
tensorboard --logdir experiments
```

To see available arguments and options to run experiments, run:

```bash
python3 main.py --help
```

## Data

See the data README for dataset details: [data/README.md](data/README.md)

## Metrics

- HISM (Histogram Intersection Similarity Method)
- FID (Fréchet Inception Distance)
- Human evalation (Blinded radiologist Turing test)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)

## Frameworks/libraries

- PyTorch
- Matplotlib
- Jupyter
- scikit-learn
- pandas
- Pydicom
