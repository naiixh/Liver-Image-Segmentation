# Liver Segmentation with U-Net

## Description
Semantic segmentation of liver images using a U-Net architecture. The repository also includes the final project report in PDF format.

## Requirements
This project uses Python with PyTorch and a small set of extra libraries for image loading, plots and progress bars.

Install the required packages from the project root:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `torch`
- `torchvision`
- `Pillow`
- `matplotlib`
- `tqdm`

## How to run
Run the main program, which provides an interactive interface to train models and generate reports:

```bash
python src/main.py
```

The project uses internal absolute paths based on the repository location, so it can be launched from the project root or from inside `src/`.

## Project structure
- `src/` -> source code (training, model, utilities, report generation)
- `data/` -> dataset (images and labels)
- `models/` -> trained model weights (`.pth` files)
- `reports/` -> generated reports for each experiment
- `requirements.txt` -> Python dependencies needed to run the project

## Results
Trained models are saved in the `models/` folder, and a report for each model is stored in the `reports/` folder.
You can view existing reports or generate new ones through the main interactive interface.

## Notes
The program automatically uses CUDA if a compatible GPU is available. Otherwise, it runs on CPU.
