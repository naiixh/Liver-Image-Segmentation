"""
=========================================================
REPORTING PIPELINE TO SHOWCASE THE PERFORMANCE OF THE TRAINED U-NET MODEL
Includes: 
=========================================================

"""
import os
import random
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from unet import UNet
from paths import REPORTS_DIR, report_path
import utils

def clear():
    os.system("cls" if os.name == "nt" else "clear")

#showcasing and visualization
def show(sample):
    img = sample["img"]
    mask = sample["mask"]
    pred = sample["pred"]
    iou = sample["iou"]

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    ax[0].imshow(img.squeeze(), cmap="gray")
    ax[0].set_title("Image")

    ax[1].imshow(mask.squeeze(), cmap="gray")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(pred.squeeze(), cmap="gray")
    ax[2].set_title(f"Prediction (IoU={iou:.3f})")

    plt.show()

def report(device, dataset, filename="initial.pth"):
    print("WELCOME TO THE REPORTING PIPELINE")
    print("This pipeline will evaluate the performance of the trained U-Net model on the validation set and showcase some examples of the predictions.")
    print("Loading the validation set...")
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False) 

    # Loading the best static model and setting it to evaluation mode
    model = UNet(n_class=2)

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Printing final evaluation metrics on the validation set
    results = []
    mIoU, mPA = utils.evaluate(model, val_loader, device, True, results)

    # Showcasing some examples of the predictions
    while True:
        clear()
        print("===== FINAL RESULTS =====")
        print(f"mIoU: {mIoU:.4f}")
        print(f"mPA: {mPA:.4f}")

        print("\n===== VISUALIZATION MENU =====")
        print("1. Best prediction")
        print("2. Worst prediction")
        print("3. Random prediction")
        print("4. Comparative of IoU and Pixel Accuracy")
        print("5. Training loss curve")
        print("6. Continue report generation")

        choice = input("Select option: ")

        if choice == "1":
            show(max(results, key=lambda x: x["iou"]))

        elif choice == "2":
            show(min(results, key=lambda x: x["iou"]))

        elif choice == "3":
            show(random.choice(results))

        elif choice == "4":
                plt.figure()
                plt.scatter([r["pa"] for r in results], [r["iou"] for r in results])
                plt.xlabel("Pixel Accuracy")
                plt.ylabel("IoU")
                plt.title("IoU vs Pixel Accuracy correlation")

                plt.show()
        
        elif choice == "5":
            if "train_losses" not in checkpoint:
                print("No training loss data found in the checkpoint. Cannot plot training loss curve.")
                continue
            plt.figure()
            plt.plot(checkpoint["train_losses"])
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("Training Loss Curve")
            plt.show()
            
        elif choice == "6":
            break
        else:
            continue


    # Saving the final report
    print("\n===== SAVING FINAL REPORT =====")
    print("Do you want to save the report? (y/n)")
    save_choice = input().lower()
    if save_choice != 'y':
        print("Report not saved. Returning to main menu.")
        return
    print("Choose a name for the report (without extension): ")
    report_name = input().strip()
    if not report_name:
        report_name = "final_report"

    
    print("Saving report...")
    best_iou = max(results, key=lambda x: x["iou"])
    best_pa = max(results, key=lambda x: x["pa"])
    worst_iou = min(results, key=lambda x: x["iou"])
    worst_pa = min(results, key=lambda x: x["pa"])
    train_losses = checkpoint.get("train_losses", [])

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(report_path(f"{report_name}.txt"), "w") as f:

        f.write("===== LIVER SEGMENTATION REPORT =====\n\n")

        f.write("Model\n")
        f.write("-----\n")
        f.write("Architecture: U-Net (Ronneberger et al. 2015)\n\n")

        f.write("Dataset\n")
        f.write("-----\n")
        f.write("Total images: 400\n")
        f.write("Train: 360\n")
        f.write("Validation: 40\n")
        f.write("Image size: 256x256\n\n")

        f.write("Hiperparameters\n")
        f.write("-----\n")
        f.write(f"Epochs: {checkpoint['epochs']}\n")
        f.write(f"Batch size: {checkpoint['batch_size']}\n")
        f.write(f"Learning rate: {checkpoint['learning_rate']}\n\n")

        f.write("Training Loss\n")
        f.write("-----\n")
        if train_losses:
            f.write(f"Best training loss: {min(train_losses):.6f}\n\n")
        else:
            f.write("No training loss data found in this checkpoint.\n\n")

        f.write("Validation Performance\n")
        f.write("-----\n")
        f.write(f"Mean IoU: {mIoU:.4f}\n")
        f.write(f"Mean Pixel Accuracy: {mPA:.4f}\n\n")

        f.write("Best segmentation\n")
        f.write("-----\n")
        f.write(f"IoU: {best_iou['iou']:.4f}\n")
        f.write(f"Pixel Accuracy: {best_pa['pa']:.4f}\n\n")

        f.write("Worst segmentation\n")
        f.write("-----\n")
        f.write(f"IoU: {worst_iou['iou']:.4f}\n")
        f.write(f"Pixel Accuracy: {worst_pa['pa']:.4f}\n")



    print(f"Report saved as {report_name}.txt.")
    print("You can check the results folder to see the saved predictions.")
