"""
=========================================================
MAIN SCRIPT TO RUN THE LIVER IMAGE SEGMENTATION PIPELINE
=========================================================

"""
import os
import torch
from torchvision import transforms

from dataset import LiverDataset
from paths import IMAGES_DIR, LABELS_DIR, model_path
from report import report
from train import train

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    clear()

    """Setting up the environment"""
    print("Setting up the environment...")
    # Select device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data setup
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = LiverDataset(IMAGES_DIR, LABELS_DIR, transform=transform)
    print("Dataset loaded successfully.")

    input("\nPress Enter to continue...")
    clear()

    """Main menu loop"""
    while True:
        clear()

        print("\n===== LIVER IMAGE SEGMENTATION PIPELINE =====")
        print("1. Train the model")
        print("2. Generate report")
        print("3. Exit")
        choice = input("Select an option (1-3): ")

        if choice == '1':
            clear()
            print("CAREFUL: TRAINING THE MODEL MAY TAKE A LONG TIME DEPENDING ON YOUR HARDWARE")
            print("Training may take several minutes to hours. Please be patient.")
            print("Do you want to proceed with training? (y/n)")
            proceed = input().lower()
            if proceed == 'y':
                clear()
                print("HIPERPARAMETERS SELECTION:")
                print("1. Automatic hiperparameters (recommended)")
                print("2. Custom hiperparameters")
                hiperparams_choice = input("Select an option (1-2): ")
                if hiperparams_choice == '1':
                    clear()
                    train(device, dataset, auto=True)
                elif hiperparams_choice == '2':
                    print("Select an epochs number or press Enter to use default (50): ")
                    epochs_input = input().strip()
                    epochs = int(epochs_input) if epochs_input else 50

                    print("Select a batch size or press Enter to use default (1): ")
                    batch_size_input = input().strip()
                    batch_size = int(batch_size_input) if batch_size_input else 1

                    print("Select a learning rate or press Enter to use default (1e-3): ")
                    learning_rate_input = input().strip()
                    learning_rate = float(learning_rate_input) if learning_rate_input else 1e-3

                    clear()
                    train(device, dataset, auto=False, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
                else:
                    print("Invalid option. Returning to main menu.")
            elif proceed == 'n':
                print("That's okay! You can also check the report of pre-trained models.")
            else:
                print("Invalid option. Returning to main menu.")
            input("\nPress Enter to return...")
        elif choice == '2':
            clear()
            print("Which report do you want to see?")
            print("1. Report of automatically pre-trained model")
            print("2. Report of custom pre-trained model")
            print("3. Report of the model you just trained")
            report_choice = input("Select an option (1-3): ")
            clear()
            if report_choice == '1':
                report(device, dataset, filename=model_path("default_pretrained.pth"))
            elif report_choice == '2':
                print("Which custom pre-trained model do you want to see?")
                print("1. Baseline (batch: 1, epochs: 50, learning rate: 1e-3)")
                print("2. Higher batch size (batch: 2, epochs: 50, learning rate: 1e-3)")
                print("3. Lower learning rate (batch: 1, epochs: 50, learning rate: 1e-4)")
                print("4. Lower epochs (batch: 1, epochs: 25, learning rate: 1e-3)")
                custom_choice = input("Select an option (1-4): ")
                if custom_choice == '1':
                    report(device, dataset, filename=model_path("custom_baseline.pth"))
                elif custom_choice == '2':
                    report(device, dataset, filename=model_path("custom_higher_batch.pth"))
                elif custom_choice == '3':
                    report(device, dataset, filename=model_path("custom_lower_lr.pth"))
                elif custom_choice == '4':
                    report(device, dataset, filename=model_path("custom_lower_epochs.pth"))
                else:
                    print("Invalid option. Returning to main menu.")
            elif report_choice == '3':
                if os.path.exists(model_path("last_model.pth")):
                    report(device, dataset, filename=model_path("last_model.pth"))
                else:
                    print("No model found, you need to train a model first to see its report. Returning to main menu.")
            else:
                print("Invalid option. Returning to main menu.")
            input("\nPress Enter to return...")
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please select 1, 2, or 3.")
            input("\nPress Enter to return...")


if __name__ == "__main__":
    main()
