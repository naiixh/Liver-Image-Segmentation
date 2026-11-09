"""
=========================================================
TRAINING PIPELINE FOR LIVER IMAGE SEGMENTATION USING U-NET
Includes: Data loading, model training, validation,
and evaluation metrics (mIoU and mPA).
=========================================================

"""
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm


from unet import UNet
from paths import MODELS_DIR, model_path
import utils

def train(device, dataset, auto=True, epochs=25, batch_size=1, learning_rate=1e-3):

    print("WELCOME TO THE TRAINING PIPELINE")
        
    """---- HIPERPARAMETERS ----""" 
    if auto:
        # Setting epochs
        epochs = 1000
        charEpoch = "∞"
        early_stopping_patience = 5
        min_delta = 1e-4
        epochs_without_improvement = 0

        # Setting batch size 
        if torch.cuda.is_available():
            memory = torch.cuda.get_device_properties(0).total_memory
            batch_size = 4 if memory > 10 * 2**30 else 2
        else:
            batch_size = 1

        # Setting dinamic learning rate
        learning_rate = 1e-3
        charLearningRate = "Dynamic"
    else:
        epochs = epochs
        charEpoch = epochs
        batch_size = batch_size
        learning_rate = learning_rate
        charLearningRate = learning_rate
        early_stopping_patience = None
        min_delta = 0
        epochs_without_improvement = 0
    loss_function_name = "CrossEntropyLoss"
    optimizer_name = "Adam"

    print("\n===== HYPERPARAMETERS =====")
    print(f"Epochs: {charEpoch}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {charLearningRate}")
    print(f"Loss function: {loss_function_name}")
    print(f"Optimizer: {optimizer_name}")
    print("============================\n")

    """---- CHECKPOINT INITIALIZATION ----"""
    checkpoint = {
        "model_state_dict": None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_losses": []
    }

    """---- DATA SETUP ----"""
    train_dataset, val_dataset = random_split(dataset, [360, 40])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    """---- MODEL, LOSS AND OPTIMIZER ----"""
    model = UNet(n_class=2).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=2) if auto else None

    best_iou = 0
    """---- TRAINING AND VALIDATION ----"""
    print("\n===== TRAINING =====")

    for epoch in range(epochs):

        # TRAINING PHASE   
        model.train()
        
        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{charEpoch}",
            unit="batch",
            dynamic_ncols=True,
            leave=True
        )

        running_loss = 0.0
        for i, (imgs, masks) in enumerate(loop):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            if logits.shape[2:] != masks.shape[1:]:
                logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(loss=loss.item(),avg_loss=running_loss / (i + 1))

        avg_train_loss = running_loss / len(train_loader)
        checkpoint["train_losses"].append(avg_train_loss)

        #  VALIDATION PHASE
        mIoU, mPA = utils.evaluate(model, val_loader, device)

        # Save best model based on mIoU 
        improved = mIoU > best_iou + min_delta

        should_stop = False

        if improved: 
            best_iou = mIoU
            epochs_without_improvement = 0
            checkpoint["model_state_dict"] = model.state_dict()
            if auto:
                checkpoint["learning_rate"] = optimizer.param_groups[0]['lr']
                checkpoint["epochs"] = epoch + 1

            MODELS_DIR.mkdir(exist_ok=True)
            torch.save(checkpoint, model_path("last_model.pth"))
        elif auto:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                should_stop = True

        scheduler.step(mIoU) if auto else None

        print(f"\n===== MODEL EVALUATION | EPOCH {epoch+1} =====")
        print(f"Mean IoU (mIoU): {mIoU:.4f}")
        print(f"Mean Pixel Accuracy (mPA): {mPA:.4f}")
        if auto:
            print(f"Best mIoU: {best_iou:.6f}")
            print(f"Epochs without improvement: {epochs_without_improvement}/{early_stopping_patience}")
        print("===============================")

        if should_stop:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break

    print("\nTraining successfully completed!")
    print("You can now generate a report to see the final results of the model.")
