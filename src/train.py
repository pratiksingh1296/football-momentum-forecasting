# Standard library
from pathlib import Path
import sys
from timeit import default_timer as timer
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# Local modules
from config import *
from debug import debug_print
from model import LSTMMomentum, DistilBertMomentum, HybridDistilbertMomentum
from dataset import MomentumDataset, HybridMomentumDataset

# Device Agnostic Setup
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if torch.backends.mps.is_available() #for macbook
    else 'cpu'
    )

# Print out information 
print("=" * 70)
print("Football Momentum Forecasting")
print("=" * 70)
print(f"Experiment : {EXPERIMENT}")
print(f"Model      : {MODEL_TYPE}")
print(f"Tokenizer  : {TOKENIZER_NAME}")
print(f"Device     : {device}")
print("=" * 70)

# Seed for reproducability
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

data_file_ext = 'parquet' if USE_HYBRID else 'csv'

# Loading Datasets
if USE_HYBRID:
    train_dataset = HybridMomentumDataset(f'train.{data_file_ext}', tokenizer=tokenizer, experiment=EXPERIMENT, max_length= MAX_LENGTH)
    val_dataset = HybridMomentumDataset(f'val.{data_file_ext}', tokenizer=tokenizer, experiment=EXPERIMENT, max_length= MAX_LENGTH)
else:
    train_dataset = MomentumDataset(f'train.{data_file_ext}', tokenizer=tokenizer, experiment=EXPERIMENT, max_length=MAX_LENGTH)
    val_dataset = MomentumDataset(f'val.{data_file_ext}', tokenizer=tokenizer, experiment=EXPERIMENT, max_length=MAX_LENGTH)

debug_print(f"Train Samples : {len(train_dataset):,}")
debug_print(f"Validation    : {len(val_dataset):,}")

# Creating Dataloaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# Calculating class weights for imbalanced classes
class_counts = train_dataset.data['label'].value_counts().sort_index()
total = len(train_dataset)

# Print Class Distribution
print("\nTraining Class Distribution")
print("-" * 30)
for i, count in enumerate(class_counts):
    pct = count / total * 100
    print(f"{CLASS_NAMES[i]:15} {count:6d} ({pct:.1f}%)")

'''
class weights (i) = total samples / number of classes * samples in class (i)
'''
class_weights = torch.tensor(
    [total / (NUM_CLASSES * count) for count in class_counts],
    dtype=torch.float
).to(device)  
print(f"Class weights: {class_weights}")

# Instantiate the model

if MODEL_TYPE == 'hybrid':
    with open(BASE_DIR / 'models' / EXPERIMENT / 'numeric_cols.json') as f:
        numeric_cols = json.load(f)

    model = HybridDistilbertMomentum(
        num_classes=NUM_CLASSES,
        num_numeric_features=len(numeric_cols),
        dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LEARNING_RATE)
    model_filename = MODEL_FILES[MODEL_TYPE]

elif MODEL_TYPE == 'lstm':
    model = LSTMMomentum(
        vocab_size= VOCAB_SIZE,
        embed_dim= EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        num_layers= NUM_LAYERS,
        num_classes= NUM_CLASSES,
        dropout= DROPOUT
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model_filename = MODEL_FILES[MODEL_TYPE]

elif MODEL_TYPE == 'distilbert':
    model = DistilBertMomentum(
        num_classes= NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LEARNING_RATE)
    model_filename = MODEL_FILES[MODEL_TYPE]

# Setup loss function 
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


# Training & Testing Loops
def train_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device=device):
    """
    Train a model for one epoch.

    Performs a full pass over the training dataset, computing the loss,
    backpropagating gradients, and updating model parameters.
    
    Args:
        model (torch.nn.Module): 
            The model to be trained.
        
        dataloader (torch.utils.data.DataLoader): 
            DataLoader containing the training data.
        
        loss_fn (torch.nn.Module): 
            Loss function used to compute the training loss.

        optimizer (torch.optim.Optimizer): 
            Optimizer used to update model parameters.

        device (torch.device): 
            Device on which training is performed.

    Returns:
    dict:
        Dictionary containing:
        - loss
        - accuracy
        - precision
        - recall
        - f1
    """

    all_preds = []
    all_labels =[]

    train_loss, train_acc = 0, 0

    # Train mode
    model.train()

    # Looping through
    for i, batch in enumerate(tqdm(dataloader, desc='Training', leave=False)):

        # Send data to target device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 1. Forward Propagation
        if USE_HYBRID:
            numeric_features = batch['numeric_features'].to(device)
            y_pred = model(input_ids, attention_mask=attention_mask, numeric_features=numeric_features)
        else:
            y_pred = model(input_ids, attention_mask=attention_mask)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, labels)
        train_loss += loss.item()

        # 3. Zero Grad
        optimizer.zero_grad()

        # 4. Back propagation
        loss.backward()

        # 5. Update Gradients
        optimizer.step()

        # Get prediction
        y_pred_class = torch.argmax(y_pred, dim=1) #can also just do argmax on dim=1

        all_preds.extend(y_pred_class.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate Metrics 
    precision, recall , f1_score , _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0
    )
    train_acc = accuracy_score(all_labels,all_preds)
    train_loss /= len(dataloader)
    

    return {
    "loss": train_loss,
    "accuracy": train_acc,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
}


def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device=device):
    """

    Evaluating the model.

    Performs evaluation by calculating evaluation metrics like average loss & accuracy.

    Args:
        model (torch.nn.Module): 
            The model to be evaluated.

        dataloader (torch.utils.data.DataLoader): 
            Dataloader containing the evaluation dataset.

        loss_fn (torch.nn.Module): 
            Loss function used to compute the evaluation loss

        device (torch.device, optional): 
            Device on which evaluation is performed. Defaults to device.
    """

    test_loss, test_acc = 0, 0
    all_preds = []
    all_labels = []

    # Evaluation mode
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            
            # Put data on device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 1. Forward Pass
            if USE_HYBRID:
                numeric_features = batch['numeric_features'].to(device)
                test_pred_logits = model(input_ids, attention_mask=attention_mask, numeric_features=numeric_features)
            else:
                test_pred_logits = model(input_ids, attention_mask=attention_mask)

            # 2. Calculate Loss
            loss = loss_fn(test_pred_logits, labels)
            test_loss += loss.item()

            # 3. Get Predictions
            test_pred_labels = test_pred_logits.argmax(dim=1)
        
            # 4. Get all preds & labels
            all_preds.extend(test_pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


        # 5. Calculate Metrics
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average='macro',
            zero_division=0
        )
        test_loss /= len(dataloader)
        test_acc = accuracy_score(all_labels, all_preds)

    return {
    "loss": test_loss,
    "accuracy": test_acc,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
}


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device=device,
        epochs:int = EPOCHS,
        model_path: Path = None,
        patience: int = EARLY_STOPPING_PATIENCE):
    """
    Train a PyTorch model with validation, checkpointing, and early stopping.

    This function trains a model for multiple epochs using the provided training
    DataLoader and evaluates it on the validation DataLoader after each epoch.
    Training and validation metrics are recorded for every epoch.

    The model with the lowest validation loss is saved (if `model_path` is
    provided). Training stops early if the validation loss does not improve for
    `patience` consecutive epochs.

    Args:
        model (torch.nn.Module): 
            The model to train.

        train_dataloader (torch.utils.data.DataLoader): 
            DataLoader containing the training dataset.

        val_dataloader (torch.utils.data.DataLoader):
            DataLoader containing the validation dataset.

        optimizer (torch.optim.Optimizer):
            Optimizer used to update model parameters during training.

        loss_fn (torch.nn.Module):
            Loss function used to calculate training and validation loss.

        device (torch.device, optional):
            Device on which training and evaluation are performed.
            Defaults to device.

        epochs (int, optional):
            Number of complete passes through the training dataset.
            Defaults to EPOCHS.
        
        model_path (Path):
            Path where model needs to be saved.

        patience: (int, optional):
            Number of consecutive epochs without validation loss improvement
            before early stopping is triggered.
            Defaults to 3.

    Returns:
        Returns:
    dict:
        Dictionary containing training history with the following structure:

        {
            "train": [
                {
                    "loss": float,
                    "accuracy": float,
                    "f1": float,
                    ...
                },
                ...
            ],
            "val": [
                {
                    "loss": float,
                    "accuracy": float,
                    "f1": float,
                    ...
                },
                ...
            ]
        }

    Notes:
    - The best model is determined solely by the lowest validation loss.
    - If early stopping is triggered, the returned history contains metrics only for the completed epochs.
    - The function assumes `train_one_epoch()` and `evaluate()` return dictionaries containing at least the keys `"loss"`, `"accuracy"`, and `"f1"`.

    """

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # 1. Empty results dictionary
    results = {
        "train": [],
        "val": []
    }

    # 2. Loop through training & testing steps for a number of epochs
    for epoch in tqdm(range(epochs),desc='Epochs'):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        val_metrics = evaluate(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # 3. Print out epoch results
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # 4. Update results dictionary
        results["train"].append(train_metrics)
        results["val"].append(val_metrics)

        # 5. Save model
        if model_path is not None and val_metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # reset counter on improvement

            torch.save(model.state_dict(), model_path)
            print(f"Best model saved at epoch {best_epoch} | Val Loss: {val_metrics['loss']:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
        
        # 6. Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    print(f'Best Epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}')

    #  7. Return Results dictionary
    return results


if __name__ == "__main__":

    MODEL_PATH = BASE_DIR / "models" / EXPERIMENT / model_filename
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    start_time = timer()

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs= EPOCHS,
        device=device,
        model_path = MODEL_PATH
    )

    end_time = timer()
    print(f"Total Training Time: {end_time - start_time:.3f} seconds")

    # Saving results dictionary to csv
    rows = []

    for epoch, (train_metrics, val_metrics) in enumerate(zip(results["train"], results["val"]),start=1):
        row = {
        "epoch": epoch,

        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],

        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
    }
        rows.append(row)

    RESULTS_PATH = ( BASE_DIR / 'models' / EXPERIMENT )
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH / f'{model_filename.replace(".pth", "")}_results.csv', index=False)


