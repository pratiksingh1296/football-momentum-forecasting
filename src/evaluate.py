# Standard library
from pathlib import Path
import sys
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Local modules
from debug import debug_print
from config import *
from model import LSTMMomentum, DistilBertMomentum, HybridDistilbertMomentum
from dataset import MomentumDataset, HybridMomentumDataset

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Define Model Path
MODEL_PATH = BASE_DIR / 'models' / EXPERIMENT

# Device Agnostic Code
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if torch.backends.mps.is_available() #for macbook
    else 'cpu'
    )

# Print Information 
print("=" * 70)
print("Model Evaluation")
print("=" * 70)
print(f"Experiment : {EXPERIMENT}")
print(f"Model      : {DISPLAY_NAMES[MODEL_TYPE]}")
print(f"Tokenizer  : {TOKENIZER_NAME}")
print(f"Device     : {device}")
print("=" * 70)

# Loading test datasets
data_file_ext = 'parquet' if USE_HYBRID else 'csv'

if USE_HYBRID:
    test_dataset = HybridMomentumDataset(f'test.{data_file_ext}', tokenizer=tokenizer, experiment=EXPERIMENT, max_length= MAX_LENGTH)
else:
    test_dataset = MomentumDataset(f'test.{data_file_ext}', tokenizer=tokenizer, experiment=EXPERIMENT, max_length= MAX_LENGTH)
debug_print(f"Test samples : {len(test_dataset):,}")

# Creating Test Dataloader
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Loading the Model
if USE_HYBRID:
    with open(BASE_DIR / 'models' / EXPERIMENT / 'numeric_cols.json') as f:
        numeric_cols = json.load(f)

    model = HybridDistilbertMomentum(
        num_classes=NUM_CLASSES,
        num_numeric_features=len(numeric_cols),
        dropout=DROPOUT
    ).to(device)


else:

    if MODEL_TYPE == 'lstm':
        model = LSTMMomentum(
            vocab_size= VOCAB_SIZE,
            embed_dim= EMBED_DIM,
            hidden_dim = HIDDEN_DIM,
            num_layers= NUM_LAYERS,
            num_classes= NUM_CLASSES,
            dropout= DROPOUT
        ).to(device)
    else:
        model = DistilBertMomentum(
            num_classes=NUM_CLASSES,
            dropout=DROPOUT
        ).to(device)

# Load in the save state_dict()

model.load_state_dict(torch.load(MODEL_PATH / MODEL_FILES[MODEL_TYPE], map_location=device, weights_only=True))

# Evaluate Loaded Model on Test Data
def evaluate_model(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        model_name: str,
        display_name: str | None = None,
        save_report: bool = True):
    """
    Evaluate a trained classification model on a test dataset.

    The model is switched to evaluation mode and predictions are generated
    for each batch in the provided dataloader. Predicted labels and ground
    truth labels are collected to compute a classification report containing
    precision, recall, F1-score, and support for each class. Optionally,
    the report can be saved to disk.

    Args:
        model (torch.nn.Module):
            Trained PyTorch model to evaluate.

        dataloader (torch.utils.data.DataLoader):
            DataLoader containing the evaluation dataset. Each batch is
            expected to contain the keys:
            - 'input_ids': Input tensor of token IDs.
            - 'attention_mask':  Tensor indicating which tokens should be attended to (1 = actual token, 0 = padding token).
            - 'label': Ground truth class labels.

        device (torch.device):
            Device used for inference (e.g., CPU or CUDA).

        model_name (str):
            Internal model identifier used when saving the evaluation
            report to disk.

        display_name (str | None, optional):
            Human-readable model name used when printing and writing
            evaluation results. If None, ``model_name`` is used.
            Defaults to None.

        save_report (bool, optional):
            Whether to save the generated classification report to a text
            file. Defaults to True.

    Returns:
        tuple[str, list[int], list[int]]:
        A tuple containing:
        - report: Classification report as a formatted string.
        - all_preds: Predicted class labels for the entire dataset.
        - all_labels: Ground truth class labels for the entire dataset.

    Notes:
        - The model is evaluated under ``torch.inference_mode()`` to reduce
        memory usage and improve inference speed.
        - Predictions are obtained using ``argmax(dim=1)`` on the model
        output logits.
        - The classification report is generated using
        ``sklearn.metrics.classification_report``.

    """

    
    all_preds = []
    all_labels = []

    # Evaluation Mode
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if USE_HYBRID:
                numeric_features = batch['numeric_features'].to(device)
                logits = model(input_ids, attention_mask=attention_mask, numeric_features=numeric_features)
            else:
                logits = model(input_ids,attention_mask=attention_mask)

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=CLASS_NAMES
        )
    
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    
    name = display_name or model_name

    print(f"{name} - Test Set Evaluation")
    print("="*45)
    print(report)
            
    # Save Report
    if save_report:
        REPORTS_PATH = BASE_DIR / 'reports' / 'metrics' / EXPERIMENT
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)

        # Save Classification Report
        with open(REPORTS_PATH / f'{model_name}.txt', 'w') as f:
            f.write(f" {name} - Test Set Evaluation\n")
            f.write("="*45 + "\n")
            f.write(report)

        with open(REPORTS_PATH / f"{model_name}_metrics.txt", "w") as f:
            json.dump(report_dict, f, indent=4)

        # Save predictions and labels for analysis
        np.save(REPORTS_PATH / f'{model_name}_preds.npy', np.array(all_preds))
        np.save(REPORTS_PATH / f'{model_name}_labels.npy', np.array(all_labels))

    return report, all_preds, all_labels

if __name__ == "__main__":
    evaluate_model(
        model, 
        test_dataloader, 
        device, 
        model_name=MODEL_FILES[MODEL_TYPE].replace(".pth",""),
        display_name=DISPLAY_NAMES[MODEL_TYPE]
        )

