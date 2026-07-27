# Standard library
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class LSTMMomentum(nn.Module):
    """
    LSTM-based model for sequence modeling and classification.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary used by the embedding layer.
    embed_dim : int
        Dimensionality of the token embeddings.
    hidden_dim : int
        Number of hidden units in each LSTM layer.
    num_layers : int
        Number of stacked LSTM layers.
    num_classes : int
        Number of output classes for the final classification layer.
    dropout : float
        Dropout probability applied between LSTM layers and/or
        before the classifier. Must be in the range [0.0, 1.0].

    Attributes
    ----------
    embedding : nn.Embedding
        Learns dense vector representations for input tokens.
    lstm : nn.LSTM
        Recurrent network that processes embedded sequences.
    classifier : nn.Linear
        Maps LSTM outputs to class logits.

    """
    def __init__(self, 
                vocab_size: int,
                embed_dim: int, 
                hidden_dim: int, 
                num_layers: int, 
                num_classes: int, 
                dropout: float):
        super().__init__()

        self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim
            )
            
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(
            p=dropout
        )
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=num_classes
        )

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
    Perform a forward pass through the network.

    Parameters
    ----------
    x : Tensor
        Input token IDs of shape (batch_size, sequence_length).

    Returns
    -------
    Tensor
        Classification logits of shape (batch_size, num_classes).
    """
        x = self.embedding(x)  

        _, (h, _) = self.lstm(x)

        x = h[-1]

        x = self.dropout(x)

        x = self.fc(x)

        return x
    

# DistilBERT Momentum Model
class DistilBertMomentum(nn.Module):
    """
    DistilBERT-based classification model with a dropout layer and linear classifier.

    This module wraps a pre-trained DistilBERT model, extracts the representation of the 
    classification token ([CLS]), applies dropout for regularization, and projects 
    the features to the target number of classes.

    Args:
        num_classes (int): The number of target classes for classification.
        dropout (float): The dropout probability applied to the [CLS] token embedding.

    Attributes:
        bert (Automodel): Pre-trained BERT transformer model.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): Linear layer mapping BERT hidden size to num_classes.

    """
    def __init__(self, num_classes: int, dropout: float, local_path: str = 'distilbert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(local_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_classes
        )
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing token indices.
            attention_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) masking padded tokens (1 for non-masked, 0 for masked).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract the embedding of the [CLS] token (first token of the sequence)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_embedding)
        logits = self.classifier(x)

        return logits


class HybridDistilbertMomentum(nn.Module):
    """
    Hybrid DistilBERT-based classification model that combines text embeddings
    with engineered numerical features.

    This model encodes textual input using a pre-trained DistilBERT model and
    projects additional numerical features into a lower-dimensional representation.
    The text and numerical representations are concatenated and passed through
    a linear classifier to produce class logits.

    Args:
        num_classes (int): Number of target classes for classification.
        num_numeric_features (int): Number of numerical input features.
        dropout (float): Dropout probability applied to the DistilBERT representation and within the numerical projection network.
        projection_dim (int, optional): Dimensionality of the projected numerical feature representation. Defaults to 32.

    Attributes:
        bert (AutoModel): Pre-trained DistilBERT encoder.
        dropout (nn.Dropout): Dropout layer applied to the text representation.
        numeric_projection (nn.Sequential): Feed-forward network that projects numerical features into a lower-dimensional representation of size projection_dim.
        classifier (nn.Linear): Linear layer mapping the concatenated text and numerical representations to the output classes.
    """

    def __init__(self, num_classes: int, num_numeric_features: int, dropout: float, projection_dim: int = 32):
        super().__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)

        # Projection of Numeric features
        self.numeric_projection = nn.Sequential(
            nn.Linear(num_numeric_features, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(self.bert.config.hidden_size + projection_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, numeric_features: torch.Tensor) -> torch.Tensor :
        """
        Performs a forward pass of the hybrid model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing token IDs.
            attention_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicating valid tokens (1 for valid tokens, 0 for padding).
            numeric_features (torch.Tensor): Tensor of shape (batch_size, num_numeric_features) containing engineered numerical features.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the embedding of the [CLS] token (first token of the sequence)
        text_repr = outputs.last_hidden_state[:, 0, :]

        text_repr = self.dropout(text_repr)

        numeric_repr = self.numeric_projection(numeric_features)

        fused_repr = torch.cat([text_repr , numeric_repr], dim=1)

        return self.classifier(fused_repr)


# Test Block

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

if __name__ == "__main__":

    # LSTM Model Test Block
    print(f"LSTM Test Block")
    print("="*45)
    model = LSTMMomentum(
        vocab_size=30522,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=3,
        dropout=0.3
    )
    x = torch.randint(0, 30522, (8, 128))
    out = model(x)
    print(out.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # DistilBert Model Test Block
    print(f"\nDistilBERT Test Block")
    print("="*45)
    dummy_text = ["The player scored a goal from outside the box."] * 8 # batch of 8

    dummy_input = tokenizer(
        dummy_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    distilbert_model = DistilBertMomentum(num_classes=3, dropout=0.3)
    with torch.inference_mode():
        logits = distilbert_model(
            input_ids=dummy_input['input_ids'],
            attention_mask=dummy_input['attention_mask']
        )
    print(logits.shape)  # should be torch.Size([8, 3])
    print(f"Model parameters: {sum(p.numel() for p in distilbert_model.parameters()):,}")

    # Hybrid Distilbert Test Block
    print(f"\nHybrid distilBERT Test Block")
    print("="*45)

    # Dummy text (batch of 8)
    dummy_text = [
        "The player scored a goal from outside the box."
    ] * 8

    dummy_input = tokenizer(
        dummy_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Dummy engineered numerical features
    num_numeric_features = 29
    dummy_numeric_features = torch.randn(8, num_numeric_features)

    hybrid_model = HybridDistilbertMomentum(
        num_classes=3,
        num_numeric_features=num_numeric_features,
        dropout=0.3,
        projection_dim=32
    )

    with torch.inference_mode():
        logits = hybrid_model(
            input_ids=dummy_input['input_ids'],
            attention_mask=dummy_input['attention_mask'],
            numeric_features=dummy_numeric_features
        )

    print(logits.shape)  # should be torch.Size([8, 3])
    print(f"Model parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")