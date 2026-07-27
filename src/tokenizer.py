# To save the tokenizer locally.

from transformers import AutoTokenizer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

save_path = PROJECT_ROOT / 'models' / 'tokenizer'
save_path.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(save_path)

print(f"Tokenizer saved to {save_path}")
print(list(save_path.glob('*')))