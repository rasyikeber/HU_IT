# train_model.py
import os
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your project directory
project_dir = 'stud'

# Define the path to save the fine-tuned model
output_model_path = os.path.join(project_dir, 'fine_tuned_model')

# Make sure the directory exists
os.makedirs(output_model_path, exist_ok=True)

# Verify GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Load the STS benchmark dataset
train_dataset = load_dataset('stsb_multi_mt', 'en', split='train')
dev_dataset = load_dataset('stsb_multi_mt', 'en', split='dev')
test_dataset = load_dataset('stsb_multi_mt', 'en', split='test')

# Initialize a pre-trained BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')
model.to(device)  # Move the model to GPU if available

# Convert datasets to InputExample format
def convert_to_input_examples(dataset):
    examples = []
    for row in dataset:
        examples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=row['similarity_score'] / 5.0))
    return examples

train_examples = convert_to_input_examples(train_dataset)
dev_examples = convert_to_input_examples(dev_dataset)
test_examples = convert_to_input_examples(test_dataset)

train_dataset = SentencesDataset(train_examples, model)
dev_dataset = SentencesDataset(dev_examples, model)
test_dataset = SentencesDataset(test_examples, model)

# Prepare DataLoaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=8)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)

# Extract sentences and scores from the dev dataset for evaluation
sentences1 = [example.texts[0] for example in dev_examples]
sentences2 = [example.texts[1] for example in dev_examples]
scores = [example.label for example in dev_examples]

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model=model)

# Fine-tune the model
num_epochs = 4
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

# Initialize the evaluator
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# Start training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=output_model_path,
    optimizer_params={'lr': 2e-5},
    show_progress_bar=True,
    use_amp=True  # Use automatic mixed precision for faster training
)

# Save the model to the specified path
model.save(output_model_path)
logger.info(f"Model saved to {output_model_path}")
