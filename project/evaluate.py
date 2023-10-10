import json
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from custom_collate import custom_collate
from evaluate_model import evaluate_model
from sklearn.model_selection import KFold
import os
import random

# Load the dataset
with open('data/my_dataset.json', 'r') as file:
    my_dataset = json.load(file)

# Specify the directory where the fine-tuned models are stored
models_directory = 'fine_tuned_models'

# Shuffle the dataset
random.shuffle(my_dataset)

# Define the number of folds for cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

best_bleu_score = -1.0
best_hyperparameters = None

best_model_directory = 'best_model'

# List all directories in the models_directory
model_names = [name for name in os.listdir(models_directory) if not name.startswith('.')]

# Iterate through models in the fine_tuned_models directory
for model_name in model_names:
    # Construct the model path
    model_path = os.path.join(models_directory, model_name, "blenderbot-400M-distill")
    print(f"File path: {model_path}")

    # Load the tokenizer and model for evaluation
    tokenizer = BlenderbotTokenizer.from_pretrained(model_path)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_path)

    # Get the learning rate, batch size and model architecture 
    parts = model_name.split("_")

    learning_rate_str = parts[0]
    batch_size_str = parts[1]
    model_architecture = parts[2]

    learning_rate = float(learning_rate_str)
    batch_size = int(batch_size_str)

    # Initialize a list to store the BLEU scores for each fold
    fold_bleu_scores = []

    # Iterate through the folds
    for train_idx, val_idx in kf.split(my_dataset):
        train_dataset = [my_dataset[i] for i in train_idx]
        val_dataset = [my_dataset[i] for i in val_idx]

        # Prepare the training and validation data loaders
        tokenized_train_dataset = CustomDataset(tokenizer, train_dataset)
        tokenized_val_dataset = CustomDataset(tokenizer, val_dataset)

        train_data_loader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        val_data_loader = DataLoader(tokenized_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        # Evaluate the model on the validation fold using the evaluation function
        bleu_score = evaluate_model(model, val_data_loader, tokenizer)

        fold_bleu_scores.append(bleu_score)
    
    # Calculate the average BLEU score across folds
    avg_bleu_score = sum(fold_bleu_scores) / num_folds

    print(f"Model: {model_name}")
    print(f"Average BLEU Score: {avg_bleu_score:.4f}")

    # Check if this configuration has the best BLEU score
    if avg_bleu_score > best_bleu_score:
        best_bleu_score = avg_bleu_score
        best_hyperparameters = {
            "model_architecture": model_architecture,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "avg_bleu_score": avg_bleu_score
        }

        # Save the best model
        model.save_pretrained(best_model_directory)
        tokenizer.save_pretrained(best_model_directory)

print("Best hyperparameters:")
print(f"Model Name: {best_hyperparameters['model_architecture']}")
print(f"Batch Size: {best_hyperparameters['batch_size']}")
print(f"Learning rate: {best_hyperparameters['learning_rate']}")
print(f"Best Average BLEU Score: {best_hyperparameters['avg_bleu_score']:.4f}")
