import json
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from custom_collate import custom_collate

# Load the dataset from a JSON file
with open('data/my_dataset.json', 'r') as file:
    dataset = json.load(file)

# Define a list of hyperparameters to search
learning_rates = [1e-5, 3e-4, 1e-4]
batch_sizes = [8, 16, 32]
model_architectures = ['facebook/blenderbot-400M-distill']

num_epochs = 10

# Loop through hyperparameter combinations
for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for model_architecture in model_architectures:
            print(f"Training with learning rate={learning_rate}, batch size={batch_size}, and model={model_architecture}")

            # Download and setup the model and tokenizer
            tokenizer = BlenderbotTokenizer.from_pretrained(model_architecture)

            # Tokenize the dataset using the tokenizer
            tokenized_dataset = CustomDataset(tokenizer, dataset)

            # Load the pre-trained model
            model = BlenderbotForConditionalGeneration.from_pretrained(model_architecture)
            
            # Initialize optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(tokenized_dataset) * num_epochs)

            # Create a data loader for batched training data
            data_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

            # Move the model to the appropriate device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Define loss function
            criterion = nn.CrossEntropyLoss()

            # Training loop
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0

                for batch in data_loader:
                    optimizer.zero_grad()

                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    decoder_input_ids = batch["decoder_input_ids"].to(device)

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
                    loss = outputs.loss

                    # Backpropagation and optimization step
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()

                average_loss = total_loss / len(data_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')

            print("Training completed for", epoch + 1, "epochs.")
            
            # Save the fine-tuned model and tokenizer
            save_directory = f'fine_tuned_models/{learning_rate}_{batch_size}_{model_architecture}'
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)

# Making an utterance
utterance = "What's your name?"

# Tokenize the utterance
inputs = tokenizer(utterance, return_tensors="pt", padding=True, truncation=True)

# Generate model results
result = model.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))

generated_response = tokenizer.decode(result[0], skip_special_tokens=True)
print("Generated Response:", generated_response)