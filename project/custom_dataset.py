from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data):
        """
        Custom dataset for training a chatbot model.
        
        Args:
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text data.
            data (list of dict): List of dictionaries containing question and answer pairs.
        """
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: A dictionary containing input and target tokenized sequences.
                  - "input_ids": Tokenized input IDs.
                  - "attention_mask": Attention mask for input.
                  - "decoder_input_ids": Tokenized target (decoder) input IDs.
        """
        input_text = self.data[idx]["question"]
        target_text = self.data[idx]["answer"]

        input_tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        target_tokens = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)

        return {
            "input_ids": input_tokens.input_ids[0],
            "attention_mask": input_tokens.attention_mask[0],
            "decoder_input_ids": target_tokens.input_ids[0],
        }