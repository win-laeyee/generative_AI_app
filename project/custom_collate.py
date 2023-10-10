import torch

def custom_collate(batch):
    """
    Custom collate function for data batching and padding.

    Args:
        batch (list of dict): List of dictionaries containing tokenized sequences.
        
    Returns:
        dict: A dictionary containing batched and padded sequences.
              - "input_ids": Padded input tokenized IDs.
              - "attention_mask": Padded attention masks for input.
              - "decoder_input_ids": Padded tokenized target (decoder) input IDs.
    """
    # Extract input_ids, attention_masks, and decoder_input_ids from the batch
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    decoder_input_ids = [item["decoder_input_ids"] for item in batch]

    # Pad the sequences to create uniform batched tensors
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "decoder_input_ids": decoder_input_ids,
    }