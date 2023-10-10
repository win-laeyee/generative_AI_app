import torch
import sacrebleu

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, data_loader, tokenizer):
    """
    Evaluate a fine-tuned generative model using BLEU score.

    Args:
        model (torch.nn.Module): The fine-tuned generative model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to encode and decode text.

    Returns:
        float: BLEU score indicating the quality of the model's responses.
    """
    model.eval()
    references = []
    predictions = []

    # Iterate through the evaluation data
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Generate predictions
        with torch.no_grad():
            result = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_beams=5, length_penalty=0.6, early_stopping=True)

        # Decode the model's output to text
        generated_responses = tokenizer.batch_decode(result, skip_special_tokens=True)

        references.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["decoder_input_ids"]])
        predictions.extend(generated_responses)

    # Calculate BLEU score using sacrebleu library
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score