def tokenize_and_embed(text, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5"):
    # Task 3: use the checkpoint-matched CLIP tokenizer and text encoder.
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").eval()
    batch = tokenizer(text, padding="max_length", truncation=True,
                      max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.inference_mode():
        output = encoder(**batch).last_hidden_state
    return {
        "tokens": tokenizer.convert_ids_to_tokens(batch["input_ids"][0]),
        "input_ids": batch["input_ids"][0].tolist(),
        "embedding_shape": list(output.shape),
        "embedding_preview": output[0, :4, :8].tolist()
    }
