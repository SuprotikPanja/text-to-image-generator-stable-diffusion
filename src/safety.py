def validate_prompt(prompt, max_chars=700):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please enter a non-empty prompt.")
    prompt = prompt.strip()
    if len(prompt) > max_chars:
        raise ValueError(f"Prompt must be at most {max_chars} characters.")
    return prompt

def validate_dimensions(width, height, max_side=768):
    width, height = int(width), int(height)
    if min(width, height) < 256 or max(width, height) > max_side:
        raise ValueError(f"Dimensions must be between 256 and {max_side}.")
    if width % 8 or height % 8:
        raise ValueError("Dimensions must be multiples of 8.")
    return width, height
