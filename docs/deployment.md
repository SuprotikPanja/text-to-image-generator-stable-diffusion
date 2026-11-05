# Deployment

A free public web page is not the same as free GPU inference. Stable Diffusion on CPU can be extremely slow; free GPU availability, ZeroGPU quota, account eligibility and provider rules can change. Never promise permanent free high-speed generation.

1. Push the source to a public GitHub repository. Do not upload raw COCO images, model checkpoints, `.env`, or tokens.
2. Create a Hugging Face Space with Gradio SDK if your account can create the chosen compute Space.
3. Add `app.py`, `src/`, and a `requirements.txt` copied from `requirements-app.txt`.
4. Add Space variables `MODEL_ID` and optionally `LORA_PATH`. Never put tokens in source.
5. Choose eligible GPU/ZeroGPU hardware if available; otherwise CPU generation may be too slow. A paid GPU may be required for a usable always-on demo.
6. Test cold start, model load, generation, queueing, error handling and resource use.

Official docs:
- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/docs/hub/spaces-gpus
- https://huggingface.co/docs/diffusers/en/training/lora

Do not commit the Stable Diffusion base model to GitHub; let the app download/cache it. Verify model and dataset licenses before public deployment.
