
import json
import gradio as gr

from src.generation import engine
from src.text import tokenize_and_embed
from src.config import settings
from src.gan.inference import generate_gan_samples


def generate(
    prompt, negative, width, height, steps,
    guidance, seed, use_lora, lora_strength
):
    try:
        image, meta = engine.generate(
            prompt, negative, width, height,
            steps, guidance, seed,
            use_lora, lora_strength
        )
        return image, json.dumps(meta, indent=2)

    except Exception as e:
        return None, json.dumps({"error": str(e)}, indent=2)


def inspect(prompt):
    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    try:
        return json.dumps(
            tokenize_and_embed(prompt, settings.model_id),
            indent=2
        )
    except Exception as e:
        return f"Text encoder error: {e}"


def generate_gan(model, shape, count, seed):
    try:
        image, metadata = generate_gan_samples(
            model, shape, count, seed
        )
        return image, json.dumps(metadata, indent=2)

    except Exception as e:
        return None, json.dumps({"error": str(e)}, indent=2)


def compare_gans(shape, count, seed):
    try:
        baseline, baseline_meta = generate_gan_samples(
            "Baseline CGAN", shape, count, seed
        )

        attention, attention_meta = generate_gan_samples(
            "Attention CGAN", shape, count, seed
        )

        details = {
            "baseline": baseline_meta,
            "attention": attention_meta,
            "comparison_note": (
                "Both models use the same shape, sample count "
                "and seed for a controlled qualitative comparison."
            )
        }

        return (
            baseline,
            attention,
            json.dumps(details, indent=2)
        )

    except Exception as e:
        return None, None, json.dumps({"error": str(e)}, indent=2)


with gr.Blocks(
    title="Text-to-Image Studio",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # Text-to-Image Studio

        An experimental generative AI application combining
        Stable Diffusion, LoRA, CLIP and Conditional GANs.
        """
    )

    with gr.Tab("Image Generation"):

        with gr.Row():
            with gr.Column():

                prompt = gr.Textbox(
                    label="Image prompt",
                    lines=3
                )

                negative = gr.Textbox(
                    label="Negative prompt",
                    value="blurry, low quality, distorted"
                )

                width = gr.Slider(
                    256, settings.max_image_side, 512,
                    step=64, label="Width"
                )

                height = gr.Slider(
                    256, settings.max_image_side, 512,
                    step=64, label="Height"
                )

                steps = gr.Slider(
                    10, 35, settings.default_steps,
                    step=1, label="Inference steps"
                )

                guidance = gr.Slider(
                    1, 12, 7,
                    step=0.5, label="Guidance scale"
                )

                seed = gr.Number(
                    value=-1, precision=0,
                    label="Seed (-1 = random)"
                )

                use_lora = gr.Checkbox(
                    value=False,
                    label="Enable COCO-trained LoRA"
                )

                lora_strength = gr.Slider(
                    0, 1, 0.75,
                    step=0.05, label="LoRA strength"
                )

                button = gr.Button(
                    "Generate Image", variant="primary"
                )

            with gr.Column():

                image = gr.Image(
                    label="Generated image", type="pil"
                )

                metadata = gr.Code(language="json")

        button.click(
            generate,
            inputs=[
                prompt, negative, width, height,
                steps, guidance, seed,
                use_lora, lora_strength
            ],
            outputs=[image, metadata]
        )

    with gr.Tab("CLIP Text Processing"):

        gr.Markdown(
            "Task 3 — CLIP tokenization and text embeddings."
        )

        text_input = gr.Textbox(
            label="Enter a text prompt", lines=3
        )

        inspect_button = gr.Button(
            "Tokenize and inspect embedding"
        )

        text_result = gr.Code(language="json")

        inspect_button.click(
            inspect,
            inputs=text_input,
            outputs=text_result
        )

    with gr.Tab("GAN Laboratory"):

        gr.Markdown(
            """
            ## Conditional GAN Experiments

            Generate geometric shapes using the trained
            Conditional GAN and Self-Attention GAN checkpoints.

            These are separate class-conditioned experiments,
            not natural-language image generation models.
            """
        )

        with gr.Row():

            with gr.Column():

                gan_model = gr.Dropdown(
                    choices=[
                        "Baseline CGAN",
                        "Attention CGAN"
                    ],
                    value="Baseline CGAN",
                    label="GAN architecture"
                )

                gan_shape = gr.Dropdown(
                    choices=["Square", "Circle", "Triangle"],
                    value="Circle",
                    label="Shape category"
                )

                gan_count = gr.Slider(
                    1, 16, 8, step=1,
                    label="Number of samples"
                )

                gan_seed = gr.Number(
                    value=42, precision=0,
                    label="Random seed"
                )

                gan_button = gr.Button(
                    "Generate GAN Samples",
                    variant="primary"
                )

            with gr.Column():

                gan_image = gr.Image(
                    label="GAN generated samples",
                    type="pil"
                )

                gan_metadata = gr.Code(
                    language="json"
                )

        gan_button.click(
            generate_gan,
            inputs=[
                gan_model, gan_shape,
                gan_count, gan_seed
            ],
            outputs=[gan_image, gan_metadata]
        )

        gr.Markdown("### Baseline versus Attention GAN")

        with gr.Row():

            compare_shape = gr.Dropdown(
                choices=["Square", "Circle", "Triangle"],
                value="Circle",
                label="Comparison shape"
            )

            compare_count = gr.Slider(
                1, 16, 8, step=1,
                label="Samples per model"
            )

            compare_seed = gr.Number(
                value=42, precision=0,
                label="Comparison seed"
            )

        compare_button = gr.Button(
            "Compare Both GAN Models"
        )

        with gr.Row():

            baseline_image = gr.Image(
                label="Baseline CGAN"
            )

            attention_image = gr.Image(
                label="Self-Attention CGAN"
            )

        compare_metadata = gr.Code(language="json")

        compare_button.click(
            compare_gans,
            inputs=[
                compare_shape,
                compare_count,
                compare_seed
            ],
            outputs=[
                baseline_image,
                attention_image,
                compare_metadata
            ]
        )

    with gr.Tab("Project Experiments"):

        gr.Markdown(
            """
            ### Task 1 — Stable Diffusion LoRA fine-tuning
            Pretrained Stable Diffusion 1.5 with an optional
            COCO-trained LoRA adapter.

            ### Task 2 — Conditional GAN
            Class-conditioned generation of geometric shapes.

            ### Task 3 — Text preprocessing
            CLIP tokenization, attention masks and embeddings.

            ### Task 4 — Dataset analysis
            COCO image and caption statistics.

            ### Task 5 — Self-Attention GAN
            Comparison of baseline and attention-based GANs.

            ### Task 6 — Integrated application
            Stable Diffusion, optional LoRA, CLIP and GAN modules.
            """
        )


if __name__ == "__main__":
    demo.queue(max_size=4).launch(
        server_name=settings.host,
        server_port=settings.port,
        show_error=True
    )
