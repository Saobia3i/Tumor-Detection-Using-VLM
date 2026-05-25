import argparse
import os
from typing import Optional

import gradio as gr


class BrainMRIVLMAssistant:
    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.use_mock = use_mock
        self.model = None
        self.processor = None

        if not self.use_mock:
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Missing model dependencies. Install requirements.txt or run with --mock."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.adapter_path:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            self.model = base_model

        self.model.eval()

    def answer(self, image, question: str) -> str:
        question = (question or "").strip()

        if image is None:
            return "Please upload a brain MRI image first."

        if not question:
            return "Please enter a question about the MRI image."

        if self.use_mock:
            return (
                "[Mock Mode] The app is wired for Qwen2-VL-2B + QLoRA adapter inference. "
                "Set --mock off and provide real model/adapters to run actual predictions."
            )

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_device = getattr(self.model, "device", "cpu")
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model_device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        trimmed_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()


def build_interface(assistant: BrainMRIVLMAssistant) -> gr.Interface:
    def _predict(image, question):
        try:
            return assistant.answer(image, question)
        except Exception as exc:  # Surface runtime issues in UI
            return f"Inference error: {exc}"

    return gr.Interface(
        fn=_predict,
        inputs=[
            gr.Image(type="pil", label="Brain MRI Image"),
            gr.Textbox(label="Question", placeholder="What abnormalities are visible?"),
        ],
        outputs=gr.Textbox(label="AI Answer"),
        title="Brain MRI Q&A with Qwen2-VL-2B",
        description=(
            "Upload a brain MRI image and ask a question. "
            "The app is designed for Qwen2-VL-2B fine-tuned with QLoRA."
        ),
        flagging_mode="never",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain MRI VLM Q&A app")
    parser.add_argument(
        "--model-id",
        default=os.getenv("VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct"),
        help="Base model ID for inference",
    )
    parser.add_argument(
        "--adapter-path",
        default=os.getenv("VLM_ADAPTER_PATH"),
        help="Path to fine-tuned QLoRA adapter",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run UI without loading model weights",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable public Gradio share link",
    )
    args = parser.parse_args()

    assistant = BrainMRIVLMAssistant(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        use_mock=args.mock,
    )

    demo = build_interface(assistant)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)


if __name__ == "__main__":
    main()
