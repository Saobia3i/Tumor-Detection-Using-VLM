# Tumor-Detection-Using-VLM

Small personal practice project for thesis work.

## Goal
- **Input:** Brain MRI image + question
- **Output:** AI answer about the image
- **Model:** Qwen2-VL-2B (fine-tuned with QLoRA)
- **UI:** Gradio
- **Training Target:** Kaggle free GPU
- **IDE:** Antigravity (local)

## Project Files
- `/home/runner/work/Tumor-Detection-Using-VLM/Tumor-Detection-Using-VLM/app.py` – Gradio inference app for MRI question-answering.
- `/home/runner/work/Tumor-Detection-Using-VLM/Tumor-Detection-Using-VLM/train_qlora.py` – Kaggle-oriented QLoRA fine-tuning scaffold.
- `/home/runner/work/Tumor-Detection-Using-VLM/Tumor-Detection-Using-VLM/requirements.txt` – Python dependencies.

## Quick Start (UI)
```bash
pip install -r requirements.txt
python app.py --mock
```

Then open `http://127.0.0.1:7860`.

> `--mock` is for local wiring checks without downloading large model weights.

To run real inference with your adapter:
```bash
python app.py --model-id Qwen/Qwen2-VL-2B-Instruct --adapter-path /path/to/qlora-adapter
```

## Kaggle QLoRA Training (Starter)
Example command on Kaggle Notebook:
```bash
pip install -r requirements.txt
python train_qlora.py --dataset-name <your_dataset_name>
```

The training script saves adapter outputs to `/kaggle/working/qwen2vl-qlora` by default.

## Suggested Dataset Schema
For best compatibility with this starter, keep your processed training split with a `text` field that contains instruction-style samples for the VLM workflow.
