# W8A16Linear_Quantizer
Custom Quantizer

# 🧮 W8A16Linear Quantizer for Mistral-7B

This project demonstrates a custom quantization technique for Hugging Face transformer models, specifically replacing `nn.Linear` layers in **Mistral-7B** with an efficient `W8A16LinearLayer` that uses **int8 weights** and **fp16 activations**.

---

## 🔍 Overview

The `W8A16Linear_Quantizer.ipynb` notebook:
- Implements a custom linear layer using INT8 weights and FP16 activations
- Recursively replaces applicable `nn.Linear` modules in the model
- Benchmarks performance impact and fidelity vs. baseline
- Works directly with models like `mistralai/Mistral-7B-Instruct-v0.2` and `mistralai/Mistral-7B-Instruct-v0.3`

---

## 🎯 Why W8A16?

- **W8 (int8 weights)** reduces model size and memory bandwidth
- **A16 (fp16 activations)** maintains output precision
- Useful for **low-latency inference** on consumer GPUs
- Keeps **output quality close to baseline** while improving speed

---

## 🛠 Features

- ✅ Drop-in layer replacement for Mistral's attention and feed-forward blocks
- ✅ Uses `torch.nn.Module` hooks to safely swap layers
- ✅ Benchmarks: inference time, generation latency, output equivalency
- ✅ Uses Hugging Face Hub for model loading and integration
- ✅ Tokenizer preservation via `AutoTokenizer`

---

## 📗 Notebook Structure

1. **Layer Design**: `W8A16LinearLayer` with custom `forward()`
2. **Traversal & Replacement**: recursive model walker
3. **Tokenization Setup**: Hugging Face `AutoTokenizer`
4. **Model Benchmarking**: output length, tokens/sec, VRAM profile
5. **Comparative Generation**: check output equivalence vs baseline

---

## 🗂 Files

```
.
├── W8A16Linear_Quantizer.ipynb    # Main notebook
```

---

## ⚙️ Requirements

Install compatible packages:

```bash
pip install torch transformers accelerate
```

Recommended: CUDA-enabled PyTorch build.

---

## 🚀 Run the Notebook

```bash
jupyter notebook W8A16Linear_Quantizer.ipynb
```

---

## 📌 Supported Models

- [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) — *Apache 2.0 License*
- [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) — *Apache 2.0 License*

You can modify the default model in the notebook:
```python
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
```

---

## 📝 License

This project is licensed under the Apache 2.0 License.

---

> Created by Patrick Hill — AI Developer, Quantization Engineer, and Performance Optimizer  
> [LinkedIn](https://www.linkedin.com/in/patrick-hill-4b9807178/)
