
# Multimodal Geometry Problem Solver-COMPUTER VISION TO APPLICATION

This repository contains a **multimodal AI model** designed to solve elementary geometry problems by combining **computer vision**, **OCR**, **natural language processing (NLP)**, and **deep learning** techniques. The pipeline processes images of geometric diagrams and problem statements to predict correct answers.

---

### Overview

The pipeline consists of the following layers:

#### 1. Image Processing Layer
- Detects geometric shapes and structures (polygons, lines, angles) in diagrams.  
- Uses a **custom EfficientNet-B4 Adapter** with **LoRA fine-tuning**.  
- Extracts convolutional features and applies adaptive pooling.

#### 2. OCR-based Text Detection
- Extracts embedded text, such as angle labels and vertices, using **EasyOCR**.

#### 3. Text Processing Layer
- Processes problem statements as a sequence-to-sequence NLP task.  
- Encodes text with **BERT (bert-base-uncased)** fine-tuned via LoRA (query and value layers).  
- Converts textual information into embeddings correlated with visual features.

#### 4. Fusion / Multi-Head Attention Layer
- Projects image and text embeddings into a shared **512-dimensional space**.  
- Applies **multi-head attention** (image features as queries, text as keys/values).  
- Produces a **2048-dimensional context-aware fused representation** for downstream tasks.

#### 5. Reasoning Model Layer
- Utilizes **Phi-1.5**, a lightweight 1.3B parameter LLM, fine-tuned via LoRA (q_proj & k_proj layers).  
- Optimized for **resource-efficient training** (GPU-friendly, e.g., T4 GPUs).  
- Trained on high-quality synthetic "textbook-like" data (science and math).

#### 6. Decoder Layer
- Uses **T5-small** to decode fused embeddings into predicted answers (A/B/C/D).  
- Trained with **teacher forcing** for supervised sequence generation.

---

### Training Details
- **Backpropagation** is applied end-to-end through the multimodal pipeline.  
- **Mixed Precision Training** with `autocast` reduces memory usage.  
- **GradScaler** prevents gradient underflow during backpropagation.  
- **Validation** performed after each epoch; best model saved as `best_model.pth`.  
- **Optimizer and Learning Rate**: Fusion layer updated with a learning rate of `3e-5`.

---

### Key Features
- **Efficient feature extraction** with LoRA-adapted EfficientNet-B4 and BERT.  
- **Context-aware fusion** of visual and textual data.  
- **Lightweight reasoning** with Phi-1.5 for geometry-specific tasks.  
- **GPU-efficient training** suitable for limited hardware resources (e.g., Colab T4).  
- **End-to-end multimodal learning** from images + text to final answer predictions.  

---

### Requirements
```bash
torch>=2.0
torchvision
transformers
peft
efficientnet-pytorch
easyocr
tqdm
