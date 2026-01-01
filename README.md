# Reproducing EvoPrompt (ICLR 2024)

Reproducible CPU-only EvoPrompt environment (WSL2/Ubuntu) with dependency fixes and GA demo scripts.

This repository documents the process of **reproducing the EvoPrompt framework** proposed in the ICLR 2024 paper:

> **EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers**

The goal of this project is not to propose a new method, but to **reconstruct, execute, debug, and analyze** the original EvoPrompt implementation under realistic and constrained environments, with a strong emphasis on **reproducibility**.

---

## 📌 Project Background

- **Program**: In-service Master’s Program  
- **Department**: Computer Science and Information Engineering  
- **Purpose**: Course project for *Metaheuristics* and preparation for thesis research  
- **Focus**:
  - Reproducing published research
  - Understanding evolutionary prompt optimization
  - Evaluating feasibility under CPU-only environments

This repository serves as an **academic reproduction log** rather than a polished production system.

---

## 📄 Reference Paper

- **Conference**: ICLR 2024  
- **Paper**: EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers  
- **OpenReview**: https://openreview.net/forum?id=ZG3RaNIsO8  
- **Original Code**: https://github.com/beeevita/EvoPrompt  

---

## 🧠 Core Idea of EvoPrompt

EvoPrompt introduces a framework that connects:

- **Large Language Models (LLMs)** as *prompt generators*
- **Evolutionary Algorithms (Metaheuristics)** as *optimization strategies*

Key characteristics:

- Genetic Algorithm (GA) and Differential Evolution (DE)
- Discrete prompt optimization
- No gradient or model fine-tuning required
- Model-agnostic and gradient-free
- Prompts evolve via **crossover, mutation, evaluation, and selection**

---

## 🛠 Environment Setup

### System Environment

- **OS**: Windows 11 Pro
- **Subsystem**: WSL2 (Windows Subsystem for Linux)
- **Linux Distribution**: Ubuntu 22.04
- **Python**: 3.10 (virtual environment)
- **Hardware**: CPU-only (no GPU)

> ⚠️ The original EvoPrompt code assumes GPU + CUDA.  
> This reproduction intentionally removes GPU dependencies to validate CPU-only feasibility.

---

## 🔧 Key Modifications for Reproducibility

During reproduction, several issues were encountered and resolved:

### 1. Dependency Issues
- `requirements.txt` caused installation failures
- GPU / CUDA-related packages removed
- Manual version alignment applied

### 2. Script Adjustments
- Modified execution scripts
- Reduced batch size and memory usage
- Created CPU-friendly execution flows

### 3. Evaluator & Inference Changes
- Disabled GPU usage
- Removed OpenAI API dependency
- Forced **local inference** using:
  - `llama.cpp`
  - Local LLaMA / Alpaca models (e.g., 7B)

---

## 🧪 Experimental Scope

- **Algorithms**:  
  - Genetic Algorithm (GA)  
  - Differential Evolution (DE)

- **Tasks**:
  - Prompt evolution
  - Sentiment classification (e.g., SST-2)
  - Fitness-based prompt selection

- **Execution Mode**:
  - CPU-only
  - Local models
  - Slower but stable inference

---

## 📊 Observations

- EvoPrompt is **reproducible**, but not trivial
- CPU-only execution is:
  - Significantly slower
  - Much more accessible
  - Stable in memory usage
- Evolutionary behaviors (crossover & mutation) are clearly observable
- Engineering effort is as important as algorithm design

---

## 🎓 Academic Value

This reproduction validates that:

- Evolutionary algorithms are effective in **discrete prompt spaces**
- Prompt optimization can be **decoupled from large-scale GPU resources**
- Reproducibility is a critical (and often underestimated) challenge in LLM research

This work also serves as a **baseline** for future thesis research involving:
- NLP
- Prompt optimization
- Knowledge bases
- ERP report analysis

---

## 🚧 Disclaimer

This repository is intended for:

- Academic study
- Course project documentation
- Research reproducibility analysis

It is **not** an official implementation of EvoPrompt and does **not** claim novelty.

---

## 👤 Author

**Frank Hsiao**  
Master’s Student, CSIE (In-service Program)

---

## 📜 License

This project follows the license terms of the original EvoPrompt repository.  
Please refer to the original authors for licensing details.
