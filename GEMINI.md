# Gemini Context: Deep Learning Study Project

This repository is a structured learning environment for Deep Learning, combining theoretical studies from Stanford's CS231n and practical, from-scratch implementations.

## Project Structure & Local Mandates

This project uses hierarchical `GEMINI.md` files to provide context-specific instructions. Always refer to local instructions in subdirectories:

### 1. CS231n Theory (`/cs231`)
- **Lecture Summaries (`/cs231/lec_summary`)**: Core study notes from the course.
  - *See `cs231/lec_summary/GEMINI.md` for writing standards and mathematical guidelines.*
- **Deep-Dive AI Summaries (`/cs231/ai_summay`)**: Detailed analysis of the 2017 lecture videos.
  - Follow the rules in `role.md` for technical term explanations and context supplements.
- **Visuals (`/cs231/images`)**: Key diagrams and screenshots used across all notes.

### 2. Deep Learning from Scratch (`/DL_scratch`)
- **Implementations (`/DL_scratch/source`)**: Fundamental DL components built with NumPy.
  - *See `DL_scratch/source/GEMINI.md` for coding standards, modularity rules, and numerical validation procedures.*
- **Datasets (`/DL_scratch/dataset`)**: MNIST and other datasets used for training scripts.

## Core Environment

- **Primary Stack**: Python 3.x, NumPy, Matplotlib, Jupyter Notebook.
- **Dependencies**: Managed via `.venv` and `requirements.txt`.
- **Workflow**:
  1.  Review theory in `cs231/lec_summary/`.
  2.  Study implementation details in `DL_scratch/source/` notebooks.
  3.  Execute `train_neuralnet.py` for model verification.

## Universal Directives

1.  **Language**: Maintain Korean (mostly) for study notes and English for code comments.
2.  **No High-Level Frameworks**: Unless specifically asked, all code implementations should remain "from scratch" using NumPy to reinforce mathematical understanding.
3.  **Cross-Referencing**: When writing notes, always cross-reference relevant source code in `DL_scratch/source/` to bridge theory and practice.
