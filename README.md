# Text-de-shuffling
Finetuning GPT2 to descramble sentences.
# Sentence Reconstruction with Finetuned GPT-2

This repository contains the code and resources for finetuning a GPT-2 model to perform sentence reconstruction, converting scrambled sentences back to their original grammatical form using the same words.

## Contents

- `training.py`: The main script for finetuning the GPT-2 model on the sentence reconstruction task.
- `inference.py`: A script for using the finetuned model to reconstruct sentences from scrambled input.
- `notebook.ipynb`: A Jupyter Notebook demonstrating the training and inference process.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Transformers (Hugging Face)
- Numpy
- Pandas

### Installation

1. Clone the repository:

### Training

To train the GPT-2 model on the sentence reconstruction task, run the `training.py` script:

This script will run training in a modal nvidia A10 GPU.
Run:
```
modal run training.py
```

1. Load the GPT-2 model and tokenizer.
2. Prepare the training data by scrambling sentences and creating input-output pairs.
3. Finetune the GPT-2 model on the sentence reconstruction task.
4. Save the finetuned model.

### Inference

To use the finetuned model for sentence reconstruction, run the `inference.py` script:

The script will output the reconstructed sentence.

### Jupyter Notebook

The `notebook.ipynb` Jupyter Notebook provides a step-by-step guide through the training and inference process. You can run the notebook to explore the code and the model's performance.

## Results

The finetuned GPT-2 model achieved an accuracy of _[insert accuracy metric]_ on the sentence reconstruction task. The model was able to handle a variety of sentence structures and word arrangements, but struggled with some more complex or unusual sentence constructions.

## Future Work

- Explore alternative model architectures or training approaches to improve performance.
- Expand the dataset to include more diverse sentence structures.
- Integrate the model into a larger application or service for practical use.
