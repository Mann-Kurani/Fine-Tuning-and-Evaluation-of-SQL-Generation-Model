# Fine-Tuning and Evaluation of SQL Generation Model

This repository contains a project for fine-tuning and evaluating a language model aimed at generating SQL queries from natural language prompts. The primary components of the project include a script for fine-tuning (`finetuner.py`) and a script for evaluation (`phx.py`).

## Overview

The project utilizes the `transformers` library for model handling and training, and `peft` for parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation). The model is trained on a synthetic text-to-SQL dataset, and its performance is evaluated using BLEU and exact match metrics.

## Fine-Tuning the Model

The `finetuner.py` script is responsible for fine-tuning the pre-trained language model on the text-to-SQL dataset. This involves the following steps:

- Loading and preparing the dataset.
- Initializing the model and tokenizer.
- Configuring training arguments and data collators.
- Training the model with LoRA and saving the trained model.

To run the fine-tuning script, execute:
```bash
python finetuner.py
```

## Evaluating the Model

The `phx.py` script evaluates the fine-tuned model's performance. It leverages the Phoenix tool for tracing and visualization and computes metrics such as BLEU and exact match scores.

To run the evaluation script, execute:
```bash
python phx.py
```

## Results

The evaluation script outputs the performance metrics of the model. Example results might look like this:

- **BLEU Score**: 61.16%
- **Exact Match**: 34.00%
- **Sentence-Level BLEU Score**: 74.90%

These metrics help in assessing the quality and accuracy of the SQL generation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
