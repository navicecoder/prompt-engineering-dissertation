# Prompt Engineering in Large Language Models
My MSc dissertation project investigating how different prompting techniques influence the accuracy and response time of Large Language Models (LLMs) on mathematical reasoning tasks.

## Prompting Techniques

Six prompting strategies were implemented and evaluated:

- Zero-Shot Prompting
- Few-Shot Prompting
- Chain-of-Thought (CoT)
- Role Prompting
- Self-Consistency
- Tree of Thought (ToT)

## Technologies Used

- Python
- Qwen3:32B
- Hugging Face Datasets
- REST APIs
- Regular Expressions
- CSV-based result logging

## Dataset

The experiments used the GSM8K mathematical reasoning dataset from Hugging Face, specifically its test split. The first 1,000 questions were evaluated using each prompting strategy.

## Implementation

A Python-based pipeline was developed to:
- Send prompts to an LLM through API requests.
- Extract numerical predictions and ground-truth answers.
- Evaluate answer correctness.
- Measure response time.
- Record experimental results and incorrect predictions.

Self-Consistency generated five candidate responses and used majority voting to select an answer.

Tree of Thought generated three candidate solution approaches, evaluated them, selected the best approach and generated a final answer.

## Experimental Results

| Prompting Strategy | Accuracy |
|--------------------|----------|
| Zero-Shot          | 96.3%    |
| Few-Shot           | 96.7%    |
| Chain-of-Thought   | 96.5%    |
| Role Prompting     | 96.5%    |
| Self-Consistency   | 95.5%    |
| Tree of Thought    | 95.9%    |

Few-Shot Prompting achieved the highest accuracy in the experiment.

## Project Structure

- `main.py` – Zero-Shot, Few-Shot, CoT and Role Prompting implementations.
- `prompts.py` – Prompt templates.
- `selfconsistency.py` – Self-Consistency implementation.
- `ToT.py` – Tree of Thought implementation.
- `results/results.csv` – Recorded experimental results.
- `results/gsm8k_errors.txt` – Incorrect predictions and corresponding model responses.

## Setup

Install the required dependencies:

>pip install -r requirements.txt

Configure the LLM API endpoint and API key using environment variables. See `.env.example` for the required configuration variables.

The scripts use question index ranges configured within the Python files. Update these ranges when selecting which questions to process.

## Notes
This repository contains the Python implementation and experimental outputs from the MSc dissertation. The reported results reflect the original experimental setup and may vary with different models, API configurations or settings.