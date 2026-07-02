# Multi-Task Sarcasm Detection

A sarcasm detection model that learns sentiment, emotion, and natural language inference alongside sarcasm itself, on the idea that sarcasm is easier to catch once a model understands the signals it usually hides behind.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-red)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-yellow)

## Overview

Sarcasm is one of the harder problems in NLP because the literal words often mean the opposite of the intended message. A model that only ever sees "is this sentence sarcastic, yes or no" tends to miss that nuance. This project takes a different approach: instead of training a model on sarcasm alone, it trains one shared model to simultaneously understand **sentiment**, **emotion**, **natural language inference (NLI)**, and **sarcasm** — the idea being that sarcasm detection improves when the model already has a strong grasp of tone, emotional context, and contradiction.

## How It Works

The model is built on a shared transformer backbone (`microsoft/deberta-v3-base`) with four separate task-specific output heads sitting on top of it:

- **Sarcasm head** — binary classification (sarcastic / not sarcastic)
- **Sentiment head** — 5-class sentiment classification
- **Emotion head** — 28-class emotion classification
- **NLI head** — 3-class entailment / contradiction / neutral classification

All four heads share the same underlying trunk, so anything the model learns while solving one task (say, recognizing an emotional tone shift) feeds into how it represents text for the others — including sarcasm. During training, each step pulls a batch from all four tasks, runs them through the shared trunk, and combines the four resulting losses (weighted toward sarcasm) into a single update.

## Datasets

| Task | Source(s) |
|---|---|
| Sarcasm | News headlines dataset, SemEval-2022 (iSarcasmEval), SemEval-2018 (TweetEval Irony), Reddit sarcasm comments |
| Sentiment | SST-5 |
| Emotion | GoEmotions |
| NLI | MultiNLI + ANLI |

## Training

- Mixed-precision training with gradient checkpointing for memory efficiency
- AdamW optimizer with a linear warmup/decay schedule
- Task losses combined with fixed weighting (sarcasm weighted highest, followed by sentiment, then emotion and NLI)
- Class-weighted loss on the sarcasm head to account for label imbalance

## Evaluation

The model is evaluated on three separate sarcasm benchmarks — iSarcasmEval, the news headlines test set, and TweetEval Irony — chosen specifically because they cover different styles of writing (tweets, headlines, and casual online comments). Testing across all three checks whether the model generalizes across domains rather than just memorizing patterns from a single style of text.

## Tech Stack

- **PyTorch** — model and training loop
- **Hugging Face Transformers & Datasets** — pretrained backbone, tokenization, and dataset loading
- **scikit-learn** — evaluation metrics

## Project Background

This project started as an exploration of whether multi-task learning — training one model to understand several related language tasks at once — could improve on traditional single-task approaches to sarcasm detection.
