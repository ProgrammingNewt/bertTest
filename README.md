# BERT Sentiment Analysis Project

## Overview

This project implements a BERT-based sentiment analysis model using PyTorch. The goal is to classify text data into sentiment categories, such as positive, negative, or neutral, by fine-tuning a pre-trained BERT model.

## Dataset

The model is trained on the SMILE Twitter Emotion Dataset, which contains emotion-annotated tweets. The dataset is preprocessed to clean the text and handle class imbalances before training.

## Model

We use the `bert-base-uncased` model from Hugging Face's Transformers library, fine-tuned for sentiment classification. The model is trained to identify sentiments based on the context provided in the tweets.

## Dependencies

- Python 3.8+
- PyTorch with CUDA support
- Transformers (Hugging Face)
- Pandas
- Scikit-learn
- TQDM

## How to Run

1. **Install Dependencies:**
   Install the required packages using conda or pip.

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install transformers pandas scikit-learn tqdm
