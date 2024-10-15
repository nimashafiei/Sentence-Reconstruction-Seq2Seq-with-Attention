# Sentence Reconstruction Using Seq2Seq with Attention

## Overview
This project focuses on reconstructing shuffled English sentences back into their original order using a sequence-to-sequence (seq2seq) model with an attention mechanism. The input is a sequence of words in a random order, and the goal is to generate the correct sentence by reconstructing the original word sequence.

### Constraints:
- No pretrained models are used.
- The neural network model has less than 20M parameters.
- The dataset is sourced from a snapshot of Wikipedia with the vocabulary restricted to the 10K most frequent words.

## Dataset
The dataset consists of a processed Wikipedia snapshot:
- Sentences are restricted to using the 10K most frequent words.
- Sentence lengths are limited to between 3 and 30 words.
- 70% of the data is used for training, and 30% is reserved for testing.

## Model Architecture
We utilize a seq2seq model architecture with an attention mechanism to better capture context from the input sentence:
- **Encoder**: Three layers of LSTM are used to process the input sequences.
- **Decoder**: A single LSTM layer combined with an attention mechanism helps focus on specific parts of the input sequence.
- **Attention**: The attention mechanism enhances the performance, especially for longer sequences.
- **Embedding Layer**: Converts input sequences into dense vector representations.

## Model Training
- **Optimizer**: RMSprop with a learning rate of 0.001.
- **Loss**: `sparse_categorical_crossentropy` is used, which is suitable for multi-class classification.
- **Training Setup**:
  - 30 epochs
  - Batch size of 64
  - ModelCheckpoint is used to save the best-performing model based on validation loss.

## Evaluation
The evaluation metric used is based on the longest common substring between the predicted and original sentences:
- **Metric**: For each test case, the longest common substring between the predicted and actual sentence is computed, and the length of this substring is divided by the length of the original sentence.
- **Final Performance**: The model achieved an average score of **0.6451** on the test set.

## Usage

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Keras
- Additional dependencies:
  - `pip install -r requirements.txt`

### Dataset Preparation
Run the script to download and preprocess the dataset from Wikipedia:
```bash
python prepare_data.py
