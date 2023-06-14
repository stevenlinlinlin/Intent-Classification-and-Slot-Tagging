# Intent Classification and Slot Tagging
This is a PyTorch implementation of intent classification and [slot tagging](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)).

# Dataset format
## Intent Classification
```
{
    "text": "i need you to book me a flight from ft lauderdale to houston on southwest",
    "intent": "book_flight",
    "id": "train-0"
  },
```
## Slot Tagging
```
{
    "tokens": [
      "i",
      "have",
      "three",
      "people",
      "for",
      "august",
      "seventh"
    ],
    "tags": [
      "O",
      "O",
      "B-people",
      "I-people",
      "O",
      "B-date",
      "O"
    ],
    "id": "train-0"
  },
```

# Installation
## package requirements
```bash
pip install -r requirements.txt
```
## download glove.840B.300d.txt
- download [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) and put it in the data directory

# Intent Classification
## Preprocessing
Use glove.840B.300d.txt for preprocessing
```bash
python preprocess_intent.py \
    --data_dir [data directory] \
    --glove_path [glove.840B.300d.txt path] \
    --output_dir [output directory] \
```
## Train
```bash
python train_intent.py \
    --data_dir [data directory] \
    --cache_dir [cache directory] \
    --ckpt_dir [checkpoint directory] \
```
## Test
```bash
python test_intent.py \
    --test_file [test json file] \
    --ckpt_path [best weight from train] \
    --pred_file [pred csv file] \
```

# Slot Tagging
## Preprocessing
Use glove.840B.300d.txt for preprocessing
```bash
python preprocess_intent.py \
    --data_dir [data directory] \
    --glove_path [glove.840B.300d.txt path] \
    --output_dir [output directory] \
```
## Train
```bash
python train_slot.py \
    --data_dir [data directory] \
    --cache_dir [cache directory] \
    --ckpt_dir [checkpoint directory] \
```
## Test
```bash
python test_slot.py \
    --test_file [test json file] \
    --ckpt_path [best weight] \
    --pred_file [pred csv file]
```