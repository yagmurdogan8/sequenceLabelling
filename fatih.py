from pathlib import Path
import re
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from transformers import (Trainer, TrainingArguments, DistilBertForTokenClassification,
                          EarlyStoppingCallback, DistilBertTokenizerFast)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score


def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


texts, tags = read_wnut('data/wnut17train.conll')

train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

unique_tags = set([item for doc in tags for item in doc])
tag2id = {item: idx for idx, item in enumerate(unique_tags)}
id2tag = {idx: item for item, idx in tag2id.items()}


class WNUTDatasetBatchLoader(torch.utils.data.Dataset):
    def __init__(self, texts, tags, tokenizer):
        self.texts = np.asanyarray(texts, dtype=list)
        self.tags = np.asanyarray(tags, dtype=list)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        encodings = self.tokenizer(self.texts[idx],
                                   is_split_into_words=True,
                                   max_length=64,
                                   padding='max_length',
                                   truncation=True)
        tags = self.tags[idx]
        labels = align_labels(tags, encodings)

        item = dict()
        item['input_ids'] = torch.tensor(encodings.input_ids)
        item['attention_mask'] = torch.tensor(encodings.attention_mask)
        item['labels'] = torch.tensor(labels)

        return item


def align_labels(tags: list, encodings: transformers.tokenization_utils_base.BatchEncoding,
                 label_all_tokens=True) -> list:
    labels = []
    word_ids = encodings.word_ids()
    prev_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != prev_word_idx:
            label_ids.append(tag2id[tags[word_idx]])
        else:
            label_ids.append(tag2id[tags[word_idx]] if label_all_tokens else -100)
    return label_ids


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

train_dataset = WNUTDatasetBatchLoader(train_texts, train_tags, tokenizer)
val_dataset = WNUTDatasetBatchLoader(val_texts, val_tags, tokenizer)

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))
cb_early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-3)

training_args = TrainingArguments(
    output_dir='./results',  # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=100,
    load_best_model_at_end=True,
    evaluation_strategy='steps',
    save_total_limit=3,
)


def compute_metrics(eval_predictions):
    predictions, labels = eval_predictions.predictions, eval_predictions.label_ids
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    score_f1 = f1_score(true_labels, true_predictions)
    score_prec = precision_score(true_labels, true_predictions)
    score_rec = recall_score(true_labels, true_predictions)
    score_acc = accuracy_score(true_labels, true_predictions)

    return {
        "precision": score_prec,
        "recall": score_rec,
        "f1": score_f1,
        "accuracy": score_acc,
    }

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.add_callback(cb_early_stop)

trainer.train()
