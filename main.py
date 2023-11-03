from transformers import AutoTokenizer
import tensorflow as tf
from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, pipeline)
import evaluate
import numpy as np

train_data = []
with open("data/wnut17train.conll", "r", encoding='utf-8') as train_file:
    for line in train_file:
        line = line.strip().split()
        if line:
            tokens, label = line[0], line[1]
            train_data.append({"tokens": tokens, "labels": label})

# read train data
train = open("data/wnut17train.conll", "r", encoding='utf-8')
print(train.read())
#
dev = open("data/emerging.dev.conll", "r", encoding='utf-8')
print(dev.read())
#
# test = open("data/emerging.test.annotated", "r", encoding='utf-8')
# # print(test.read())

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Tokenize and prepare the data
def tokenize_and_preprocess_data(example):
    tokens = example['tokens']
    labels = example['labels']

    # Tokenize the text
    tokenized_input = tokenizer(tokens, is_split_into_words=True)

    # Ensure that the tokenized input matches the token boundaries
    assert len(tokenized_input['input_ids']) == len(tokens)

    return {
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'labels': labels,
    }


# Apply the preprocessing to the dataset
train_data = [tokenize_and_preprocess_data(example) for example in train_data]

# for line in train:
#     parts = line.strip().split(' ')
#
#     tokens = parts[0]
#     labels = parts[1]
#
#     # dictionary storing sentences
#     sentence_data = {
#         "tokens": tokens,
#         "labels": labels
#     }
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

train = open("data/wnut17train.conll", "r", encoding='utf-8')

# Initialize your tokenizer and model
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

train_list = []

for line in train:
    line = line.strip().split()
    if line:
        tokens, label = line[0], line[1]
        train_list.append({"tokens": tokens, "labels": label})

# Tokenize the sequences and prepare them for the model
batch = tokenizer(train_list, padding=True, truncation=True, return_tensors="tf")

# This is new
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Define your labels
labels = tf.convert_to_tensor([1, 1])  # Replace with your actual labels

# Train on the batch
model.train_on_batch(batch, labels)


raw_datasets = load_dataset("wnut_17")
print(raw_datasets)

print(raw_datasets["train"][0]["tokens"])
print(raw_datasets["train"][0]["ner_tags"])

ner_feature = raw_datasets["train"].features["ner_tags"]
print(ner_feature)

label_names = ner_feature.feature.names
print(label_names)

words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(tokenizer.is_fast)

inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
print(inputs.tokens())

print(inputs.word_ids())


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
print(batch["labels"])

for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])

metric = evaluate.load("seqeval")

labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
print(labels)

predictions = labels.copy()
predictions[2] = "O"
metric.compute(predictions=[predictions], references=[labels])


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

model.config.num_labels

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()

model_checkpoint = "huggingface-course/bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

from pathlib import Path
import re
from sklearn.model_selection import train_test_split


def read_wnut(file_path):
    file_path = Path('data/wnut17train.conll')

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
