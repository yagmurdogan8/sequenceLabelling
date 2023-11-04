import evaluate
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import AutoTokenizer, DataCollatorForTokenClassification

sentences = []


def convert_iob_to_hf_format(input_file):
    with open(input_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    current_sentence = []
    for line in lines:
        line = line.strip()
        if not line:  # Handle empty lines
            if current_sentence:
                sentences.append(current_sentence)
            current_sentence = []
        elif len(line) == 1:
            current_sentence.append(line)
        else:
            token, label = line.split("\t")
            current_sentence.append((token, label))
    return sentences


train_data = convert_iob_to_hf_format('data/wnut17train.conll')
dev_data = convert_iob_to_hf_format('data/emerging.dev.conll')
test_data = convert_iob_to_hf_format('data/emerging.test.annotated')


def ids_tokens_nertags(sentences):
    res_ids = []
    res_tokens = []
    res_ner_tags = []
    for id, token_ner_tags in enumerate(sentences):
        res_ids.append(id)
        tokens, ner_tags = zip(*token_ner_tags)
        res_tokens.append(list(tokens))
        res_ner_tags.append(list(ner_tags))

    return res_ids, res_tokens, res_ner_tags


ids_train, tokens_train, ner_tags_str_train = ids_tokens_nertags(train_data)
ids_dev, tokens_dev, ner_tags_str_dev = ids_tokens_nertags(dev_data)
ids_test, tokens_test, ner_tags_str_test = ids_tokens_nertags(test_data)

# print(ner_tags_str[0])

ner_tag_to_int = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12
}

ner_tags_int_train = []
ner_tags_int_dev = []
ner_tags_int_test = []

for sentence_ner_tags in ner_tags_str_train:
    single_ner_tags_int = [ner_tag_to_int[ner_tags] for ner_tags in sentence_ner_tags]
    ner_tags_int_train.append(single_ner_tags_int)

for sentence_ner_tags in ner_tags_str_dev:
    single_ner_tags_int = [ner_tag_to_int[ner_tags] for ner_tags in sentence_ner_tags]
    ner_tags_int_dev.append(single_ner_tags_int)

for sentence_ner_tags in ner_tags_str_test:
    single_ner_tags_int = [ner_tag_to_int[ner_tags] for ner_tags in sentence_ner_tags]
    ner_tags_int_test.append(single_ner_tags_int)

train_dict = {
    'id': ids_train,
    'tokens': tokens_train,
    'ner_tags_int': ner_tags_int_train
}
dev_dict = {
    'id': ids_dev,
    'tokens': tokens_dev,
    'ner_tags_int': ner_tags_int_dev
}
test_dict = {
    'id': ids_test,
    'tokens': tokens_test,
    'ner_tags_int': ner_tags_int_test
}

train_dataset = Dataset.from_dict(train_dict)
dev_dataset = Dataset.from_dict(dev_dict)
test_dataset = Dataset.from_dict(test_dict)

dataset = DatasetDict({'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset})
print(dataset)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# print(inputs.tokens)


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


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags_int"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_train_dataset = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

tokenized_dev_dataset = dev_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["dev"].column_names,
)

tokenized_test_dataset = test_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["test"].column_names,
)
# Fine tuning

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

batch = data_collator([tokenized_train_dataset[i] for i in range(2)])
# print(batch["labels"])
# print(data_collator)

# evaluate

tf_train_dataset = tokenized_train_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)

tf_dev_dataset = tokenized_dev_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)

tf_test_dataset = tokenized_test_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)
#
# metric = evaluate.load("seqeval")
#
# labels = ner_tags_str_train[train_dataset[0]["id"]]
#
# predictions = ["O",
#                "B-corporation",
#                "I-corporation",
#                "B-creative-work",
#                "I-creative-work",
#                "B-group",
#                "I-group",
#                "B-location",
#                "I-location",
#                "B-person",
#                "I-person",
#                "B-product",
#                "I-product"]
#
# f1 = f1_score(labels, predictions)
# precision = precision_score(labels, predictions)
# recall = recall_score(labels, predictions)
#
# # Generate a classification report
# report = classification_report(labels, predictions)
#
# print("F1 Score:", f1)
# print("Precision:", precision)
# print("Recall:", recall)
# print("Classification Report:", report)
