import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForTokenClassification


def convert_iob_to_hf_format(input_file):
    sentences = []
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
for sentence_ner_tags in ner_tags_str_train:
    single_ner_tags_int = [ner_tag_to_int[ner_tags] for ner_tags in sentence_ner_tags]
    ner_tags_int_train.append(single_ner_tags_int)

ner_tags_int_dev = []
for sentence_ner_tags in ner_tags_str_dev:
    single_ner_tags_int = [ner_tag_to_int[ner_tags] for ner_tags in sentence_ner_tags]
    ner_tags_int_dev.append(single_ner_tags_int)

ner_tags_int_test = []
for sentence_ner_tags in ner_tags_str_test:
    single_ner_tags_int = [ner_tag_to_int[ner_tags] for ner_tags in sentence_ner_tags]
    ner_tags_int_test.append(single_ner_tags_int)

train_dict = {
    'id': ids_train,
    'tokens': tokens_train,
    'ner_tags': ner_tags_int_train
}

dev_dict = {
    'id': ids_dev,
    'tokens': tokens_dev,
    'ner_tags': ner_tags_int_dev
}

test_dict = {
    'id': ids_test,
    'tokens': tokens_test,
    'ner_tags': ner_tags_int_test
}
train_dataset = Dataset.from_dict(train_dict)
test_dataset = Dataset.from_dict(test_dict)
dev_dataset = Dataset.from_dict(dev_dict)

all_dataset = DatasetDict({'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset})
print(all_dataset)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_dataset = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=all_dataset['train'].column_names
)
print(tokenized_dataset)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_dataset[i] for i in range(2)])
# print(batch["labels"])
# print(data_collator)

# evaluate

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
# metric.compute(predictions=[predictions], references=[labels])
