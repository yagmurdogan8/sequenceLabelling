from transformers import AutoTokenizer

train_data = []


def read_wnut(file_path):
    with open(file_path, "r", encoding='utf-8') as train_file:
        for line in train_file:
            line = line.strip().split()
            if line:
                tokens, label = line[0], line[1]
                train_data.append({"tokens": tokens, "labels": label})

    return train_data

file_path = "data/wnut17train.conll"
texts, tags = read_wnut('data/wnut17train.conll')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# def ids_tokens_nertags(sentences):
#     res_ids = []
#     res_tokens = []
#     res_ner_tags = []
#     for id, token_ner_tags in enumerate(sentences):
#         res_ids.append(id)
#         tokens, ner_tags = zip(*token_ner_tags)
#         res_tokens.append(list(tokens))
#         res_ner_tags.append(list(ner_tags))
#
#     return res_ids, res_tokens, res_ner_tags
#
#
# ids, tokens, ner_tags_str = ids_tokens_nertags(sentences)


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


# preprocess data
train_data = [tokenize_and_preprocess_data(example) for example in train_data]

ner_feature = train_data[0]["labels"]
print(ner_feature)

print(train_data)
