from transformers import AutoTokenizer

sentences = []


def convert_iob_to_hf_format(input_file):
    with open(input_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    current_sentence = []
    for line in lines:
        line = line.strip()
        if line == "":
            if current_sentence:
                sentences.append(current_sentence)
            current_sentence = []
        else:
            token, label = line.split("\t")
            current_sentence.append((token, label))
    return sentences


train_data = convert_iob_to_hf_format('data/wnut17train.conll')


# dev_data = convert_iob_to_hf_format('data/emerging.dev.conll')
# test_data = convert_iob_to_hf_format('data/emerging.test.annotated')


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


ids, tokens, ner_tags_str = ids_tokens_nertags(sentences)
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

ner_tags_int = []

# print(sentences)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

inputs = tokenizer(train_data[0], is_split_into_words=True)


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
#
#
# def read_from_wnut_file(path):
#     train_data = []
#     with open(path, 'r', encoding='utf-8') as file:
#         for line in file:
#             line = line.strip()
#             if line:
#                 tokens, label = line[0], line[1]
#                 train_data.append({"tokens": tokens, "labels": label})
#
#     return train_data
#
#
# wnut_file = 'data/wnut17train.conll'
# sentences = read_from_wnut_file(wnut_file)
#
#
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
# #
# # train_data = []
# #
# #
# # def read_wnut():
# #     with open("data/wnut17train.conll", "r", encoding='utf-8') as train_file:
# #         for line in train_file:
# #             line = line.strip().split()
# #             if line:
# #                 tokens, label = line[0], line[1]
# #                 train_data.append({"tokens": tokens, "labels": label})
# #
# #     return train_data
# #
# #
# # file_path = "data/wnut17train.conll"
# #
# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# #
# #
# # # def ids_tokens_nertags(sentences):
# # #     res_ids = []
# # #     res_tokens = []
# # #     res_ner_tags = []
# # #     for id, token_ner_tags in enumerate(sentences):
# # #         res_ids.append(id)
# # #         tokens, ner_tags = zip(*token_ner_tags)
# # #         res_tokens.append(list(tokens))
# # #         res_ner_tags.append(list(ner_tags))
# # #
# # #     return res_ids, res_tokens, res_ner_tags
# # #
# # #
# # # ids, tokens, ner_tags_str = ids_tokens_nertags(sentences)
# #
# #
# # def tokenize_and_preprocess_data(example):
# #     tokens = example['tokens']
# #     labels = example['labels']
# #
# #     # Tokenize the text
# #     tokenized_input = tokenizer(tokens, is_split_into_words=True)
# #
# #     # Ensure that the tokenized input matches the token boundaries
# #     assert len(tokenized_input['input_ids']) == len(tokens)
# #
# #     return {
# #         'input_ids': tokenized_input['input_ids'],
# #         'attention_mask': tokenized_input['attention_mask'],
# #         'labels': labels,
# #     }
# #
# #
# # # preprocess data
# # train_data = [tokenize_and_preprocess_data(example) for example in train_data]
# #
# # # ner_feature = train_data[0]["labels"]
# # # print(ner_feature)
# #
# # print(train_data)
