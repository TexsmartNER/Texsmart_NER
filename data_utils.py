import torch
from typing import List, Dict

from torch.utils.data import IterableDataset, DataLoader, Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk

english_tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
english_idx2tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 5: 'B-ORG', 6: 'I-ORG'}

chinese_tag2idx = {'O': 0, 'B-medicine': 1, 'I-medicine': 2, 'B-other': 3, 'I-other': 4, 'B-person_occupation': 5,
                   'I-person_occupation': 6, 'B-time_generic': 7, 'I-time_generic': 8, 'B-product_generic': 9,
                   'I-product_generic': 10, 'B-food_generic': 11, 'I-food_generic': 12, 'B-work_generic': 13,
                   'I-work_generic': 14, 'B-loc_other': 15, 'I-loc_other': 16, 'B-org_other': 17, 'I-org_other': 18,
                   'B-quantity_generic': 19, 'I-quantity_generic': 20, 'B-org_generic': 21, 'I-org_generic': 22,
                   'B-person_other': 23, 'I-person_other': 24, 'B-loc_generic': 25, 'I-loc_generic': 26,
                   'B-person_generic': 27, 'I-person_generic': 28, 'B-unmarked': 29, 'I-unmarked': 30,
                   'B-life_organism': 31, 'I-life_organism': 32, 'B-event_generic': 33, 'I-event_generic': 34}
chinese_idx2tag = {0: 'O', 1: 'B-medicine', 2: 'I-medicine', 3: 'B-other', 4: 'I-other', 5: 'B-person_occupation',
                   6: 'I-person_occupation', 7: 'B-time_generic', 8: 'I-time_generic', 9: 'B-product_generic',
                   10: 'I-product_generic', 11: 'B-food_generic', 12: 'I-food_generic', 13: 'B-work_generic',
                   14: 'I-work_generic', 15: 'B-loc_other', 16: 'I-loc_other', 17: 'B-org_other', 18: 'I-org_other',
                   19: 'B-quantity_generic', 20: 'I-quantity_generic', 21: 'B-org_generic', 22: 'I-org_generic',
                   23: 'B-person_other', 24: 'I-person_other', 25: 'B-loc_generic', 26: 'I-loc_generic',
                   27: 'B-person_generic', 28: 'I-person_generic', 29: 'B-unmarked', 30: 'I-unmarked',
                   31: 'B-life_organism', 32: 'I-life_organism', 33: 'B-event_generic', 34: 'I-event_generic'}


def pad_to_length(sequence: List[List[int]], length: int, pad_value: int) -> torch.Tensor:
    for i in range(len(sequence)):
        sequence[i] += [pad_value] * (length - len(sequence[i]))

    return torch.tensor(sequence)


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer, lang="en"):
    """
    copy from https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L118
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = [tokenizer.cls_token]
    labels = [-100]

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word) if lang == "en" else [word]
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    tokenized_sentence.append(tokenizer.sep_token)
    labels.append(-100)

    tokenized_sentence = tokenizer.convert_tokens_to_ids(tokenized_sentence)[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]
    return tokenized_sentence, labels


class PretrainDataset(IterableDataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        with open(self.data_path, 'r') as file:
            for line in file:
                text = line.strip()
                if len(text) == 0:
                    continue
                else:
                    yield text


class FinetuneDataset(Dataset):
    def __init__(self, data_path, split="test", lang="en"):
        self.dataset = load_from_disk(data_path)[split]
        self.tag2idx = english_tag2idx if lang == "en" else chinese_tag2idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.dataset[idx]['tokens']
        labels = [self.tag2idx[tag] for tag in self.dataset[idx]['ner_tags']]
        return {
            "tokens": tokens,
            "labels": labels
        }


class PretrainCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mask_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

    def __call__(self, instances: List[str]):
        inputs = [self.tokenizer(text, return_special_tokens_mask=True, truncation=True, padding="max_length",
                                 max_length=self.tokenizer.model_max_length) for text in instances]
        return self.mask_collator(inputs)


class FinetuneCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, lang="en"):
        self.tokenizer = tokenizer
        self.lang = lang

    def __call__(self, instances: List[Dict]):
        max_len = -1
        input_ids, labels = [], []
        for ins in instances:
            token_list, label_list = tokenize_and_preserve_labels(ins['tokens'], ins['labels'], self.tokenizer,
                                                                  self.lang)
            input_ids.append(token_list)
            labels.append(label_list)
            max_len = max(max_len, len(token_list))

        input_ids = pad_to_length(input_ids, max_len, self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.int)
        labels = pad_to_length(labels, max_len, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
