from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, BertForTokenClassification

from data_utils import *
from functools import partial
from eval_utils import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="bert-large-uncased", metadata={"help": "Path to model"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data"})


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="hello")
    language: str = field(default="en", metadata={"help": "language to finetune ner task"})


def collate(batch, tokenizer, lang):
    words, labels = [], []
    for ins in batch:
        words.append(ins['tokens'])
        labels.append(ins['labels'])

    input_ids, positions = [], []

    max_len = -1
    word_position = [list(range(len(text))) for text in words]
    for words_list, pos_list in zip(words, word_position):
        # print("words_list", words_list)
        # print("pos_list", pos_list)
        token_list, pos_list = tokenize_and_preserve_labels(words_list, pos_list, tokenizer, lang)
        # print(token_list)
        # print(pos_list)
        input_ids.append(token_list)
        positions.append(pos_list)
        max_len = max(max_len, len(token_list))

    input_ids = pad_to_length(input_ids, max_len, tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.int)
    positions = pad_to_length(positions, max_len, -100).tolist()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": positions,
        "labels": labels,
    }

@torch.no_grad()
def evaluate():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # load dataset and data_collator
    test_dataset = FinetuneDataset(data_path=data_args.data_path, split="test", lang=training_args.language)
    collator = partial(collate, tokenizer=tokenizer, lang=training_args.language)
    dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=collator)

    # load model
    model = BertForTokenClassification.from_pretrained(model_args.model_name_or_path, device_map=device, torch_dtype=torch.float16)
    model.eval()
    print("Total parameters: {:.2f}M".format(model.num_parameters() / 1e6))

    eval_preds, eval_labels = [], []
    for batch in tqdm(dataloader):
        input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        labels, positions = batch["labels"], batch["positions"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, num_labels)
        predicitons = logits.argmax(dim=-1).tolist()  # (batch_size, seq_len)

        predictions = get_prediction(predicitons, positions, -100)

        eval_preds.extend(predictions)
        eval_labels.extend(labels)

    precision, recall, f1, accuracy = 0, 0, 0, 0
    for pred, label in tqdm(zip(eval_preds, eval_labels)):
        precision += precision_score(pred, label, average="micro")
        recall += recall_score(pred, label, average="micro")
        f1 += f1_score(pred, label, average="micro")
        accuracy += accuracy_score(pred, label)

    precision /= len(eval_labels)
    recall /= len(eval_labels)
    f1 /= len(eval_labels)
    accuracy /= len(eval_labels)

    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, Accuracy: {:.2f}".format(precision, recall, f1, accuracy))


if __name__ == '__main__':
    evaluate()