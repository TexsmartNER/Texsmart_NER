from dataclasses import dataclass, field

from transformers import HfArgumentParser, TrainingArguments, AutoConfig, Trainer, BertForTokenClassification

from data_utils import *


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="bert-large-uncased", metadata={"help": "Path to model"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data"})


@dataclass
class TrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    language: str = field(default="en", metadata={"help": "language to finetune ner task"})
    from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether to train from scratch"}
    )


def finetune():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=training_args.model_max_length)

    # load dataset and data_collator
    train_dataset = FinetuneDataset(data_path=data_args.data_path, split="train", lang=training_args.language)
    data_collator = FinetuneCollator(tokenizer=tokenizer, lang=training_args.language)

    # load model
    num_labels = len(english_tag2idx) if training_args.language == "en" else len(chinese_tag2idx)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = num_labels
    if training_args.from_scratch:
        model = BertForTokenClassification(config)
    else:
        model = BertForTokenClassification.from_pretrained(model_args.model_name_or_path, config=config)

    print("Total parameters: {:.2f}M".format(model.num_parameters() / 1e6))

    # training
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset,
                      data_collator=data_collator)
    trainer.train()


if __name__ == '__main__':
    finetune()
