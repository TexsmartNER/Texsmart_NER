import glob
from dataclasses import dataclass, field

from transformers import HfArgumentParser, TrainingArguments, AutoConfig, BertForMaskedLM, Trainer

from data_utils import *


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="bert-large-uncased", metadata={"help": "Path to model"})
    tokenizer_path: str = field(default="bert-en-zh", metadata={"help": "Path to tokenizer"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data"})

@dataclass
class TrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "The probability with which to (randomly) mask tokens in the input"}
    )
    stage: int = field(default=1, metadata={"help": "which stage"})


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        model_max_length=training_args.model_max_length
    )

    # load dataset and data_collator
    train_dataset = PretrainDataset(data_path=data_args.data_path)
    data_collator = PretrainCollator(tokenizer=tokenizer, mlm_probability=training_args.mlm_probability)

    # load model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.vocab_size = len(tokenizer)
    config.max_position_embeddings = 2048  # we train a bert model which max_seq_len = 2048
    if training_args.stage == 1:
        model = BertForMaskedLM(config)
    else:
        model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path, config=config)

    print("Total parameters: {:.2f}M".format(model.num_parameters() / 1e6))

    # training
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset,
                            data_collator=data_collator)

    # resume training
    if glob.glob(f"{training_args.output_dir}/checkpoint-*"):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == '__main__':
    train()
