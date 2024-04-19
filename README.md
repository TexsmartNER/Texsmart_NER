# BERT-NER

**Display Language: [English](README.md), [简体中文](README_zh.md).**

## Introduction
**BERT-NER** is a fine-tuned BERT model that is ready to use for **Named Entity recognition** task. We merge vocabulary of [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese), then pretrain a bilingual bert-large model using MLM loss. After that, we finetune the model on English and Chinese NER dataset respectively and release the corresponding checkpoints. Note that though we set `max_position_embeddings` to 2048, the maximum length we trained on was 1536.

## Demo
For a given sentence, we will first split words by space (punctuation will be regarded as a single word), then give token-level and word-level prediction. Finally we will return exact entity in the sequence with the  format of  `{'string': entity_string, 'hit': [start_index, offset], 'type': entity_type}`.

**1. English Demo**

```bash
# example1
text: This morning I met with Senators Inabo and Senior from Palau to discuss my role as Chair of the Public Works
words: [{'word': 'This', 'hit': [0, 4]}, {'word': 'morning', 'hit': [5, 7]}, {'word': 'I', 'hit': [13, 1]}, {'word': 'met', 'hit': [15, 3]}, {'word': 'with', 'hit': [19, 4]}, {'word': 'Senators', 'hit': [24, 8]}, {'word': 'Inabo', 'hit': [33, 5]}, {'word': 'and', 'hit': [39, 3]}, {'word': 'Senior', 'hit': [43, 6]}, {'word': 'from', 'hit': [50, 4]}, {'word': 'Palau', 'hit': [55, 5]}, {'word': 'to', 'hit': [61, 2]}, {'word': 'discuss', 'hit': [64, 7]}, {'word': 'my', 'hit': [72, 2]}, {'word': 'role', 'hit': [75, 4]}, {'word': 'as', 'hit': [80, 2]}, {'word': 'Chair', 'hit': [83, 5]}, {'word': 'of', 'hit': [89, 2]}, {'word': 'the', 'hit': [92, 3]}, {'word': 'Public', 'hit': [96, 6]}, {'word': 'Works', 'hit': [103, 5]}]
label predictions: ['O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'B-PER', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
entity predictions: [{'string': 'Inabo', 'hit': [33, 5], 'type': 'PER'}, {'string': 'Senior', 'hit': [43, 6], 'type': 'PER'}, {'string': 'Palau', 'hit': [55, 5], 'type': 'LOC'}]

# example2
text: Best wishes to Kevin , Therese & their family as they embark on the next stage of their lives.
words: [{'word': 'Best', 'hit': [0, 4]}, {'word': 'wishes', 'hit': [5, 6]}, {'word': 'to', 'hit': [12, 2]}, {'word': 'Kevin', 'hit': [15, 5]}, {'word': ',', 'hit': [21, 1]}, {'word': 'Therese', 'hit': [23, 7]}, {'word': '&', 'hit': [31, 1]}, {'word': 'their', 'hit': [33, 5]}, {'word': 'family', 'hit': [39, 6]}, {'word': 'as', 'hit': [46, 2]}, {'word': 'they', 'hit': [49, 4]}, {'word': 'embark', 'hit': [54, 6]}, {'word': 'on', 'hit': [61, 2]}, {'word': 'the', 'hit': [64, 3]}, {'word': 'next', 'hit': [68, 4]}, {'word': 'stage', 'hit': [73, 5]}, {'word': 'of', 'hit': [79, 2]}, {'word': 'their', 'hit': [82, 5]}, {'word': 'lives', 'hit': [88, 5]}, {'word': '.', 'hit': [93, 1]}]
label predictions: ['O', 'O', 'O', 'B-PER', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
entity predictions: [{'string': 'Kevin', 'hit': [15, 5], 'type': 'PER'}, {'string': 'Therese', 'hit': [23, 7], 'type': 'PER'}]

# example3
text: Today in QT I outlined some of the positive changes the Coalition will introduce for ADF personnel and their families.
words: [{'word': 'Today', 'hit': [0, 5]}, {'word': 'in', 'hit': [6, 2]}, {'word': 'QT', 'hit': [9, 2]}, {'word': 'I', 'hit': [12, 1]}, {'word': 'outlined', 'hit': [14, 8]}, {'word': 'some', 'hit': [23, 4]}, {'word': 'of', 'hit': [28, 2]}, {'word': 'the', 'hit': [31, 3]}, {'word': 'positive', 'hit': [35, 8]}, {'word': 'changes', 'hit': [44, 7]}, {'word': 'the', 'hit': [52, 3]}, {'word': 'Coalition', 'hit': [56, 9]}, {'word': 'will', 'hit': [66, 4]}, {'word': 'introduce', 'hit': [71, 9]}, {'word': 'for', 'hit': [81, 3]}, {'word': 'ADF', 'hit': [85, 3]}, {'word': 'personnel', 'hit': [89, 9]}, {'word': 'and', 'hit': [99, 3]}, {'word': 'their', 'hit': [103, 5]}, {'word': 'families', 'hit': [109, 8]}, {'word': '.', 'hit': [117, 1]}]
label predictions: ['O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O']
entity predictions: [{'string': 'QT', 'hit': [9, 2], 'type': 'ORG'}, {'string': 'ADF', 'hit': [85, 3], 'type': 'ORG'}]
```

**2. Chinese Demo**

```bash
# example1
text: 浙江省金华的义乌市，我国最大的小商品交易基地。
words: [{'word': '浙', 'hit': [0, 1]}, {'word': '江', 'hit': [1, 1]}, {'word': '省', 'hit': [2, 1]}, {'word': '金', 'hit': [3, 1]}, {'word': '华', 'hit': [4, 1]}, {'word': '的', 'hit': [5, 1]}, {'word': '义', 'hit': [6, 1]}, {'word': '乌', 'hit': [7, 1]}, {'word': '市', 'hit': [8, 1]}, {'word': '，', 'hit': [9, 1]}, {'word': '我', 'hit': [10, 1]}, {'word': '国', 'hit': [11, 1]}, {'word': '最', 'hit': [12, 1]}, {'word': '大', 'hit': [13, 1]}, {'word': '的', 'hit': [14, 1]}, {'word': '小', 'hit': [15, 1]}, {'word': '商', 'hit': [16, 1]}, {'word': '品', 'hit': [17, 1]}, {'word': '交', 'hit': [18, 1]}, {'word': '易', 'hit': [19, 1]}, {'word': '基', 'hit': [20, 1]}, {'word': '地', 'hit': [21, 1]}, {'word': '。', 'hit': [22, 1]}]
label predictions: ['B-loc_generic', 'I-loc_generic', 'I-loc_generic', 'I-loc_generic', 'I-loc_generic', 'O', 'B-loc_generic', 'I-loc_generic', 'I-loc_generic', 'O', 'B-loc_other', 'I-loc_other', 'O', 'O', 'O', 'O', 'I-loc_other', 'I-loc_other', 'I-loc_other', 'I-loc_other', 'I-loc_other', 'I-loc_other', 'O']
entity predictions: [{'string': '浙江省金华', 'hit': [0, 5], 'type': 'loc_generic'}, {'string': '义乌市', 'hit': [6, 3], 'type': 'loc_generic'}, {'string': '我国', 'hit': [10, 2], 'type': 'loc_other'}]

# example2
text: 这意味着其核酸检测试剂盒将正式获准进入欧盟市场。
words: [{'word': '这', 'hit': [0, 1]}, {'word': '意', 'hit': [1, 1]}, {'word': '味', 'hit': [2, 1]}, {'word': '着', 'hit': [3, 1]}, {'word': '其', 'hit': [4, 1]}, {'word': '核', 'hit': [5, 1]}, {'word': '酸', 'hit': [6, 1]}, {'word': '检', 'hit': [7, 1]}, {'word': '测', 'hit': [8, 1]}, {'word': '试', 'hit': [9, 1]}, {'word': '剂', 'hit': [10, 1]}, {'word': '盒', 'hit': [11, 1]}, {'word': '将', 'hit': [12, 1]}, {'word': '正', 'hit': [13, 1]}, {'word': '式', 'hit': [14, 1]}, {'word': '获', 'hit': [15, 1]}, {'word': '准', 'hit': [16, 1]}, {'word': '进', 'hit': [17, 1]}, {'word': '入', 'hit': [18, 1]}, {'word': '欧', 'hit': [19, 1]}, {'word': '盟', 'hit': [20, 1]}, {'word': '市', 'hit': [21, 1]}, {'word': '场', 'hit': [22, 1]}, {'word': '。', 'hit': [23, 1]}]
label predictions: ['O', 'O', 'O', 'O', 'O', 'B-product_generic', 'I-product_generic', 'I-product_generic', 'I-product_generic', 'I-product_generic', 'I-product_generic', 'I-product_generic', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-org_generic', 'I-org_generic', 'O', 'O', 'O']
entity predictions: [{'string': '核酸检测试剂盒', 'hit': [5, 7], 'type': 'product_generic'}, {'string': '欧盟', 'hit': [19, 2], 'type': 'org_generic'}]

# example3
text: 刘备张飞关羽等人是三国时期的著名人物。
words: [{'word': '刘', 'hit': [0, 1]}, {'word': '备', 'hit': [1, 1]}, {'word': '张', 'hit': [2, 1]}, {'word': '飞', 'hit': [3, 1]}, {'word': '关', 'hit': [4, 1]}, {'word': '羽', 'hit': [5, 1]}, {'word': '等', 'hit': [6, 1]}, {'word': '人', 'hit': [7, 1]}, {'word': '是', 'hit': [8, 1]}, {'word': '三', 'hit': [9, 1]}, {'word': '国', 'hit': [10, 1]}, {'word': '时', 'hit': [11, 1]}, {'word': '期', 'hit': [12, 1]}, {'word': '的', 'hit': [13, 1]}, {'word': '著', 'hit': [14, 1]}, {'word': '名', 'hit': [15, 1]}, {'word': '人', 'hit': [16, 1]}, {'word': '物', 'hit': [17, 1]}, {'word': '。', 'hit': [18, 1]}]
label predictions: ['B-person_generic', 'I-person_generic', 'B-person_generic', 'I-person_generic', 'B-person_generic', 'I-person_generic', 'O', 'O', 'O', 'B-time_generic', 'I-time_generic', 'I-time_generic', 'I-time_generic', 'O', 'O', 'O', 'I-person_other', 'I-person_other', 'O']
entity predictions: [{'string': '刘备', 'hit': [0, 2], 'type': 'person_generic'}, {'string': '张飞', 'hit': [2, 2], 'type': 'person_generic'}, {'string': '关羽', 'hit': [4, 2], 'type': 'person_generic'}, {'string': '三国时期', 'hit': [9, 4], 'type': 'time_generic'}]
```

## Getting Started
### Installation

**1. Prepare the code and environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/...
cd BERT-NER
conda env create -f environment.yml
conda activate bert_ner
```

**2. Launching Demo Locally**

For English, run
```bash
python demo.py --model_name_or_path checkpoints/ckpt_en --lang en
```
For Chinese, run
```bash
python demo.py --model_name_or_path checkpoints/ckpt_zh --lang zh
```

## Training data

This model was fine-tuned on [NER Dataset](ner). 

For English dataset, each token will be classified as one of the following classes:

| Abbreviation | Description                                                    |
|--------------|----------------------------------------------------------------|
| O            | Outside of a named entity                                      |
| B-PER        | Beginning of a person’s name right after another person’s name |
| I-PER        | Person’s name                                                  |
| B-ORG        | Beginning of an organization right after another organization  |
| I-ORG        | organization                                                   |
| B-LOC        | Beginning of a location right after another location           |
| I-LOC        | Location                                                       |

For Chinese dataset, each token will be classified as one of the following classes:

```bash
O, B-medicine, I-medicine, B-other, I-other, B-person_occupation, I-person_occupation, B-time_generic, I-time_generic, B-product_generic, I-product_generic, B-food_generic, I-food_generic, B-work_generic, I-work_generic, B-loc_other, I-loc_other, B-org_other, I-org_other, B-quantity_generic, I-quantity_generic, B-org_generic, I-org_generic, B-person_other, I-person_other, B-loc_generic, I-loc_generic, B-person_generic, I-person_generic, B-unmarked, I-unmarked, B-life_organism, I-life_organism, B-event_generic, I-event_generic
```
