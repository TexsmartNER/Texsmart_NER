
import argparse

import torch
from transformers import BertForTokenClassification

from data_utils import *
from eval_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="checkpoints/ckpt_en", help="path to model")
    parser.add_argument("--lang", type=str, choices=['en', 'zh'], default='en', help="which language")

    args = parser.parse_args()
    return args


@torch.no_grad()
def ner(args, examples):
    if isinstance(examples, str):
        examples = [examples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if device == torch.device('cpu'):
        print("GPU is not available !")
        model = BertForTokenClassification.from_pretrained(args.model_name_or_path, device_map=device)
    else:
        print("GPU is available !")
        model = BertForTokenClassification.from_pretrained(args.model_name_or_path, device_map=device,
                                                           torch_dtype=torch.float16)
    model.eval()

    # split text to words
    sentence_words = get_sentence(examples, lang=args.lang)
    sentence_words_offset = get_offset(examples, sentence_words)
    sentence_word_position = [list(range(len(text))) for text in sentence_words]

    # prepare input
    max_len = -1
    input_ids, positions = [], []
    for words_list, pos_list in zip(sentence_words, sentence_word_position):
        token_list, label_list = tokenize_and_preserve_labels(words_list, pos_list, tokenizer, args.lang)
        input_ids.append(token_list)
        positions.append(label_list)
        max_len = max(max_len, len(token_list))

    input_ids = pad_to_length(input_ids, max_len, tokenizer.pad_token_id).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.int).to(device)
    positions = pad_to_length(positions, max_len, -100).tolist()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch_size, seq_len, num_labels)
    predictions = logits.argmax(dim=-1).tolist()  # (batch_size, seq_len)

    if args.lang == "en":
        token_predictions = [[english_idx2tag[i] for i in piece] for piece in predictions]
    else:
        token_predictions = [[chinese_idx2tag[i] for i in piece] for piece in predictions]

    predictions = get_prediction(token_predictions, positions, -100)
    for example, word, pred, offset in zip(examples, sentence_words, predictions, sentence_words_offset):
        print("text:", example)
        print("words:", [{"word": w, "hit": o} for w, o in zip(word, offset)])
        print("label predictions:", pred)
        print("entity predictions:", get_span_result(word, pred, example, offset))
        print()


if __name__ == '__main__':
    args = parse_args()

    english_examples = [
        "This morning I met with Senators Inabo and Senior from Palau to discuss my role as Chair of the Public Works",
        "Best wishes to Kevin , Therese & their family as they embark on the next stage of their lives.",
        "Today in QT I outlined some of the positive changes the Coalition will introduce for ADF personnel and their families.",
        "Welcom to New York City, my favorite city is San Francisco",
        'John Smith stayed in San Francisco last month.',
        'Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to analyze and understand human language.',
        'TensorFlow (released as open source on November 2015) is a library developed by Google to accelerate deep learning research.',
        'Captain Marvel was premiered in Los Angeles 14 months ago.',
        'Welcom%%;to New York City, my favorite city is San Francisco.'
    ]

    chinese_examples = [
        "浙江省金华的义乌市，我国最大的小商品交易基地。",
        "这意味着其核酸检测试剂盒将正式获准进入欧盟市场。",
        "这样才可能真正理解设计，雕琢出设计者所期待的玉器。",
        '自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。',
        '支持在具备条件的行业领域和企业范围探索大数据、人工智能、云计算、数字孪生、5G、物联网和区块链等新一代数字技术应用和集成创新。',
        '2020年2月7日，经中央批准，国家监察委员会决定派出调查组赴湖北省武汉市，就群众反映的涉及李文亮医生的有关问题作全面调查。',
        '上个月30号，南昌王先生在自己家里边看流浪地球边吃煲仔饭。',
        '董明珠出手，格力电器60亿回购计划亮相',
        '刘备张飞关羽等人是三国时期的著名人物。'
    ]

    if args.lang == "en":
        ner(args, english_examples)
    else:
        ner(args, chinese_examples)
