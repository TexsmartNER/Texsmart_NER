from typing import List, Union

punctuation = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_', '、', '#', '—', '［', '＼', '”', '－', '〜', '&', '】',
               '=', '`', '‛', '\u3000', '?', '～', '＾', '〘', '？', '〔', '’', '＂', "'", '（', '，', '(', '！', '｛', '「',
               '\\', '＜', '｣', '〟', '}', '·', '､', '<', '％', '］', '$', '‘', '〚', '｜', '》', '–', '|', ';', '＄', '＃',
               '｠', ']', '〃', '/', '＋', ',', '）', '〈', '。', '｟', '…', '＇', '＞', '｡', '』', '〖', '﹑', '^', '%', '〕',
               '>', '~', '｝', '﹔', '“', '.', '[', '+', '＝', '〉', '【', '〗', '{', '„', '*', '《', '〝', '」', ':', '‧',
               '｢', '＊', '：', '"', '『', '〾', '．', '@', '〞', '｀', '!', ')', '／', '〰', '〛', '〙', '﹏', '；', '-', '‟',
               '＆', '＠', '〿', '＿']


def get_sentence(text: List[str], lang="en"):
    result = []
    for sent in text:
        if lang == "en":
            for char in punctuation:
                sent = sent.replace(char, f" {char} ")
            result.append(sent.split())
        else:
            result.append(list(sent))

    return result


def get_span_result(word_list: List[str], pred_list: List[str], sentence: str, offset: List[List[int]]):
    result = []
    right = 0
    while right < len(word_list):
        if pred_list[right] == "O":
            right += 1
            continue
        elif pred_list[right].startswith("B-"):
            left = right
            right = right + 1
            while right < len(word_list) and pred_list[right].startswith("I-") and pred_list[right][2:] == pred_list[left][2:]:
                right += 1

            hit = [offset[left][0], offset[right-1][0] + offset[right-1][1] - offset[left][0]]
            string = sentence[hit[0]: hit[0] + hit[1]]
            result.append({
                "string": string,
                "hit": hit,
                "type": pred_list[left][2:],
            })
        else:
            right += 1
            continue

    return result


def get_prediction(predictions: List[List[int]], positions: List[List[int]], ignore_id: int = -100):
    result = []
    for i in range(len(predictions)):
        pred, pos = predictions[i], positions[i]
        rec = []

        for j in range(len(pos)):
            if pos[j] == ignore_id or pos[j] == pos[j - 1]:
                continue
            else:
                rec.append(pred[j])

        result.append(rec)

    return result


def get_offset(sentence: List[str], words: List[List[str]]):
    offset = []
    for sent, word_list in zip(sentence, words):
        temp = []
        start = 0
        cnt = 0
        while start < len(sent):
            if sent[start].isspace():
                start += 1
                continue
            else:
                temp.append([start, len(word_list[cnt])])
                start += len(word_list[cnt])
                cnt += 1

        offset.append(temp)

    return offset
