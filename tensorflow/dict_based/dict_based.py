import os
import re
from tqdm import tqdm
import json


def build_dict(dict_path):
    dictionary = []
    same = []
    for file in os.listdir(dict_path):
        lines = open(os.path.join(dict_path, file), encoding='utf-8').readlines()
        for line in lines:
            if len(line) < 1:
                continue
            if '=' in line:
                a, b = line[:-1].split('=')
                same.append((a, b, file[1:-4]))
                dictionary.append((a, file[1:-4]))
                dictionary.append((b, file[1:-4]))
            else:
                dictionary.append((line[:-1], file[1:-4]))

    return sorted(dictionary, key=lambda x: len(x[0]), reverse=True), same


def lookup(line):
    sentences = re.split('！|。|？', line)
    if sentences[-1] == '':
        sentences = sentences[:-1]
    line_res = []
    for sent in sentences:
        label = ['O' for _ in range(len(sent))]
        for v, c in dictionary:
            index = sent.find(v)
            if index >= 0:
                if len(set(label[index:index + len(v)])) == 1:
                    label[index] = 'B-' + c
                    for i in range(index + 1, index + len(v)):
                        label[i] = 'I-' + c
        line_res.append((sent, label))

    return line_res


if __name__ == "__main__":
    data_path = './data'
    save_root = './json'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    dictionary, same = build_dict('./dictionary')
    files = os.listdir(data_path)
    for id, file in enumerate(files):
        print('%d/%d, %s' % (id + 1, len(files), file))
        res = []
        lines = open(os.path.join(data_path, file), encoding='utf-8').readlines()
        lines = [re.sub('\s', '', line) for line in lines if len(line) > 1]
        for line in tqdm(lines):
            line_res = lookup(line)
            res.append(line_res)
        name, _ = os.path.splitext(file)


        save_path = os.path.join(save_root, name + '.json')
        json.dump({"data": res}, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
