#!usr/bin/env python
#-*- coding:utf-8 -*-

#按拼音随机构造训练数据

import random
import pandas as pd
from pypinyin import lazy_pinyin
from step1data_loader import load_dataset, save_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
def is_china_char(ch):
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    return False


def gen_char_dict(dataset):
    """
    统计字频，字:字频率
    """
    char_dict = {}
    for line in dataset:
        line = line.strip()
        for char in line:
            if len(char) != 0 and is_china_char(char):
                char_dict[char] = char_dict.get(char, 0) + 1
    return char_dict


def gen_pinyin_dict(dataset, char_dict,filename):
    """
    统计拼音频率，拼音:频率
    """
    char_pinyin_dict = {}
    for i, line in tqdm(enumerate(dataset)):
        # print(i)
        line = line.strip()
        for c in line:
            if is_china_char(c):
                word_pinyin = ''.join(lazy_pinyin(c))
                if len(word_pinyin) != 0:
                    if word_pinyin not in char_pinyin_dict:
                        char_pinyin_dict[word_pinyin] = c + '_' + str(char_dict.get(c, 0))
                    else:
                        char_pinyin_dict[word_pinyin] += ';' + c + '_' + str(char_dict.get(c, 0)) #多音字直接加法加上

    data = {}
    for pinyin, words in char_pinyin_dict.items():
        tmp = {}
        for word in words.split(';'):
            # print(word)
            if len(word) != 0:
                word_word = word.split('_')[0]
                word_count = int(word.split('_')[1])
                tmp[word_word] = word_count
        data[pinyin] = tmp
    #生成拼音字典，每个拼音下还有对应的字对应的字频的字典
    f = open(f'data/{filename}.model', 'w',encoding='utf-8')
    f.write(str(data))
    f.close()

def load_pinyin_dict(file_path):
    f = open(file_path, 'r',encoding='utf-8')
    a = f.read()
    char_pinyin_dict = eval(a)
    f.close()
    return char_pinyin_dict

#构造数据集：随机拼音，随机mask
def random_word(sentence, char_pinyin_dict, char_dict, confusion_dict=None):
    tokens = [x for x in sentence]
    out = []
    for i, token in enumerate(sentence):
        if not is_china_char(token):
            out.append(str(0))
            continue
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            # 谐音 80%
            if prob < 0.80:
                candiation = char_pinyin_dict.get(''.join(lazy_pinyin(token)), {token:''})
                #拼音对应字频字典，按字频排序
                candiation = sorted(candiation.items(), key=lambda x:x[1], reverse=True)
                candiation = candiation[:int(len(candiation)/2+0.5)]
                candiation = [x[0] for x in candiation if x[0] != token]
                if len(candiation) == 0:
                    out.append(str(0))
                    continue
                tokens[i] = random.choice(candiation)
            # 随机 20%
            else:
                candiation = sorted(char_dict.items(), key=lambda x:x[1], reverse=True)
                candiation = candiation[:int(len(candiation)/2+0.5)]
                candiation = [x[0] for x in candiation if x[0] != token]
                if len(candiation) == 0:
                    out.append(str(0))
                    continue
                tokens[i] = random.choice(candiation)
            out.append(str(1))
            # # 删除 5%
            # elif prob < 0.95:
            #     tokens[i] = ''
            # # 添加 5%
            # else:
            #     candiation = sorted(char_dict.items(), key=lambda x:x[1], reverse=True)
            #     candiation = candiation[:int(len(candiation)/2+0.5)]
            #     candiation = [x[0] for x in candiation]
            #     tokens.insert(i+1, random.choice(candiation))
        else:
            out.append(str(0))
    return ''.join(tokens), ' '.join(out)


def random_dataset(dataset, char_pinyin_dict, char_dict):
    text = []
    out = []
    for ids, line in tqdm(enumerate(dataset)):
        # print(ids)
        line, label = random_word(line, char_pinyin_dict, char_dict)
        text.append(line)
        out.append(label)
    return text, out


if __name__ == '__main__':
    ################头条语料
    # dataset = load_dataset('data/processed_data/all_same_765376/all_data_765376.txt')
    # char_dict = gen_char_dict(dataset)
    # gen_pinyin_dict(dataset, char_dict,'pinyin2char')
    # char_pinyin_dict = load_pinyin_dict('data/pinyin2char.model')
    # process_dataset, process_label = random_dataset(dataset, char_pinyin_dict, char_dict)
    # save_data(process_dataset, 'data/processed_data/process_data_765376.txt')
    # df = pd.DataFrame(columns=['origin_text','random_text','label'])
    # df['origin_text'] = dataset
    # df['random_text'] = process_dataset
    # df['label'] = process_label
    # df.to_csv('data/processed_data/processed_765376.csv', index=False)


    # # df = pd.read_csv('data/processed_data/all_same_765376/processed_765376.csv')
    #
    # dataset = df.values
    # train, test = train_test_split(dataset, test_size=0.1)
    # df = pd.DataFrame(columns=['origin_text','random_text','label'], data=train)
    # df.to_csv('data/processed_data/all_same_765376/train.csv', index=False)
    # df = pd.DataFrame(columns=['origin_text','random_text','label'], data=test)
    # df.to_csv('data/processed_data/all_same_765376/test.csv', index=False)



    ##########################我的语料
    # for windows in [1,2,3,4,5]:
    #     dataset = load_dataset(f'data/processed_data/my/{windows}windows_utterence.txt')
    #     char_dict = gen_char_dict(dataset)
    #     gen_pinyin_dict(dataset, char_dict,'mypinyin2char')
    #     char_pinyin_dict = load_pinyin_dict('data/mypinyin2char.model')
    #
    #     process_dataset, process_label = random_dataset(dataset, char_pinyin_dict, char_dict)
    #     save_data(process_dataset, 'data/processed_data/my_process_data.txt')
    #     df = pd.DataFrame(columns=['origin_text','random_text','label'])
    #     df['origin_text'] = dataset
    #     df['random_text'] = process_dataset
    #     df['label'] = process_label
    #     df.to_csv('data/processed_data/my_processed.csv', index=False)
    #
    #     # df = pd.read_csv('data/processed_data/all_same_765376/processed_765376.csv')
    #
    #     dataset = df.values
    #     df = pd.DataFrame(columns=['origin_text','random_text','label'], data=dataset)
    #     df.to_csv(f'data/processed_data/my/windwos{windows}_dataset.csv', index=False)





    ####################对话语料
    # dataset = load_dataset(f'data/processed_data/my/LCCC.txt')
    # char_dict = gen_char_dict(dataset)
    # gen_pinyin_dict(dataset, char_dict, 'LCCCpinyin2char')
    # char_pinyin_dict = load_pinyin_dict('data/LCCCpinyin2char.model')
    #
    # process_dataset, process_label = random_dataset(dataset, char_pinyin_dict, char_dict)
    # save_data(process_dataset, 'data/processed_data/LCCC_process_data.txt')
    # df = pd.DataFrame(columns=['origin_text', 'random_text', 'label'])
    # df['origin_text'] = dataset
    # df['random_text'] = process_dataset
    # df['label'] = process_label
    # df.to_csv('data/processed_data/LCCC_processed.csv', index=False)
    #
    # # df = pd.read_csv('data/processed_data/all_same_765376/processed_765376.csv')
    #
    # dataset = df.values
    # df = pd.DataFrame(columns=['origin_text', 'random_text', 'label'], data=dataset)
    # df.to_csv(f'data/processed_data/my/LCCC_dataset.csv', index=False)

    #################垂域数据
    dataset = load_dataset(f'data/processed_data/垂域数据/data.txt')
    char_dict = gen_char_dict(dataset)
    # gen_pinyin_dict(dataset, char_dict, 'mypinyin2char')
    char_pinyin_dict = load_pinyin_dict('data/mypinyin2char.model')

    process_dataset, process_label = random_dataset(dataset, char_pinyin_dict, char_dict)
    save_data(process_dataset, 'data/processed_data/垂域数据/data_processed.txt')
    df = pd.DataFrame(columns=['origin_text', 'random_text', 'label'])
    df['origin_text'] = dataset
    df['random_text'] = process_dataset
    df['label'] = process_label
    df.to_csv('data/processed_data/垂域数据/data_processed.csv', index=False)

    # df = pd.read_csv('data/processed_data/all_same_765376/processed_765376.csv')
    dataset = df.values
    df = pd.DataFrame(columns=['origin_text', 'random_text', 'label'], data=dataset)
    df.to_csv(f'data/processed_data/垂域数据/data_processed.csv', index=False)



