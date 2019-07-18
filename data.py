import glob
import os.path as osp
import string
import torch
import unicodedata

all_letter = string.ascii_letters + " .,;'"


def findfiles(filename):
    return glob.glob(filename)


def readlines(filename):
    return open(filename, encoding='utf-8').read().strip().split('\n')


def preprocess_data(path):
    all_categories = []
    category_line = {}
    for file_path in findfiles(path):
        category = osp.splitext(osp.basename(file_path))[0]
        lines = readlines(file_path)
        all_categories.append(category)
        category_line[category] = lines
    return all_categories, category_line


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letter
    )


def letter_tens(letter):
    letter_tensor = torch.zeros((1, len(all_letter)))
    letter_tensor[0][all_letter.index(letter)] = 1
    return letter_tensor


def line_tens(name):
    name = unicode_to_ascii(name)
    line_tensor = torch.zeros((len(name), 1, len(all_letter)))
    for li in range(len(name)):
        line_tensor[li][0][all_letter.index(name[li])] = 1
    return line_tensor


def category_tens(category, all_categories):
    idx = all_categories.index(category)
    category_tensor = torch.tensor([idx], dtype=torch.long)
    return category_tensor


def get_data(path='data/names/*.txt', train_ratio=0.9):

    all_categories, category_line = preprocess_data(path)

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for category in all_categories:
        lines = category_line[category]
        for l in lines[:int(len(lines) * train_ratio)]:
            x_train.append(line_tens(l))
            y_train.append(category_tens(category, all_categories))
        for l in lines[int(len(lines) * train_ratio):]:
            x_val.append(line_tens(l))
            y_val.append(category_tens(category, all_categories))
    return all_letter, all_categories, x_train, y_train, x_val, y_val
            

if __name__ == '__main__':
    # print(findfiles())
    # print(preprocess_data())
    # print(string.ascii_letters)
    # all_letter = string.ascii_letters + " .,;'"
    # print(line('ab', all_letter))
    get_data()
