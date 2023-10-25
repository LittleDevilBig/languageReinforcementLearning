r"""Train policy using curiosity."""
r"""Train with using QA. (Ext + int reward)"""
import sys
sys.path.append("/home/yaomeng/clevr_robot_env")
from agents import *
from config import *
from utils import *
from torch.multiprocessing import Pipe

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from video_utils import *
from sklearn import metrics
from torch.utils import data
import os
import random
import numpy as np
import nltk
import torch
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import gensim

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


best_acc = 0.6


import numpy as np
import time
import copy
import csv

from clevr_robot_env import ClevrEnv

import argparse
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clevr_robot_env import ClevrEnv

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn



#import iep.utils as utils

#import iep.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--use_gpu', default=1, type=int)

# For running on a preprocessed dataset
parser.add_argument('--input_question_h5', default='data/val_questions.h5')
parser.add_argument('--input_features_h5', default='data-ssd/val_features.h5')
parser.add_argument('--use_gt_programs', default=0, type=int)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--question', default=None)
parser.add_argument('--image', default=None)
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=64, type=int)
parser.add_argument('--image_height', default=64, type=int)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--family_split_file', default=None)

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)

#from scripts.run_model import run_single_example

writer = SummaryWriter()


import time
# class MR_Dataset(data.Dataset):
#     def __init__(self, state="train", k=0, embedding_type="word2vec"):
#
#         self.path = os.path.abspath('.')
#         if "data" not in self.path:
#             self.path += "/data"
#         # 数据集加载，每行都是一句话
#         self.k = k
#         self.state = state
#         self.embedding_type = embedding_type
#
#     def get_data_ym(self, pos_samples, neg_samples, pos_state, neg_state):
#
#         if pos_samples == []:
#             raise
#         if neg_samples == []:
#             raise
#         if pos_state == []:
#             raise
#         if neg_state == []:
#             raise
#
#         state_data = pos_state + neg_state
#         # 把正负样本放在一块
#         datas = pos_samples + neg_samples
#         # datas = [nltk.word_tokenize(data) for data in datas]
#         # 用空格进行分词
#         datas = [data.split() for data in datas]
#         # 求句子最大长度，将所有句子pad成一样的长度；
#         # 有时也可以pad为平均长度，如果使用平均长度，对于超过长度的句子需要截断。
#         max_sample_length = max([len(sample) for sample in datas])
#         # 为正负样本设置标签
#         labels = [1] * len(pos_samples) + [0] * len(neg_samples)
#         word2id = {"<pad>": 0}  # 生成word2id，pad对应0
#         datases = [[] for data in datas]
#         for i, data in enumerate(datas):
#             for j, word in enumerate(data):
#                 # 词不还没加入word2id中则加入，并设置其id
#                 if word2id.get(word) is None:
#                     word2id[word] = len(word2id)
#                 # 设置每个句子中所有词，替换为ID
#                 datas[i][j] = word2id[word]
#             # 将句子按最大长度进行pad
#             datas[i] = datas[i] + [0] * (max_sample_length - len(datas[i]))
#             datases[i].append(datas[i])
#             datases[i].append(state_data[i])
#
#             # 如果是按平均长度则按下面语句进行截断或补齐,max_sample_length代表平均长度
#             # datas[i] = datas[i][0:max_sample_length]+[0]*(max_sample_length-len(datas[i]))
#         self.n_vocab = len(word2id)
#         self.word2id = word2id
#
#         #根据配置取不同的预训练词向量
#         # if self.embedding_type == "word2vec":
#         #     self.get_word2vec()
#         # elif self.embedding_type == "glove":
#         #     self.get_glove_embedding()
#         # else:
#         #     pass
#         # self.get_word2vec()
#         # 由于训练集中的数据前半部分是正样本，后半部分是负样本，需要打乱训练集
#         # 把数据和标签放到一起打乱
#         c = list(zip(datases, labels))  # 打乱训练集
#         random.seed(1)
#         random.shuffle(c)
#         # 再把数据和标签分开
#         datas[:], labels[:] = zip(*c)
#
#         # 生成训练集、验证集和测试集
#         # 总的数据分成10份，第k份作为测试集，其他9份再分
#         # 其他9分的后10%做为验证集，前90%做为训练集
#         if self.state == "train":  # 生成训练集
#             # 取除第k份外其他9份
#             self.datas = datas[:int(self.k * len(datas) / 10)] + datas[int((self.k + 1) * len(datas) / 10):]
#             self.labels = labels[:int(self.k * len(datas) / 10)] + labels[int((self.k + 1) * len(labels) / 10):]
#             # 取前90%做为训练集
#             self.datas = np.array(self.datas[0:int(0.9 * len(self.datas))])
#             self.labels = np.array(self.labels[0:int(0.9 * len(self.labels))])
#         elif self.state == "valid":  # 生成验证集
#             # 取除第k份外其他9份
#             self.datas = datas[:int(self.k * len(datas) / 10)] + datas[int((self.k + 1) * len(datas) / 10):]
#             self.labels = labels[:int(self.k * len(datas) / 10)] + labels[int((self.k + 1) * len(labels) / 10):]
#             # 取后10%做为验证集
#             self.datas = np.array(self.datas[int(0.9 * len(self.datas)):])
#             self.labels = np.array(self.labels[int(0.9 * len(self.labels)):])
#         elif self.state == "test":  # 生成测试集
#             # 第k份作为测试集
#             self.datas = np.array(datas[int(self.k * len(datas) / 10):int((self.k + 1) * len(datas) / 10)])
#             self.labels = np.array(labels[int(self.k * len(datas) / 10):int((self.k + 1) * len(datas) / 10)])
#
#     def __getitem__(self, index):
#         return self.datas[index], self.labels[index]
#
#     def __len__(self):
#         return len(self.datas)
#
#     def get_glove_embedding(self):
#         '''
#         生成glove词向量
#         :return: 根据词表生成词向量
#         '''
#         if not os.path.exists(self.path + "/glove_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
#             # 与word2vec不一样的是glove文件是txt格式，要先转换为word2vec格式
#             # 这个转换过程比较慢，所以转换好就先保存，下次直接读。
#             if not os.path.exists(self.path + "/test_word2vec.txt"):
#                 glove_file = datapath(self.path + '/glove.840B.300d.txt')
#                 # 指定转化为word2vec格式后文件的位置
#                 tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")
#                 from gensim.scripts.glove2word2vec import glove2word2vec
#                 glove2word2vec(glove_file, tmp_file)
#             else:
#                 tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")
#             print("Reading Glove Embedding...")
#             # 注意这里的binary=True不用写。
#             wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
#
#             # 求词向量均值和方差
#             # 论文中提到，用方差对未知词进行初始化对于训练词向量的效果很不错
#             tmp = []
#             for word, index in self.word2id.items():
#                 try:
#                     tmp.append(wvmodel.get_vector(word))
#                 except:
#                     pass
#             mean = np.mean(np.array(tmp))
#             std = np.std(np.array(tmp))
#             print(mean, std)
#             # 用上面的词向量均值和方差来生成词向量
#             vocab_size = self.n_vocab
#             embed_size = 300
#             embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正态分布初始化方法
#             for word, index in self.word2id.items():
#                 try:
#                     # 如果预训练词向量中有对应的词就使用预训练的词向量，否则就用正态分布初始化的词向量
#                     embedding_weights[index, :] = wvmodel.get_vector(word)
#                 except:
#                     pass
#             # 由于每次读取这个东西很费时，所以处理好后保存下来，下次直接读取
#             np.save(self.path + "/glove_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
#         else:
#             embedding_weights = np.load(self.path + "/glove_embedding_mr.npy")  # 载入生成的词向量
#         self.weight = embedding_weights
#
#     def get_word2vec(self):
#         '''
#         生成word2vec词向量
#         :return: 根据词表生成的词向量
#         '''
#         if not os.path.exists(self.path + "/word2vec_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
#             print("Reading word2vec Embedding...")
#             # 加载预训练的Word2Vec词向量
#             wvmodel = KeyedVectors.load_word2vec_format(self.path + "/GoogleNews-vectors-negative300.bin.gz",
#                                                         binary=True)
#             tmp = []
#             for word, index in self.word2id.items():
#                 try:
#                     tmp.append(wvmodel.get_vector(word))
#                 except:
#                     pass
#             mean = np.mean(np.array(tmp))
#             std = np.std(np.array(tmp))
#             print(mean, std)
#             vocab_size = self.n_vocab
#             embed_size = 300
#             embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正太分布初始化方法
#             for word, index in self.word2id.items():
#                 try:
#                     embedding_weights[index, :] = wvmodel.get_vector(word)
#                 except:
#                     pass
#             np.save(self.path + "/word2vec_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
#         else:
#             embedding_weights = np.load(self.path + "/word2vec_embedding_mr.npy")  # 载入生成的词向量
#         self.weight = embedding_weights


class MR_Dataset(data.Dataset):
    def __init__(self, state="train", k=0, embedding_type="word2vec"):

        self.path = os.path.abspath('.')
        if "data" not in self.path:
            self.path += "/data"
        # 数据集加载，每行都是一句话
        self.k = k
        self.state = state
        self.embedding_type = embedding_type

    def get_data_ym(self, pos_samples, neg_samples, pos_state, neg_state):

        if pos_samples == []:
            raise
        if neg_samples == []:
            raise
        if pos_state == []:
            raise
        if neg_state == []:
            raise

        state_data = pos_state + neg_state
        # 把正负样本放在一块
        datas = pos_samples + neg_samples
        # datas = [nltk.word_tokenize(data) for data in datas]
        # 用空格进行分词
        datas = [data.split() for data in datas]
        # 求句子最大长度，将所有句子pad成一样的长度；
        # 有时也可以pad为平均长度，如果使用平均长度，对于超过长度的句子需要截断。
        max_sample_length = max([len(sample) for sample in datas])
        # 为正负样本设置标签
        labels = [1] * len(pos_samples) + [0] * len(neg_samples)
        word2id = {"<pad>": 0}  # 生成word2id，pad对应0
        # datases = [[] for data in datas]

        for i, data in enumerate(datas):
            for j, word in enumerate(data):
                # 词不还没加入word2id中则加入，并设置其id
                if word2id.get(word) is None:
                    word2id[word] = len(word2id)
                # 设置每个句子中所有词，替换为ID
                datas[i][j] = word2id[word]
            # 将句子按最大长度进行pad
            datas[i] = datas[i] + [0] * (max_sample_length - len(datas[i]))
            # datases[i].append(datas[i])
            # datases[i].append(state_data[i])

            # 如果是按平均长度则按下面语句进行截断或补齐,max_sample_length代表平均长度
            # datas[i] = datas[i][0:max_sample_length]+[0]*(max_sample_length-len(datas[i]))
        self.n_vocab = len(word2id)
        self.word2id = word2id

        #根据配置取不同的预训练词向量
        # if self.embedding_type == "word2vec":
        #     self.get_word2vec()
        # elif self.embedding_type == "glove":
        #     self.get_glove_embedding()
        # else:
        #     pass
        # self.get_word2vec()
        # 由于训练集中的数据前半部分是正样本，后半部分是负样本，需要打乱训练集
        # 把数据和标签放到一起打乱
        c = list(zip(datas, state_data, labels))  # 打乱训练集
        random.seed(2)
        random.shuffle(c)
        # 再把数据和标签分开
        datas[:], state_data[:], labels[:] = zip(*c)

        # 生成训练集、验证集和测试集
        # 总的数据分成10份，第k份作为测试集，其他9份再分
        # 其他9分的后10%做为验证集，前90%做为训练集
        if self.state == "train":  # 生成训练集
            # 取除第k份外其他9份
            self.datas = datas[:int(self.k * len(datas) / 10)] + datas[int((self.k + 1) * len(datas) / 10):]
            self.state_datas = state_data[:int(self.k * len(state_data) / 10)] + state_data[int((self.k + 1) * len(state_data) / 10):]

            self.labels = labels[:int(self.k * len(datas) / 10)] + labels[int((self.k + 1) * len(labels) / 10):]
            # 取前90%做为训练集
            self.datas = np.array(self.datas[0:int(0.9 * len(self.datas))])
            self.state_datas = np.array(self.state_datas[0:int(0.9 * len(self.state_datas))])
            self.labels = np.array(self.labels[0:int(0.9 * len(self.labels))])
        elif self.state == "valid":  # 生成验证集
            # 取除第k份外其他9份
            self.datas = datas[:int(self.k * len(datas) / 10)] + datas[int((self.k + 1) * len(datas) / 10):]
            self.state_datas = state_data[:int(self.k * len(state_data) / 10)] + state_data[int((self.k + 1) * len(state_data) / 10):]
            self.labels = labels[:int(self.k * len(datas) / 10)] + labels[int((self.k + 1) * len(labels) / 10):]
            # 取后10%做为验证集
            self.datas = np.array(self.datas[int(0.9 * len(self.datas)):])
            self.state_datas = np.array(self.state_datas[int(0.9 * len(self.state_datas)):])
            self.labels = np.array(self.labels[int(0.9 * len(self.labels)):])
        elif self.state == "test":  # 生成测试集
            # 第k份作为测试集
            self.datas = np.array(datas[int(self.k * len(datas) / 10):int((self.k + 1) * len(datas) / 10)])
            self.state_datas = np.array(state_data[int(self.k * len(state_data) / 10):int((self.k + 1) * len(state_data) / 10)])
            self.labels = np.array(labels[int(self.k * len(datas) / 10):int((self.k + 1) * len(datas) / 10)])

    def __getitem__(self, index):
        return self.datas[index], self.state_datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)

    def get_glove_embedding(self):
        '''
        生成glove词向量
        :return: 根据词表生成词向量
        '''
        if not os.path.exists(self.path + "/glove_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
            # 与word2vec不一样的是glove文件是txt格式，要先转换为word2vec格式
            # 这个转换过程比较慢，所以转换好就先保存，下次直接读。
            if not os.path.exists(self.path + "/test_word2vec.txt"):
                glove_file = datapath(self.path + '/glove.840B.300d.txt')
                # 指定转化为word2vec格式后文件的位置
                tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")
                from gensim.scripts.glove2word2vec import glove2word2vec
                glove2word2vec(glove_file, tmp_file)
            else:
                tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")
            print("Reading Glove Embedding...")
            # 注意这里的binary=True不用写。
            wvmodel = KeyedVectors.load_word2vec_format(tmp_file)

            # 求词向量均值和方差
            # 论文中提到，用方差对未知词进行初始化对于训练词向量的效果很不错
            tmp = []
            for word, index in self.word2id.items():
                try:
                    tmp.append(wvmodel.get_vector(word))
                except:
                    pass
            mean = np.mean(np.array(tmp))
            std = np.std(np.array(tmp))
            print(mean, std)
            # 用上面的词向量均值和方差来生成词向量
            vocab_size = self.n_vocab
            embed_size = 300
            embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正态分布初始化方法
            for word, index in self.word2id.items():
                try:
                    # 如果预训练词向量中有对应的词就使用预训练的词向量，否则就用正态分布初始化的词向量
                    embedding_weights[index, :] = wvmodel.get_vector(word)
                except:
                    pass
            # 由于每次读取这个东西很费时，所以处理好后保存下来，下次直接读取
            np.save(self.path + "/glove_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
        else:
            embedding_weights = np.load(self.path + "/glove_embedding_mr.npy")  # 载入生成的词向量
        self.weight = embedding_weights

    def get_word2vec(self):
        '''
        生成word2vec词向量
        :return: 根据词表生成的词向量
        '''
        if not os.path.exists(self.path + "/word2vec_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
            print("Reading word2vec Embedding...")
            # 加载预训练的Word2Vec词向量
            wvmodel = KeyedVectors.load_word2vec_format(self.path + "/GoogleNews-vectors-negative300.bin.gz",
                                                        binary=True)
            tmp = []
            for word, index in self.word2id.items():
                try:
                    tmp.append(wvmodel.get_vector(word))
                except:
                    pass
            mean = np.mean(np.array(tmp))
            std = np.std(np.array(tmp))
            print(mean, std)
            vocab_size = self.n_vocab
            embed_size = 300
            embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正太分布初始化方法
            for word, index in self.word2id.items():
                try:
                    embedding_weights[index, :] = wvmodel.get_vector(word)
                except:
                    pass
            np.save(self.path + "/word2vec_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
        else:
            embedding_weights = np.load(self.path + "/word2vec_embedding_mr.npy")  # 载入生成的词向量
        self.weight = embedding_weights


class MR_Dataset_test(data.Dataset):
    def __init__(self, state="train", k=0, embedding_type="word2vec"):

        self.path = os.path.abspath('.')
        if "data" not in self.path:
            self.path += "/data"
        # 数据集加载，每行都是一句话
        self.k = k
        self.state = state
        self.embedding_type = embedding_type

    def get_data_ym(self, pos_samples, neg_samples, pos_state, neg_state):

        if pos_samples == []:
            raise
        # if neg_samples == []:
        #     raise
        if pos_state == []:
            raise
        # if neg_state == []:
        #     raise

        state_data = pos_state + neg_state
        # 把正负样本放在一块
        datas = pos_samples + neg_samples
        # datas = [nltk.word_tokenize(data) for data in datas]
        # 用空格进行分词
        datas = [data.split() for data in datas]
        # 求句子最大长度，将所有句子pad成一样的长度；
        # 有时也可以pad为平均长度，如果使用平均长度，对于超过长度的句子需要截断。
        max_sample_length = max([len(sample) for sample in datas])
        # 为正负样本设置标签
        labels = [1] * len(pos_samples) + [0] * len(neg_samples)
        word2id = {"<pad>": 0}  # 生成word2id，pad对应0
        # datases = [[] for data in datas]

        for i, data in enumerate(datas):
            for j, word in enumerate(data):
                # 词不还没加入word2id中则加入，并设置其id
                if word2id.get(word) is None:
                    word2id[word] = len(word2id)
                # 设置每个句子中所有词，替换为ID
                datas[i][j] = word2id[word]
            # 将句子按最大长度进行pad
            datas[i] = datas[i] + [0] * (max_sample_length - len(datas[i]))
            # datases[i].append(datas[i])
            # datases[i].append(state_data[i])

            # 如果是按平均长度则按下面语句进行截断或补齐,max_sample_length代表平均长度
            # datas[i] = datas[i][0:max_sample_length]+[0]*(max_sample_length-len(datas[i]))
        self.n_vocab = len(word2id)
        self.word2id = word2id

        #根据配置取不同的预训练词向量
        # if self.embedding_type == "word2vec":
        #     self.get_word2vec()
        # elif self.embedding_type == "glove":
        #     self.get_glove_embedding()
        # else:
        #     pass
        # self.get_word2vec()
        # 由于训练集中的数据前半部分是正样本，后半部分是负样本，需要打乱训练集
        # 把数据和标签放到一起打乱
        c = list(zip(datas, state_data, labels))  # 打乱训练集
        random.seed(2)
        random.shuffle(c)
        # 再把数据和标签分开
        datas[:], state_data[:], labels[:] = zip(*c)

        # 生成训练集、验证集和测试集
        # 总的数据分成10份，第k份作为测试集，其他9份再分
        # 其他9分的后10%做为验证集，前90%做为训练集
        if self.state == "train":  # 生成训练集
            # 取除第k份外其他9份
            self.datas = datas[:int(self.k * len(datas) / 10)] + datas[int((self.k + 1) * len(datas) / 10):]
            self.state_datas = state_data[:int(self.k * len(state_data) / 10)] + state_data[int((self.k + 1) * len(state_data) / 10):]

            self.labels = labels[:int(self.k * len(datas) / 10)] + labels[int((self.k + 1) * len(labels) / 10):]
            # 取前90%做为训练集
            self.datas = np.array(self.datas[0:int(0.9 * len(self.datas))])
            self.state_datas = np.array(self.state_datas[0:int(0.9 * len(self.state_datas))])
            self.labels = np.array(self.labels[0:int(0.9 * len(self.labels))])
        elif self.state == "valid":  # 生成验证集
            # 取除第k份外其他9份
            self.datas = datas[:int(self.k * len(datas) / 10)] + datas[int((self.k + 1) * len(datas) / 10):]
            self.state_datas = state_data[:int(self.k * len(state_data) / 10)] + state_data[int((self.k + 1) * len(state_data) / 10):]
            self.labels = labels[:int(self.k * len(datas) / 10)] + labels[int((self.k + 1) * len(labels) / 10):]
            # 取后10%做为验证集
            self.datas = np.array(self.datas[int(0.9 * len(self.datas)):])
            self.state_datas = np.array(self.state_datas[int(0.9 * len(self.state_datas)):])
            self.labels = np.array(self.labels[int(0.9 * len(self.labels)):])
        elif self.state == "test":  # 生成测试集
            # 第k份作为测试集
            self.datas = np.array(datas[:])
            self.state_datas = np.array(state_data[:])
            self.labels = np.array(labels[:])

    def __getitem__(self, index):
        return self.datas[index], self.state_datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)

    def get_glove_embedding(self):
        '''
        生成glove词向量
        :return: 根据词表生成词向量
        '''
        if not os.path.exists(self.path + "/glove_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
            # 与word2vec不一样的是glove文件是txt格式，要先转换为word2vec格式
            # 这个转换过程比较慢，所以转换好就先保存，下次直接读。
            if not os.path.exists(self.path + "/test_word2vec.txt"):
                glove_file = datapath(self.path + '/glove.840B.300d.txt')
                # 指定转化为word2vec格式后文件的位置
                tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")
                from gensim.scripts.glove2word2vec import glove2word2vec
                glove2word2vec(glove_file, tmp_file)
            else:
                tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")
            print("Reading Glove Embedding...")
            # 注意这里的binary=True不用写。
            wvmodel = KeyedVectors.load_word2vec_format(tmp_file)

            # 求词向量均值和方差
            # 论文中提到，用方差对未知词进行初始化对于训练词向量的效果很不错
            tmp = []
            for word, index in self.word2id.items():
                try:
                    tmp.append(wvmodel.get_vector(word))
                except:
                    pass
            mean = np.mean(np.array(tmp))
            std = np.std(np.array(tmp))
            print(mean, std)
            # 用上面的词向量均值和方差来生成词向量
            vocab_size = self.n_vocab
            embed_size = 300
            embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正态分布初始化方法
            for word, index in self.word2id.items():
                try:
                    # 如果预训练词向量中有对应的词就使用预训练的词向量，否则就用正态分布初始化的词向量
                    embedding_weights[index, :] = wvmodel.get_vector(word)
                except:
                    pass
            # 由于每次读取这个东西很费时，所以处理好后保存下来，下次直接读取
            np.save(self.path + "/glove_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
        else:
            embedding_weights = np.load(self.path + "/glove_embedding_mr.npy")  # 载入生成的词向量
        self.weight = embedding_weights

    def get_word2vec(self):
        '''
        生成word2vec词向量
        :return: 根据词表生成的词向量
        '''
        if not os.path.exists(self.path + "/word2vec_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
            print("Reading word2vec Embedding...")
            # 加载预训练的Word2Vec词向量
            wvmodel = KeyedVectors.load_word2vec_format(self.path + "/GoogleNews-vectors-negative300.bin.gz",
                                                        binary=True)
            tmp = []
            for word, index in self.word2id.items():
                try:
                    tmp.append(wvmodel.get_vector(word))
                except:
                    pass
            mean = np.mean(np.array(tmp))
            std = np.std(np.array(tmp))
            print(mean, std)
            vocab_size = self.n_vocab
            embed_size = 300
            embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正太分布初始化方法
            for word, index in self.word2id.items():
                try:
                    embedding_weights[index, :] = wvmodel.get_vector(word)
                except:
                    pass
            np.save(self.path + "/word2vec_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
        else:
            embedding_weights = np.load(self.path + "/word2vec_embedding_mr.npy")  # 载入生成的词向量
        self.weight = embedding_weights

def class_2(pos_samples, neg_samples, pos_state, neg_state, index=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    state_y = 'train'

    trainset = MR_Dataset(state_y)
    trainset.get_data_ym(pos_samples, neg_samples, pos_state, neg_state)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    state_y = 'test'
    testset = MR_Dataset(state_y)
    testset.get_data_ym(pos_samples, neg_samples, pos_state, neg_state)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=True, num_workers=2)

    # state_y = 'valid'
    # validset = MR_Dataset(state_y)
    # validloader = torch.utils.data.DataLoader(
    #     validset, batch_size=128, shuffle=True, num_workers=2)

    # Model

    print('==> Building model..')
    from model import TextCNN
    filter_sizes = "1 2 3 4 5"
    filter_sizes = [int(val) for val in filter_sizes.split()]
    net = TextCNN(300, 200, filter_sizes,
                  128, 0.4, trainset.n_vocab)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    resume = False
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x,state, targets) in enumerate(trainloader):
            # todo  gai
            # x, state = inputs
            x, state, targets = x.to(device=device), state.to(device=device, dtype=torch.float), targets.to(device=device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = net(x, state)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            writer.add_scalar('data/loss', train_loss, batch_idx)
            writer.add_scalar('data/acc', 100.*correct/total, batch_idx)

            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, state, targets) in enumerate(testloader):
                # todo gai
                # x, state = inputs
                x, state, targets = x.to(device=device), state.to(device=device, dtype=torch.float), targets.to(device=device, dtype=torch.long)
                outputs = net(x, state)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt50000_20230810.pth')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()


def main(args):
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']

    train_fenlei = False
    ##

    use_pick_model = True
    load_mode = True

    env = ClevrEnv(action_type='discrete',)

    # program_generator, _ = utils.load_program_generator(args.program_generator)
    # execution_engine, _ = utils.load_execution_engine(args.execution_engine, verbose=False)
    # model = (program_generator, execution_engine)
    # model = utils.load_baseline(args.baseline_model)

    input_size = env.observation_space.shape  # (64,64,3)
    output_size = env.action_space.n  # 4

    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    icm_path = 'models/{}.icm'.format(env_id)



    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    save_video = default_config.getboolean('SaveVideo')
    video_interval = 500
    save_dir = 'videos-epoch3-gt'

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    eta = float(default_config['ETA'])

    clip_grad_norm = float(default_config['ClipGradNorm'])

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 3, 64, 64)) #???

    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(gamma)

    agent = ICMAgent

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        eta=eta,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )
    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    #torch.save(agent.model.state_dict(), model_path)
    #torch.save(agent.icm.state_dict(), icm_path)

    # set a fixed goal
    goal_text = 'There is a blue rubber sphere; are there any green rubber spheres to the left of it?'
    # goal_program = [{'type': 'scene', 'inputs': []},
    # {'type': 'filter_color', 'inputs': [0],
    # 'side_inputs': ['blue']},
    # {'type': 'filter_material', 'inputs': [1], 'side_inputs': ['rubber']},
    # {'type': 'filter_shape', 'inputs': [2], 'side_inputs': ['sphere']},
    # {'type': 'exist', 'inputs': [3]}, {'type': 'relate', 'inputs': [3], 'side_inputs': ['left']},
    # {'type': 'filter_color', 'inputs': [5], 'side_inputs': ['green']},
    # {'type': 'filter_material', 'inputs': [6], 'side_inputs': ['rubber']},
    # {'type': 'filter_shape', 'inputs': [7], 'side_inputs': ['sphere']},
    # {'type': 'exist', 'inputs': [8]}]
    goal_program = [{'type': 'scene', 'inputs': []},
  {'type': 'filter_color', 'inputs': [0], 'side_inputs': ['purple']},
  {'type': 'filter_material', 'inputs': [1], 'side_inputs': ['rubber']},
  {'type': 'filter_shape', 'inputs': [2], 'side_inputs': ['sphere']},
  {'type': 'exist', 'inputs': [3]},
  {'type': 'relate', 'inputs': [3], 'side_inputs': ['right']},
  {'type': 'filter_color', 'inputs': [5], 'side_inputs': ['red']},
  {'type': 'filter_shape', 'inputs': [6], 'side_inputs': ['sphere']},
  {'type': 'exist', 'inputs': [7]}]

    env.set_goal(goal_text, goal_program)

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    num_iterations = 0

    next_obs = []
    steps = 0

    # introducing language objective
    counter_questions = {}
    total_questions = []
    all_question_ym = []
    threshold = 0.9
    max_ques = 256
    pos_question = []
    neg_question = []
    pos_state = []
    neg_state = []

    while len(counter_questions) < max_ques:
        goal_text, goal_program = env.sample_goal()
        if goal_text not in counter_questions:
            total_questions.append(goal_text)
            all_question_ym.append(goal_text)
        counter_questions[goal_text] = [0, goal_program]

    questions_set = total_questions[:num_step]
    total_questions = total_questions[num_step:]
    end_training = 0

    data_num = 0
    data_num_neg = 0

    while True:

        class_batch_size = []

        if global_update%100 == 0:
            print("ENV GOAL: ", env.current_goal_text)
            print("ENV GOAL PROGRAM: ", env.current_goal)

        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_log_prob, total_policy, total_int_ext  = \
            [], [], [], [], [], [], [], [], [], [], []
        all_frames = []
        global_step += num_step
        global_update += 1

        acc_step = 0

        states = env.reset()
        while env.answer_question(env.current_goal) == 1:
            states = env.reset()
        states = states.reshape(1, 3, 64, 64)

        eps_reward = 0

        # shuffling question set
        sample_range = np.arange(num_step)
        np.random.shuffle(sample_range)

        num_episodes_pre_update = 0

        account = 0
        tot = 0

        s1 = time.time()

        program2=0
        while num_episodes_pre_update < 10:

            # tot += 1

            num_episodes_pre_update += 1
            print('Starting rollout no. {}'.format(num_episodes_pre_update))
            # Step 1. n-step rollout
            t1 = time.time()
            for idx in range(num_step):
                print("idx",idx)
                num_iterations += 1
                if idx%10==0:

                #all_frames.append(pad_image(env.render(mod e='rgb_array')))

                    tt1 = time.time()
                    if use_pick_model == True:
                        b = MR_Dataset_test('test')
                        tt2 = time.time()
                        #print("b_MRdataset time is ", tt2 - tt1)
                        state_list = [states.squeeze(0) for i in range(len(all_question_ym))]
                        b.get_data_ym(all_question_ym, [], state_list, [])
                        tt3 = time.time()
                        #print("get_data_ym time is ", tt3 - tt2)
                        testloader = torch.utils.data.DataLoader(
                            b, batch_size=len(all_question_ym)+1, shuffle=True, num_workers=1)
                        tt4 = time.time()
                        #print("testloader time is ", tt4-tt3)

                        if load_mode:
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            from model import TextCNN
                            filter_sizes = "1 2 3 4 5"
                            filter_sizes = [int(val) for val in filter_sizes.split()]
                            net = TextCNN(300, 200, filter_sizes,
                                        128, 0.4, b.n_vocab)

                            net = net.to(device)
                            if device == 'cuda':
                                net = torch.nn.DataParallel(net)
                                cudnn.benchmark = True

                            checkpoint = torch.load('/home/guozhourui/language-curiosity/checkpoint_bak/checkpoint/ckpt10000.pth')
                            net.load_state_dict(checkpoint['net'])
                            print("load classer done")
                            load_mode = False

                            tt5 = time.time()
                            print("load model all time is ", tt5-tt4)
                        tt5 = time.time()
                        with torch.no_grad():
                            for batch_idx, (x, state, targets) in enumerate(testloader):
                                #print("yao batch_idx is ", batch_idx)
                                # print(state.shape)
                                # todo gai
                                # x, state = inputs
                                ttt1 = time.time()
                                x, state, targets = x.to(device=device), state.to(device=device, dtype=torch.float), targets.to(
                                    device=device, dtype=torch.long)
                                ttt2 = time.time()
                                #print("ttt2 is ", ttt2-ttt1)
                                outputs = net(x, state)
                                ttt3 = time.time()
                                #print("ttt3 is ", ttt3 - ttt2)
                                _, predicted = outputs.max(1)
                                ttt4 = time.time()
                                #print("ttt4 is ", ttt4 -ttt3)
                                # print(predicted)
                                p = predicted.cpu().data.numpy()
                                ttt5 = time.time()
                                #print("ttt5 is ", ttt5 - ttt4)
                                #print(p)
                                #print(len(p))
                            tt6 = time.time()
                            print("forward all time is ", tt6 - tt5)
                                # p2 = predicted.cpu().numpy()
                                # print(p)
                                # print(p2)


                        if np.all(p==0):
                            print("no user classfier")
                            sample_idx = sample_range[idx]
                            question = questions_set[sample_idx]
                            program = counter_questions[question][1]
                            ans_pre_step = env.answer_question(program)
                        else:
                            print("user classfier done")
                            x = np.where(p == 1)
                            #print("x[0]",x[0])
                            #print("x[0][0]",x[0][0])
                            question = all_question_ym[x[0][0]]
                            program = counter_questions[question][1]
                            if x[0].size>1:
                            #print("x[0][1]",x[0][1])
                                question2 = all_question_ym[x[0][1]]
                                program2 = counter_questions[question2][1]
                else:
                    sample_idx = sample_range[idx]
                    question = questions_set[sample_idx]
                    program = counter_questions[question][1]
                    ans_pre_step = env.answer_question(program)


                #ans_pre_step = run_single_example(args, model, question, states.reshape(64, 64, 3))
                #print("QUESTION: ", question)
                #print("ANS PRE STEP: ", ans_pre_step)

                actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

                # for parent_conn, action in zip(parent_conns, actions):
                #     parent_conn.send(action)
                #
                # next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                # for parent_conn in parent_conns:
                #     s, r, d, rd, lr = parent_conn.recv()
                #     next_states.append(s)
                #     rewards.append(r)
                #     dones.append(d)
                #     real_dones.append(rd)
                #     log_rewards.append(lr)

                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                episode_over = 0

                for action in actions:
                   # act = 2*np.random.rand(4) - 1
                   # act = env.sample_random_action()
                    s, r, d, info = env.step(action, record_achieved_goal = True)
                    episode_over = r
                    eps_reward += r
                    #achieved_goal_text = env.get_achieved_goals()
                    #print("ACHIEVED: ", achieved_goal_text)
                    s = s.reshape(1, 3, 64, 64)
                    next_states = s
                    rewards.append(r)
                    dones = d

                ans_post_step = env.answer_question(program)
                #if program2:
                #    ans_post_step2 = env.answer_question(program2)
                #else: ans_post_step2=ans_pre_step
                #ans_post_step = run_single_example(args, model, question, next_states.reshape(64, 64, 3))
                #print("ANS POST STEP: ", ans_post_step)

                # next_states = np.stack(next_states)
                # rewards = np.hstack(rewards)
                # dones = np.hstack(dones)

                # total reward = int reward

                writer.add_scalar('data/reward_per_step', episode_over, num_iterations)
                intrinsic_reward = agent.compute_intrinsic_reward(
                    (states - obs_rms.mean) / np.sqrt(obs_rms.var),
                    (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                    actions)
        
                #or ans_pre_step != ans_post_step2
                if ans_pre_step != ans_post_step :
                    data_num += 1
                    pos_question.append(question)
                    pos_state.append(states.squeeze(0))
                    # print("Pre and post answer change")
                    # print("data num is ", data_num, )
                    intrinsic_reward += 10
                    counter_questions[question][0] += 1
                    if counter_questions[question][0]/(sample_episode+1) > threshold:
                        if len(total_questions) > 0:
                            questions_set[sample_idx] = total_questions[0]
                            total_questions.pop(0)
                        else:
                            end_training = 1
                            break

                    if train_fenlei == True:
                        if len(pos_question) > 50000:
                            class_2(pos_question, neg_question, pos_state, neg_state)
                            train_fenlei = False

                else:
                    data_num_neg += 1
                    # print("data num is ", data_num, "data num neg is ", data_num_neg, "len pos question is ", len(pos_question))
                    neg_question.append(question)
                    neg_state.append(states.squeeze(0))

                    if train_fenlei == True:
                        if len(pos_question) > 50000:
                            class_2(pos_question, neg_question, pos_state, neg_state)
                            train_fenlei = False




                #print('intrinsic:{}'.format(intrinsic_reward))
                # print('val:{}'.format(value))
                sample_i_rall += intrinsic_reward[sample_env_idx]

                total_int_reward.append(intrinsic_reward)
                total_state.append(states)
                total_next_state.append(next_states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_values.append(value)
                #total_log_prob.append(log_prob)
                total_policy.append(policy)

                states = next_states[:, :, :, :]

                sample_rall += rewards[0]
                sample_step += 1
                if sample_step >= num_step or episode_over:
                    tot += 1
                    if episode_over:
                        account += 1
                    sample_episode += 1
                    writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                    writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                    writer.add_scalar('data/step', sample_step, sample_episode)
                    print("Episode: %d Sum of rewards: %.2f. Length: %d." % (sample_episode, sample_rall, sample_step))
                    obs = env.reset()
                    while env.answer_question(env.current_goal) == 1:
                        obs = env.reset()
                    obs = obs.reshape(1, 3, 64, 64)
                    states = obs
                    sample_rall = 0
                    sample_step = 0
                    sample_i_rall = 0
                    # break

                # sample_rall += log_rewards[sample_env_idx]
                #
                # sample_step += 1
                # if real_dones[sample_env_idx]:
                #     sample_episode += 1
                #     writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                #     writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                #     writer.add_scalar('data/step', sample_step, sample_episode)
                #     sample_rall = 0
                #     sample_step = 0
                #     sample_i_rall = 0
            t2 = time.time()
            print("num_step time is ", t2-t1 )
            # language model exploited by agent -> end training
        s2 = time.time()

        print("while num_episodes_pre_update time is ", s2-s1)

        acc_step += 1
        accg = account / tot
        writer.add_scalar('data/aa：cg', accg, global_update)
        print("data/aacg is", accg, "acc_step is", acc_step)
        with open('10step_seed2.csv','a',newline='') as file:
            a_writer = csv.writer(file)
            a_writer.writerow([accg])
        print("tot is ", tot)
        print("account is ", account)

        if end_training:
            print('Ending training')
            break
        # if train_fenlei == True:
        #     if len(pos_question) > 10:
        #         class_2(pos_question, neg_question, pos_state, neg_state)
        #         train_fenlei = False

        # calculate last next value
        _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))
        total_values.append(value)
        # --------------------------------------------------
        # Save video
       # if save_video and global_update % video_interval == 0:
        #    video_dir = os.path.join(save_dir, 'episode_{}.mp4'.format(global_update))
         #   print('Saving video to {}'.format(video_dir))
          #  save_video_file(np.uint8(all_frames), video_dir, fps=5)
           # print('Video saved...')

        #if save_video and eps_reward !=0 and len(total_reward) > 1: #policy improving
         #   video_dir = os.path.join(save_dir, 'reward_episode_{}.mp4'.format(global_update))
         #   print('Saving reward episode to {}'.format(video_dir))
         #   save_video_file(np.uint8(all_frames), video_dir, fps=5)

        # --------------------------------------------------
        total_reward = np.stack(total_reward).transpose()
        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 3, 64, 64])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 3, 64, 64])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_values = np.stack(total_values).transpose()
        # total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        # writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        
        target, adv = make_train_data(total_reward + total_int_reward,
                                      np.zeros_like(total_int_reward),
                                      total_values,
                                      gamma,
                                      num_step*num_episodes_pre_update, #num_step,
                                      num_worker)

        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        # -----------------------------------------------

        # Step 5. Training!
        agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                           (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                           target, total_action,
                           adv,
                           total_policy)

        '''
        agent.train_model_continuous((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          target, total_action,
                          adv,
                          total_log_prob,
                          global_update)
    
        agent.clear_actions()
        '''

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.icm.state_dict(), icm_path)

    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
