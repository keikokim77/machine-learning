# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 20:39:32 2016

@author: yonezawakeiko
"""

#coding: utf-8
import os
import sys
import pickle
import struct
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10
# 80 million tiny imagesのサブセット
# Alex Krizhevsky, Vinod Nair, Geoffrey Hintonが収集
# 32x32のカラー画像60000枚
# 10クラスで各クラス6000枚
# 50000枚の訓練画像と10000枚（各クラス1000枚）のテスト画像
# クラスラベルは排他的
# PythonのcPickle形式で提供されている

DATA_DIR = "/Users/yonezawakeiko/work/CIFAR10_data/"
label_file = DATA_DIR + "batches.meta.txt"
data_file = DATA_DIR + "data_batch_1.bin"

def unpickle(f):
    #print(sys.version_info.major)
    fo = open(f, 'rb')
    d = pickle.load(fo, encoding='latin-1')
    #d = pickle.load(fo)
    fo.close()
    return d

# ラベル名をロード
fo = open(label_file)
label_names = fo.read()
#label_names = d["label_names"]
#print (label_names)
fo.close()

#label_names = unpickle(label_file)["label_names"]
#print (label_names)

# dataをロード
print (data_file)
n = 32 * 32 * 3 + 1
fo = open(data_file, 'rb')

nsamples = 500
labels = np.zeros(nsamples, dtype=np.uint8)
data = np.zeros((nsamples, 3072), dtype=np.uint8)
for i in range(nsamples):
  d = fo.read(n)
  #label = d[0]
  labels[i] = d[0]
  #print(label)
  #data = struct.unpack('3072B', d[1:])
  data[i] = struct.unpack('3072B', d[1:])
  #print(data)

# 各クラスの画像をランダムに10枚抽出して描画
nclasses = 10
pos = 1
for i in range(nclasses):
    # クラスiの画像のインデックスリストを取得
    targets = np.where(labels == i)[0]
    np.random.shuffle(targets)
    # 最初の10枚の画像を描画
    for idx in targets[:10]:
        plt.subplot(10, 10, pos)
        img = data[idx]
        # (channel, row, column) => (row, column, channel)
        plt.imshow(img.reshape(3, 32, 32).transpose(1, 2, 0))
        plt.axis('off')
        label = label_names[i]
        pos += 1
plt.show()