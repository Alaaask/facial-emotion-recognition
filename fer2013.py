import os
import csv
import random
import numpy as np
from PIL import Image
from pylab import array
import matplotlib.pyplot as plt


class Fer2013(object):
    def __init__(self, datasetType):
        self.datasetType = datasetType
        self.batchSize = 20         # 批次数
        self.imageOrgSize = 48      # 输入图片尺寸
        self.imageSize = 42         # 裁剪后图片尺寸
        self.emotionsTypeNum = 7    # 表情类型数
        self.croppedNum = 8           # 裁剪次数
        self.data = []
        self.readCSVData()

    def readCSVData(self):
        csvPath = os.path.join(os.getcwd(), 'fer2013/',
                               'fer2013_'+self.datasetType + '.csv')
        csvReader = csv.reader(open(csvPath, encoding='utf-8'))
        for row in csvReader:
            self.data.append(row)

    def get(self):
        samples = random.sample(
            self.data, self.batchSize)  # 随机选择 batchSize 个数据
        images = np.zeros((self.batchSize * self.croppedNum,
                           self.imageSize, self.imageSize, 1))
        labels = np.zeros((self.batchSize * self.croppedNum,
                           self.emotionsTypeNum), dtype='int')

        for i in range(self.batchSize):
            pixels = samples[i][1].split(' ')  # 取得所有像素数据
            image = np.reshape(
                pixels, (self.imageOrgSize, self.imageOrgSize))  # 原始图像 整理为  48*48的矩阵

            for j in range(self.croppedNum):
                x = random.randint(0, self.imageOrgSize - self.imageSize)
                y = random.randint(0, self.imageOrgSize - self.imageSize)
                croppedImage = image[x:x+self.imageSize,
                                     y:y+self.imageSize]  # 整理为 42*42
                images[i*self.croppedNum+j] = np.reshape(
                    croppedImage, (self.imageSize, self.imageSize, 1))
                labels[i*self.croppedNum+j][int(samples[i][0])] = 1
        return images, labels

    def get_test(self):
        samples = random.sample(self.data, self.batchSize)

        images = np.zeros(
            (self.batchSize * 4, self.imageSize, self.imageSize, 1))
        labels = np.zeros((self.batchSize, self.emotionsTypeNum), dtype='int')

        for i in range(self.batchSize):
            pixels = samples[i][1].split(' ')  # 取得所有像素数据
            image = np.reshape(
                pixels, (self.imageOrgSize, self.imageOrgSize))

            leftTopImage = image[:self.imageSize, :self.imageSize]
            leftBottomImage = image[:self.imageSize,
                                    self.imageOrgSize-self.imageSize:]
            rightTopImage = image[self.imageOrgSize -
                                  self.imageSize:, :self.imageSize]
            rightBottomImage = image[self.imageOrgSize -
                                     self.imageSize:, self.imageOrgSize - self.imageSize:]
            images[i * 4 + 0] = np.reshape(leftTopImage,
                                           (self.imageSize, self.imageSize, 1))
            images[i * 4 + 1] = np.reshape(leftBottomImage,
                                           (self.imageSize, self.imageSize, 1))
            images[i * 4 + 2] = np.reshape(rightTopImage,
                                           (self.imageSize, self.imageSize, 1))
            images[i * 4 + 3] = np.reshape(rightBottomImage,
                                           (self.imageSize, self.imageSize, 1))
            # images[i * 4 + 4] = np.reshape([i.reverse() for i in left_top_image], (self.image_size, self.image_size, 1))
            # images[i * 4 + 5] = np.reshape([i.reverse() for i in left_bottom_image], (self.image_size, self.image_size, 1))
            # images[i * 4 + 6] = np.reshape([i.reverse() for i in right_top_image], (self.image_size, self.image_size, 1))
            # images[i * 4 + 7] = np.reshape([i.reverse() for i in right_bottom_image], (self.image_size, self.image_size, 1))
            labels[i][int(samples[i][0])] = 1
        return images, labels

fer2013 = Fer2013('train')