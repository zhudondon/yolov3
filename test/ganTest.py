from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
# from keras.optimizers import Adam
from keras.optimizer_v1 import Adam

import matplotlib.pyplot as plt

import sys
import os
import numpy as np

class GAN():
    def __init__(self):
        # --------------------------------- #
        #   行28，列28，也就是mnist的shape
        # --------------------------------- #
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        # 28,28,1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        # adam优化器 学习率和beta1（不知道什么意思）
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        # 指定 二项交叉熵损失，优化器使用adam
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # 构建 生成器
        self.generator = self.build_generator()
        # 生产输入随机值
        gan_input = Input(shape=(self.latent_dim,))
        # 生成图片
        img = self.generator(gan_input)
        # 在训练generate的时候不训练discriminator
        self.discriminator.trainable = False
        # 对生成的假图片进行预测
        validity = self.discriminator(img)
        # 输入输出 联合训练 得到最终损失
        self.combined = Model(gan_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):
        # --------------------------------- #
        #   生成器，输入一串随机数字
        # --------------------------------- #
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # ----------------------------------- #
        #   评价器，对输入进来的图片进行评价
        # ----------------------------------- #
        model = Sequential()
        # 输入一张图片
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # 判断真伪
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # 获得数据
        (X_train, _), (_, _) = mnist.load_data()

        # 进行标准化
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # 创建标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # 这里是同时训练3个网络，g网络 ，d网络 ，gd网络
            # 这里批次大小 一批128, 一轮128个图片,随机获取
            # --------------------------- #
            #   随机选取batch_size个图片
            #   对discriminator进行训练
            # --------------------------- #
            # randint(low, high=None, size=None, dtype=int) 最小值,最大值 生成的随机整数的上限（不包含）,形状
            # 生成的数组的形状，可以是一个整数（表示生成一个一维数组）或一个元组（表示生成多维数组）。
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # normal(loc=0.0, scale=1.0, size=None) loc 高斯概率分布的均值 ；scale 高斯概率分布的标准差；形状size(批次，潜在空间的维度)
            # latent_dim是一个重要的超参数，‌它影响着生成器能够生成的数据的复杂度和多样性。‌通过调整latent_dim的大小，‌可以控制生成器生成的样本的复杂度和细节程度
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 根据输入 g 预测值
            # predict(X,batch_size,) 基于批处理式的
            # predict()
            # 对数据进行批量循环(实际上，可以通过predict(x, batch_size=64)
            # 指定批量大小)，
            # 测试样本，类型为array-like或sp matrix，shape为[n_samples, n_features]。
            # 其中n_samples 表示样本的数量，n_features 表示特征的数量。
            # 这里就是 有128个样本，每个样本有100个特征
            # 返回值 Numpy array(s) of predictions. 返回多个预测，也就是多张图片
            gen_imgs = self.generator.predict(noise)

            # train_on_batch 批量训练，得到一个损失值，迭代，这个框架就可以减少 循环批次然后进行 训练的麻烦，写无效的循环代码
            # 这个train方法，是动态生成的
            # _make_train_function 调用这个 生成 训练方法
            # 训练过程大概是，收集需要更新的权重参数，loss损失值
            #  updates = self.optimizer.get_updates(
            #               params=self._collected_trainable_weights, loss=self.total_loss)
            # Gets loss and metrics. Updates weights at each call.
            # backend.function 通过这个方法来执行
            # def function(inputs, outputs, updates=None, name=None, **kwargs)
            # 输入，输出，更新的参数
            # GraphExecutionFunction(inputs, outputs, updates=updates, name=name, **kwargs)
            # GraphExecutionFunction
            #

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # --------------------------- #
            #  训练generator
            # --------------------------- #
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # 联合 训练 得到全局损失
            g_loss = self.combined.train_on_batch(noise, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs("./images")
    gan = GAN()
    gan.train(epochs=30000, batch_size=256, sample_interval=200)
