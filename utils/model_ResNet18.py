import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Add
from tensorflow.keras import Model

class ResidualBlock(Model):
    def __init__(self, filters, strides=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding="same")
        self.bn1 = BatchNormalization()
        self.relu = ReLU()##########
        self.conv2 = Conv2D(filters, kernel_size=(3, 3), strides=1, padding="same")
        self.bn2 = BatchNormalization()

        self.downsample = downsample
        if self.downsample:
            self.shortcut = tf.keras.Sequential([
                Conv2D(filters, kernel_size=(1, 1), strides=strides),
                BatchNormalization()
            ])

    def call(self, x):
        shortcut = x
        if self.downsample:
            shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = Add()([x, shortcut])
        x = self.relu(x)
        return x

class ResNet18(Model):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=2, padding="same")
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")

        self.layer1 = self._build_resblock(64, 2, first_block=True)
        self.layer2 = self._build_resblock(128, 2, strides=2)
        self.layer3 = self._build_resblock(256, 2, strides=2)
        self.layer4 = self._build_resblock(512, 2, strides=2)

        self.global_avg_pool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation="softmax")

    def _build_resblock(self, filters, blocks, strides=1, first_block=False):
        res_blocks = []
        for i in range(blocks):
            if i == 0 and not first_block:
                res_blocks.append(ResidualBlock(filters, strides, downsample=True))
            else:
                res_blocks.append(ResidualBlock(filters))
        return tf.keras.Sequential(res_blocks)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x