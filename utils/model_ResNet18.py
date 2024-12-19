import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Add, Dropout
from tensorflow.keras import Model

class ResidualBlock(Model):
    def __init__(self, filters, strides=1, downsample=False):
        super().__init__()
        self.conv1 = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding="same")#######use to be 7*7 but the image is small so will lead to overfitting or not useful
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(filters, kernel_size=(3, 3), strides=1, padding="same")#used to be max,but also the image is small
        self.bn2 = BatchNormalization()

        self.downsample = downsample# A flag to determine if downsampling is required for the shortcut path
        if self.downsample:# If downsampling is required, define a shortcut path using a sequence of layers
            self.shortcut = tf.keras.Sequential([
                Conv2D(filters, kernel_size=(1, 1), strides=2),
                BatchNormalization()
            ])

    def call(self, x):
        shortcut = x
        if self.downsample:# If downsampling is required, apply the shortcut transformation defined earlier
            shortcut = self.shortcut(x)
# Apply the downsampling shortcut (1x1 Conv + BatchNorm)
        x = self.conv1(x)# Pass the input through the first convolutional layer
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x += shortcut########### Add the shortcut connection to the output of the second convolution (residual connection)
        x = self.relu(x)
        return x

class ResNet18(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", 
                            kernel_regularizer=regularizers.l2(0.001))
        self.bn1 = BatchNormalization()
        self.relu = ReLU()

        self.layer1 = self._build_resblock(64, 2, first_block=True)###ResNet setting
        self.layer2 = self._build_resblock(128, 2, strides=2)
        self.layer3 = self._build_resblock(256, 2, strides=2)
        self.layer4 = self._build_resblock(512, 2, strides=2)

        self.global_avg_pool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.fc = Dense(num_classes, activation="softmax")###softmax
        self.dropout = Dropout(0.5)

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
       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)###dont forget
        x = self.dropout(x)
        x = self.fc(x)
        return x
