import tensorflow as tf
import datetime
from utils.model_ResNet18 import ResNet18

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

class ResNet18Trainer():
    def __init__(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=256, lr=1e-3):
        self.X_train = X_train.astype("float32")
        self.y_train = y_train.astype("float32")
        self.X_val = X_val.astype("float32")
        self.y_val = y_val.astype("float32")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def init_model(self):
        self.model = ResNet18(num_classes=len(tf.unique(self.y_train)[0]))

    def init_loss(self):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_function(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train_epoch(self, epoch):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

        train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).shuffle(10000).batch(self.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val)).batch(self.batch_size)

        for images, labels in train_ds:
            self.train_step(images, labels)

        for test_images, test_labels in test_ds:
            self.test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                               self.train_loss.result(),
                               self.train_accuracy.result() * 100,
                               self.test_loss.result(),
                               self.test_accuracy.result() * 100))

    def run(self):
        self.init_model()
        self.init_loss()
        self.init_optimizer()

        for epoch in range(self.epochs):
            print(f"Training Epoch {epoch + 1}")
            self.train_epoch(epoch)
