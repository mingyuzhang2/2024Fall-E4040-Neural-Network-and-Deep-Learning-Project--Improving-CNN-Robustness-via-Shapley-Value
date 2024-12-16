import tensorflow as tf
import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from utils.model_ResNet18 import ResNet18, ResidualBlock
from tensorflow.keras.callbacks import LearningRateScheduler


mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
std = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)

def transform_train(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    
    image = (image - mean) / std
    
    return image, label

def transform_test(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - mean) / std
    
    return image, label

def load_cifar10_dataset(batch_size):
    (X_train, y_train), (X_val, y_val) = tf.keras.datasets.cifar10.load_data()
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(transform_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = test_ds.map(transform_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds




def learning_rate_schedule(epoch):
    if epoch >= 1 and epoch <= 120:
        return 0.1
    elif epoch >= 120 and epoch <= 180:
        return 0.01
    #elif epoch >= 180 and epoch <= 240:
    #    return 0.001
    else:
        return 0.001

lr_scheduler = LearningRateScheduler(learning_rate_schedule)

    
#class WarmUpScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
#    def __init__(self, initial_lr, warmup_epochs, total_epochs):
#        super(WarmUpScheduler, self).__init__()
#        self.initial_lr = initial_lr
#        self.warmup_epochs = warmup_epochs
#        self.total_epochs = total_epochs

#    def __call__(self, epoch):
#        if epoch < self.warmup_epochs:
#            return self.initial_lr * (epoch + 1) / self.warmup_epochs
#        else:
#            return self.initial_lr


#class ResNet18(Model):
#    def __init__(self, num_classes):
#        super(ResNet18, self).__init__()
#        self.conv1 = Conv2D(64, kernel_size=(3, 3), strides=2, padding="same")
#        self.bn1 = BatchNormalization()
#        self.relu = ReLU()
#        #self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")
#
#        self.layer1 = self._build_resblock(64, 2, first_block=True)
#        self.layer2 = self._build_resblock(128, 2, strides=2)
#        self.layer3 = self._build_resblock(256, 2, strides=2)
#        self.layer4 = self._build_resblock(512, 2, strides=2)

#        self.global_avg_pool = GlobalAveragePooling2D()
#        self.fc = Dense(num_classes, activation="softmax")

#    def _build_resblock(self, filters, blocks, strides=1, first_block=False):
#        res_blocks = []
#        for i in range(blocks):
#            if i == 0 and not first_block:
#                res_blocks.append(ResidualBlock(filters, strides, downsample=True))
#            else:
#                res_blocks.append(ResidualBlock(filters))
#        return tf.keras.Sequential(res_blocks)

#    def call(self, x):
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        x = self.maxpool(x)

#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)

#        x = self.global_avg_pool(x)
#        x = self.fc(x)
#        return x

class ResNet18_trainer():
    #def __init__(self, X_train, y_train, X_val, y_val, num_classes, epochs, batch_size, lr, momentum, decay):
    def __init__(self, train_ds, test_ds, num_classes, epochs, batch_size, lr, momentum, decay):
        
        #self.X_train = X_train.astype("float32")
        #self.y_train = y_train.astype("float32")
        #self.X_val = X_val.astype("float32")
        #self.y_val = y_val.astype("float32")
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

    def init_model(self):
        self.model = ResNet18(num_classes=self.num_classes)

    def init_loss(self):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def init_optimizer(self):
        # warm-up
        #initial_lr = 1e-3
        #warmup_epochs = 5
        #total_epochs = self.epochs
        #lr_scheduler = WarmUpScheduler(initial_lr, warmup_epochs, total_epochs)

        
        self.optimizer = tf.keras.optimizers.SGD(
        learning_rate=self.lr,
        momentum=self.momentum,
        decay=self.decay
    )

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
        
        new_lr = learning_rate_schedule(epoch)
        tf.keras.backend.set_value(self.optimizer.lr, new_lr)

        #train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).shuffle(10000).batch(self.batch_size)
        #test_ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val)).batch(self.batch_size)

        for images, labels in self.train_ds:
            self.train_step(images, labels)

        for test_images, test_labels in self.test_ds:
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
