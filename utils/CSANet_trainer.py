import tensorflow as tf
import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from utils.model_CSANet import CSANet
from tensorflow.keras.callbacks import LearningRateScheduler
import os


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
    if epoch >= 1 and epoch <= 80:
        return 0.1
    else:
        return 0.01
lr_scheduler = LearningRateScheduler(learning_rate_schedule)




def generate_adversarial_example(model, image, label, epsilon=0.1):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
    
    gradients = tape.gradient(loss, image)
    adversarial_image = image + epsilon * tf.sign(gradients)
    adversarial_image = tf.clip_by_value(adversarial_image, 0.0, 1.0)
    
    return adversarial_image



class ResNet18_trainer():
    def __init__(self, train_ds, test_ds, num_classes, epochs, batch_size, lr, momentum, decay, checkpoint_dir, adv_train=False):
        
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.checkpoint_dir = checkpoint_dir
        
        
        self.adv_train = adv_train

    def init_model(self):
        self.model = ResNet18(num_classes=self.num_classes)
        self.model.build((None, 32, 32, 3))

    def init_loss(self):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def init_optimizer(self):
        
        self.optimizer = tf.keras.optimizers.SGD(
        learning_rate=self.lr,
        momentum=self.momentum,
        decay=self.decay
    )

    def train_step(self, images, labels):
        if self.adv_train:
            # Generate adversarial examples
            adv_images = generate_adversarial_example(self.model, images, labels)
            images = adv_images  # Replace original images with adversarial images

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
        
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch + 1}.h5")
        self.model.save_weights(checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    def load_checkpoint(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.model.load_weights(latest_checkpoint)
            print(f"Restored from checkpoint: {latest_checkpoint}")
            
            checkpoint_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])

           
            print(f"Restored from epoch {checkpoint_epoch}: ")
            
        else:
            print("No checkpoint found. Starting from scratch.")
            
    def delete_previous_checkpoints(self):
        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                file_path = os.path.join(self.checkpoint_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Deleted previous checkpoints.")
        
        

    def run(self):
        self.init_model()
        self.init_loss()
        self.init_optimizer()
        
        self.load_checkpoint()
        
       
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            epoch_start = int(latest_checkpoint.split('_')[-1].split('.')[0])
        else:
            epoch_start = 0 

        for epoch in range(epoch_start, self.epochs):
            print(f"Training Epoch {epoch + 1}")
            self.train_epoch(epoch)
            
            self.save_checkpoint(epoch)
