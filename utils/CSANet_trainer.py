import tensorflow as tf
import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from utils.model_CSANet import CSANet
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import random
import numpy as np
from utils.model_ResNet18 import ResNet18

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

def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=10):

    images_adv = tf.Variable(images, dtype=tf.float32)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(images_adv)
            predictions = model(images_adv, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, images_adv)
        signed_grad = tf.sign(gradients)
        images_adv.assign_add(alpha * signed_grad)
        perturbation = tf.clip_by_value(images_adv - images, -epsilon, epsilon)
        images_adv.assign(images + perturbation)

    return images_adv

def learning_rate_schedule(epoch):
    if epoch >= 1 and epoch <= 90:
        return 0.1
    elif epoch > 90 and epoch <= 120:
        return 0.01
    else:
        return 0.001

lr_scheduler = LearningRateScheduler(learning_rate_schedule)


class CSANet_trainer():
    def __init__(self, train_ds, test_ds, num_classes, epochs, batch_size, lr, momentum, decay, checkpoint_dir, pgd_attack, use_csa, backbone=None, conf_per_class=5000, conf_path=None, epsilon=0.1, pgd_iter=10, pgd_alpha=0.01):##########
        
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.checkpoint_dir = checkpoint_dir
        
        self.epsilon = epsilon
        self.pgd_iter = pgd_iter
        self.pgd_alpha = pgd_alpha
        self.pgd_attack = pgd_attack
        
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.test_adv_accuracy_history = []
        
        self.backbone = backbone# if backbone else ResNet18(num_classes=num_classes)  # ResNet18
        self.use_csa = use_csa
        self.conf_per_class = conf_per_class
        self.conf_path = conf_path

        
    def init_model(self):
        if self.use_csa:
            self.model = CSANet(backbone=self.backbone, num_classes=self.num_classes, conf_per_class=self.conf_per_class,conf_path=self.conf_path)
        else:
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
       
    def train_step(self, images, labels, adversarial_training=False):
        if self.pgd_attack:
            images = pgd_attack(self.model, images, labels, epsilon=self.epsilon, alpha=self.pgd_alpha, num_iter=self.pgd_iter)
            
            
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_function(labels, predictions)
        #gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = tape.gradient(loss, self.model.trainable_variables, 
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
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
            adversarial_training = True  #(epoch >= 30)
            self.train_step(images, labels, adversarial_training)

        for test_images, test_labels in self.test_ds:
            self.test_step(test_images, test_labels)

        print(f'Epoch {epoch + 1}, Loss: {self.train_loss.result()}, Accuracy: {self.train_accuracy.result() * 100}, Test Loss: {self.test_loss.result()}, Test Accuracy: {self.test_accuracy.result() * 100}')
        
        self.train_loss_history.append(self.train_loss.result().numpy())
        self.train_accuracy_history.append(self.train_accuracy.result().numpy() * 100)
        self.test_loss_history.append(self.test_loss.result().numpy())
        self.test_accuracy_history.append(self.test_accuracy.result().numpy() * 100)

        
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch + 1}.h5")
        self.model.save_weights(os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch + 1}.ckpt"))

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
