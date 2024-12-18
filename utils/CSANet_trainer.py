import tensorflow as tf
import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from utils.model_CSANet import CSANet
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import random
import numpy as np


def madrys_loss(model,x_natural,y,optimizer,step_size=0.007,epsilon=0.031,perturb_steps=10,beta=6.0,distance='l_inf'):
    batch_size = tf.shape(x_natural)[0]
    
    x_adv = x_natural + 0.001 * tf.random.normal(tf.shape(x_natural))
    
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                logits = model(x_adv, training=False)
                loss_ce = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
            grad = tape.gradient(loss_ce, x_adv)
            x_adv = x_adv + step_size * tf.sign(grad)
            x_adv = tf.clip_by_value(x_adv, x_natural - epsilon, x_natural + epsilon)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    
    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        logits = model(x_adv, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
    adv_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), y), tf.float32))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, adv_acc


def pgd(model,x_natural,y,step_size=0.007,epsilon=0.031,perturb_steps=10,distance='l_inf',is_normalize=False):
    if is_normalize:
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        mu = tf.convert_to_tensor(cifar10_mean, dtype=tf.float32)[None, None, :]
        std = tf.convert_to_tensor(cifar10_std, dtype=tf.float32)[None, None, :]
        upper_limit = (1 - mu) / std
        lower_limit = (0 - mu) / std
    
    batch_size = tf.shape(x_natural)[0]
    
    x_adv = x_natural + 0.001 * tf.random.normal(tf.shape(x_natural))
    
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                logits = model(x_adv, training=False)
                loss_ce = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
            grad = tape.gradient(loss_ce, x_adv)
            x_adv = x_adv + step_size * tf.sign(grad)
            x_adv = tf.clip_by_value(x_adv, x_natural - epsilon, x_natural + epsilon)
            if is_normalize:
                x_adv = tf.clip_by_value(x_adv, lower_limit, upper_limit)
            else:
                x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    
    else:
        if is_normalize:
            x_adv = tf.clip_by_value(x_adv, lower_limit, upper_limit)
        else:
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    
    return x_adv




#def set_seed(seed_value=111):
#    random.seed(seed_value)
#    np.random.seed(seed_value)
#    tf.random.set_seed(seed_value)
#set_seed(111)

def conf_alpha_schedule(epoch):
    if epoch > 80:
        return 1
    else:
        return epoch / 80


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
    elif epoch > 80 and epoch <= 120:
        return 0.01
    else:
        return 0.001

lr_scheduler = LearningRateScheduler(learning_rate_schedule)

    

class CSANet_trainer():
    def __init__(self, train_ds, test_ds, num_classes, epochs, batch_size, lr, momentum, decay, checkpoint_dir, backbone=None, conf_per_class=5000, conf_path=None):##########
        
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.checkpoint_dir = checkpoint_dir
        
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.test_adv_accuracy_history = []
        self.backbone = backbone# if backbone else ResNet18(num_classes=num_classes)  # ResNet18 是一个例子
        self.conf_per_class = conf_per_class
        self.conf_path = conf_path

        
    def init_model(self):
        #self.model = ResNet18(num_classes=self.num_classes)
        self.model = CSANet(backbone=self.backbone, num_classes=self.num_classes, conf_per_class=self.conf_per_class,
                            conf_path=self.conf_path)
        self.model.build((None, 32, 32, 3))

    def init_loss(self):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        ########
        self.adv_loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_adv_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_adv_accuracy')
        self.test_adv_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_adv_accuracy')



    def init_optimizer(self):
        
        self.optimizer = tf.keras.optimizers.SGD(
        learning_rate=self.lr,
        momentum=self.momentum,
        decay=self.decay
    )
        ####################
    #def generate_adversarial_images(self, images, labels, epoch):
    #    adv_images, adv_acc = pdg(self.model, images, labels, self.optimizer)
    #    return adv_images, adv_acc

    def train_step(self, images, labels):
        #with tf.GradientTape() as tape:
        #    predictions = self.model(images, training=True)
        #    loss = self.loss_function(labels, predictions)
            #####
        #    if adv_images is not None and adv_labels is not None:
         #       adv_predictions = self.model(adv_images, training=True)
        #        adv_loss = self.adv_loss_function(adv_labels, adv_predictions)
        #    else:
         #       adv_loss = 0

        #    total_loss = loss + adv_loss ##############   
        
        #gradients = tape.gradient(loss, self.model.trainable_variables)
        #self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #self.train_loss(loss)
        #self.train_accuracy(labels, predictions)
        
        #if adv_images is not None and adv_labels is not None:
        #    self.train_adv_accuracy(adv_labels, adv_predictions)
        #x_natural = images
        loss, adv_acc = madrys_loss(
            model=self.model,
            x_natural=images,
            y=labels,
            optimizer=self.optimizer,
            step_size=0.007,
            epsilon=0.031,
            perturb_steps=10,
            beta=6.0
        )
        
        self.train_loss(loss)
        self.train_accuracy(y, self.model(x_natural, training=False))  # 使用原始样本评估准确率
        self.train_adv_accuracy(y, adv_acc)# 使用对抗样本评估准确率
        

    def test_step(self, images, labels):
        #predictions = self.model(images, training=False)
        #t_loss = self.loss_function(labels, predictions)

        #if adv_images is not None and adv_labels is not None:
        #    adv_predictions = self.model(adv_images, training=False)
        #    adv_loss = self.adv_loss_function(adv_labels, adv_predictions)
        #else:
         #   adv_loss = 0
                
        #self.test_loss(t_loss + adv_loss)
        #self.test_accuracy(labels, predictions)
        
        #if adv_images is not None and adv_labels is not None:
        #    self.test_adv_accuracy(adv_labels, adv_predictions)
        x_adv = pgd(
            model=self.model,
            #x_natural=x_natural,
            #y=y,
            x_natural=images,
            y=labels,
            step_size=0.007,
            epsilon=0.031,
            perturb_steps=10,
            distance='l_inf'
        )
        
        # 自然样本的评估
        logits = self.model(x_natural, training=False)
        natural_loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        self.test_accuracy(y, logits)

        # 对抗样本的评估
        logits_adv = self.model(x_adv, training=False)
        adv_loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits_adv)
        self.test_adv_accuracy(y, logits_adv)
        
        # 总测试损失
        self.test_loss(natural_loss + adv_loss)

        

    def train_epoch(self, epoch):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        self.train_adv_accuracy.reset_states()
        self.test_adv_accuracy.reset_states() 
        
        new_lr = learning_rate_schedule(epoch)
        tf.keras.backend.set_value(self.optimizer.lr, new_lr)

        #for images, labels in self.train_ds:
        #    adv_images, adv_labels = self.generate_adversarial_images(images, labels, epoch)
        #    #self.train_step(images, labels)
         #   self.train_step(images, labels, adv_images, adv_labels)


        #for test_images, test_labels in self.test_ds:
        #    adv_test_images, adv_test_labels = self.generate_adversarial_images(test_images, test_labels, epoch)
        #    self.test_step(test_images, test_labels, adv_test_images, adv_test_labels)
            #self.test_step(test_images, test_labels)

        #template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        # 训练步骤
        for x_natural, y in self.train_ds:
            self.train_step(x_natural, y)

        # 测试步骤
        for x_natural, y in self.test_ds:
            self.test_step(x_natural, y)# 训练步骤
        for x_natural, y in self.train_ds:
            self.train_step(x_natural, y)

        # 测试步骤
        for x_natural, y in self.test_ds:
            self.test_step(x_natural, y)
            
        template = ('Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, '
                    'Test Loss: {:.4f}, Test Accuracy: {:.2f}%, '
                    'Test Adv Accuracy: {:.2f}%')

        print(template.format(epoch + 1,
                           self.train_loss.result(),
                           self.train_accuracy.result() * 100,
                           self.test_loss.result(),
                           self.test_accuracy.result() * 100,
                           self.test_adv_accuracy.result() * 100))
        
        self.train_loss_history.append(self.train_loss.result().numpy())
        self.train_accuracy_history.append(self.train_accuracy.result().numpy() * 100)
        self.test_loss_history.append(self.test_loss.result().numpy())
        self.test_accuracy_history.append(self.test_accuracy.result().numpy() * 100)
        self.test_adv_accuracy_history.append(self.test_adv_accuracy.result().numpy() * 100)

        
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

            #self.saved_train_loss = self.train_loss.result().numpy()
            #self.saved_train_accuracy = self.train_accuracy.result().numpy()
            #self.saved_test_loss = self.test_loss.result().numpy()
            #self.saved_test_accuracy = self.test_accuracy.result().numpy()
            
            print(f"Restored from epoch {checkpoint_epoch}: ")
            #print(f"Previous Train Loss: {self.saved_train_loss}, Train Accuracy: {self.saved_train_accuracy * 100}%")
            #print(f"Previous Test Loss: {self.saved_test_loss}, Test Accuracy: {self.saved_test_accuracy * 100}%")
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
