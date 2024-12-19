import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from PIL import Image
from tensorflow.keras import layers, Model
######################################I refrence to https://github.com/Ytchen981/CSA/blob/main/model/NFCBank.py when I did this codes.

class NFCBank(tf.keras.Model):
    def __init__(self, conf_path=None, num_classes=10, conf_per_class=5000):
        super().__init__()
        self.conf_path = conf_path  Path to the confounder images
        conf_set = [[] for _ in range(num_classes)]# Loop through all classes to determine the number of confounders for each class
        path = conf_path
        for i in range(num_classes):
            num = 0
            shap_path = os.path.join(path, f"{i}")
            assert Path(shap_path).exists()
            files = os.listdir(shap_path)
            for f in files:
                if "neg" in f:#"neg" for NFC, negative shapley values
                    num += 1
            if num < conf_per_class:
                conf_per_class = num# Adjust conf_per_class based on the available number of confounders

        for i in range(num_classes):
            shap_path = os.path.join(path, f"{i}")# Path to the confounder images of class i
            assert Path(shap_path).exists()
            files = os.listdir(shap_path)
            num = 0
            for f in files:
                if "neg" in f and num < conf_per_class:
                    image = tf.io.read_file(os.path.join(shap_path, f))
                    image = tf.image.decode_jpeg(image, channels=3) # Read the image, decode it, and normalize the pixel values
                    image = tf.cast(image, tf.float32) / 255.0
                    conf_set[i].append(image)
                    num += 1
############this took me a long time trying to get the right format,especially axis=1,-1,0...
            conf_set[i] = tf.stack(conf_set[i], axis=0)# Stack the images for this class along the first axis
        conf_set = tf.concat(conf_set, axis=0)  # Concatenate all classes' confounders along the first axis

        self.confounder_queue = tf.Variable(conf_set, trainable=True)
        class_num= self.confounder_queue.shape
        print(self.confounder_queue.shape)

        self.K = class_num
        self.N = 10# Number of confounders to sample per class

    def batch_sample_set(self, x_s, label):
        bs_size = tf.shape(x_s)[0]
        conf_set = []
        index_list = [i for i in range(self.K)]
        for j in range(bs_size):
            conf_example = []
            for i in range(self.nclass):
                if i == label[j] and not cfg.train.other_class_conf:
                    selected = np.random.choice(index_list, self.N, replace=False)
                    conf_class_choosed = tf.gather(self.confounder_queue[i], selected)
                    conf_example.append(conf_class_choosed)
                #if i != label[j] and cfg.train.other_class_conf:
                #    selected = np.random.choice(index_list, self.N, replace=False)
                 #   conf_class_choosed = tf.gather(self.confounder_queue[i], selected)
                 #   conf_example.append(conf_class_choosed)
                #elif i == label[j] and not cfg.train.other_class_conf:
                #    selected = np.random.choice(index_list, self.N, replace=False)
                #    conf_class_choosed = tf.gather(self.confounder_queue[i], selected)
                 #   conf_example.append(conf_class_choosed)
            conf_example = tf.concat(conf_example, axis=0)# Concatenate all selected confounders for the current example
            conf_set.append(conf_example)
        conf_set = tf.stack(conf_set, axis=0)

        return conf_set


class CSANet(tf.keras.Model):
    def __init__(self, backbone=None, num_classes=10, conf_per_class=5000, conf_path=None):
        super().__init__()
        self.backbone = backbone
        self.erb = NFCBank(conf_path=conf_path, num_classes=num_classes, conf_per_class=conf_per_class)
       


    def call(self, x):
        preds = self.backbone(x)# Pass input through the backbone network to get predictions
        return preds
