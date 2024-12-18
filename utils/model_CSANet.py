import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from PIL import Image
from tensorflow.keras import layers, Model

class NFCBank(tf.keras.Model):
    def __init__(self, conf_path=None, num_classes=10, conf_per_class=5000):
        #super(NFCBank, self).__init__()
        super().__init__()
        #if conf_path is not None:
        #    cfg.train.conf_path = conf_path
        self.conf_path = conf_path
        conf_set = [[] for _ in range(num_classes)]
        path = conf_path
        for i in range(num_classes):
            num = 0
            shap_path = os.path.join(path, f"{i}")
            assert Path(shap_path).exists()
            files = os.listdir(shap_path)
            for f in files:
                if "neg.pt" in f:
                    num += 1
            if num < conf_per_class:
                conf_per_class = num

        for i in range(num_classes):
            shap_path = os.path.join(path, f"{i}")
            assert Path(shap_path).exists()
            files = os.listdir(shap_path)
            num = 0
            for f in files:
                if "neg.pt" in f and num < conf_per_class:  # 限制读取每类最多 conf_per_class 个文件
                    image = tf.io.read_file(os.path.join(shap_path, f))
                    image = tf.image.decode_jpeg(image, channels=3)
                    image = tf.cast(image, tf.float32) / 255.0
                    conf_set[i].append(image)
                    num += 1

            conf_set[i] = tf.stack(conf_set[i], axis=0)
        conf_set = tf.concat(conf_set, axis=0)

############################################
        self.confounder_queue = tf.Variable(conf_set)
        #_, class_num, _, _ = self.confounder_queue.shape
        class_num= self.confounder_queue.shape
        print(self.confounder_queue.shape)

        self.K = class_num
        self.N = 10
        #self.nclass = num_classes
############################################
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
            conf_example = tf.concat(conf_example, axis=0)
            conf_set.append(conf_example)
        conf_set = tf.stack(conf_set, axis=0)

        return conf_set


class CSANet(tf.keras.Model):
    def __init__(self, backbone=None, num_classes=10, conf_per_class=5000, use_conf=False, mask_alpha=None, conf_path=None):
        #super(CSANetwork, self).__init__()
        super().__init__()
        self.backbone = backbone
        self.erb = NFCBank(conf_path=conf_path, num_classes=num_classes, conf_per_class=conf_per_class)
        self.test_CSA = use_conf
        self.mask_alpha = mask_alpha
        if self.test_CSA:
            print("Test with CSA")

    def call(self, x):
        if self.test_CSA and not self.training:
            output = self.backbone(x)
            y_pred = tf.argmax(output, axis=1)###########
            x, x_conf = self.split_x(x, y_pred)
            x_new = x + self.mask_alpha * x_conf
            preds = self.backbone(x_new)
        else:
            preds = self.backbone(x)
        return preds

    def split_x(self, x, y):
        x_v_set = self.erb.batch_sample_set(x, y)
        x_v_att = tf.reduce_mean(x_v_set, axis=1)

        return x, x_v_att
