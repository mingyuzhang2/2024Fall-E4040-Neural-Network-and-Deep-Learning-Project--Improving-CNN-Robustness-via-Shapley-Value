import os
import tensorflow as tf
import numpy as np
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import load_model
from pathlib import Path
import time
from utils.ResNet18_trainer import std, mean
from tensorflow.keras.preprocessing.image import save_img
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mlt
import numpy as np


def transform_fft(img):
    if len(img.shape) > 3:

        #img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        img = tf.reshape(img, [-1, img.shape[-3], img.shape[-2], img.shape[-1]])
        result = []
        for i in range(img.size[0]):
            tmp = np.array(img[i])
            tmp = np.fft.fft2(tmp)
            tmp = np.fft.fftshift(tmp)
            result.append(np.expand_dims(tmp, axis=0))

    else:
        img = np.array(img)
        img = np.fft.fft2(img)
        img = np.fft.fftshift(img)
        return np.array(img)


    result = tf.concat(result, axis=0)
    return result

def transform_ifft(img):
    if len(img.shape) > 3:

        img = tf.reshape(img, [-1, img.shape[-3], img.shape[-2], img.shape[-1]])

        result = []
        for i in range(img.shape[0]):
            tmp = np.array(img[i])
            tmp = np.fft.ifft2(np.fft.ifftshift(tmp))
            if np.abs(np.sum(np.imag(tmp))) > 1e-5:
                raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")
            result.append(np.real(tmp).astype(np.float32)[np.newaxis, ...])

    else:
        img = np.array(img)
        img = np.fft.ifft2(np.fft.ifftshift(img))
        if np.abs(np.sum(np.imag(img))) > 1e-5:
            raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")
        return np.real(img).astype(np.float32)

    result = tf.concat(result, axis=0)
    return result









def sample_mask(img_w, img_h, mask_w, mask_h, static_center=False):
    length = mask_w * mask_h + 1
    order = np.random.permutation(np.arange(0, mask_w * mask_h, 1))
    mask = np.ones((length, 3, mask_w, mask_h))
    mask = mask.reshape(length, 3, -1)

    for j in range(1, length):
        mask[j:, :, order[j - 1]] = 0
    mask = mask.reshape(length, 3, mask_w, mask_h)
    
    if static_center:
        mask[:, :, mask_w // 2, mask_h // 2] = 1
    ###mask = F.interpolate(mask.clone(), size=[img_w, img_h], mode="nearest").float()
    
    mask = tf.image.resize(mask, [img_w, img_h], method='nearest')
    mask = tf.cast(mask, tf.float32)

    return mask, order


def getShapley_pixel(img, label, model, sample_times, mask_size, k=0):
    b, w, h, c = img.shape
    shap_value = tf.zeros((mask_size ** 2,), dtype=tf.float32)
    #img_k = tf.expand_dims(img[k], axis=0)

    for i in range(sample_times):
        mask, order = sample_mask(w, h, mask_size, mask_size)
        #base = tf.repeat(img_k, repeats=mask.shape[0], axis=0)
        base = tf.tile(tf.expand_dims(img[k], axis=0), multiples=[mask.shape[0], 1, 1, 1])
        masked_img = base * mask
        output = model(masked_img)

        #if tf.reduce_any(tf.math.is_nan(output)):
            #raise ValueError("NAN in output")

        y = output[:, label[k]]
        yy = y[:-1]
        dy = yy - y[-1]
        shap_value[order] += dy
    shap_value /= sample_times
    return shap_value


def getShapley_freq(img, label, model, sample_times, mask_size, k=0, n_per_batch=1, split_n=1, static_center=False,
                    fix_masks=False, mask_path=None):
    b, c, w, h = img.shape
    length = mask_size ** 2 + 1
    shap_value = tf.zeros((mask_size ** 2,), dtype=tf.float32)

    for i in range(sample_times // n_per_batch):
        masks = []
        orders = []
        if not fix_masks:
            for j in range(n_per_batch):
                mask, order = sample_mask(w, h, mask_size, mask_size, static_center=static_center)
                masks.append(mask)
                orders.append(order)
            masks = tf.concat(masks, axis=0)
            assert maskes.shape[0] == n_per_batch * length
        else:
            maskes = np.load(os.path.join(mask_path, f"mask_{i}.npy"))
            orders = np.load(os.path.join(mask_path, f"order_{i}.npy"))

        if split_n > 1:
            #base = tf.signal.fft2d(tf.cast(img[k], tf.complex64))  # FFT of the input image
            base = transform_fft(img[k])
            bs = masks.shape[0] // split_n
            outputs = []

            for j in range(maskes.shape[0] // bs):
                if j == maskes.shape[0] // bs - 1:
                    current_mask = maskes[j * bs:]
                else:
                    current_mask = maskes[j * bs:(j + 1) * bs]
                    
                    
                #masked_img = tf.tile(base[None, :, :, :], [current_mask.shape[0], 1, 1, 1]) * current_mask
                masked_img = np.expand_dims(current_mask, axis=0)  # 扩展维度
                masked_img = np.repeat(masked_img, 3, axis=1)  # 重复扩展到3通道
                masked_img = np.tile(masked_img, (1, 1, w, h))
                
                #masked_img = base * tf.cast(current_mask, tf.complex64)
                #masked_img = tf.signal.ifft2d(masked_img)
                #masked_img = tf.math.real(masked_img)  ########## Extract the real part
                masked_img = transform_ifft(masked_img)
                masked_img = np.clip(masked_img, 0., 1.)
                outputs.append(model(masked_img))

            output = tf.concat(outputs, axis=0)
        else:
            #base = tf.signal.fft2d(tf.cast(img[k], tf.complex64))  # FFT of the input image
            #base = tf.signal.fft2d(tf.cast(img[k], tf.complex64))
            base = transform_fft(img[k])
            base = tf.expand_dims(base, 0)  # 扩展维度
            base = tf.repeat(base, 3, axis=0)  # 扩展到 3 通道
            base = tf.tile(base, [1, 1, w, h])  # 填充到 w, h 尺寸
            #masked_img = base * tf.cast(masks, tf.complex64)
            masked_img = base * maskes
            #masked_img = tf.signal.ifft2d(masked_img)
            #masked_img = tf.math.real(masked_img) ########
            masked_img = transform_ifft(masked_img)
            masked_img = np.clip(masked_img, 0.0, 1.0)

            output = model(masked_img)


        for j in range(n_per_batch):
            y = output[j * length:(j + 1) * length, label[k]]  # Extract predictions for target label
            yy = y[:-1]  # Exclude the last mask
            dy = yy - y[1:]  # Difference in predictions
            #if tf.reduce_any(tf.math.is_nan(dy)):
                #raise ValueError("NaN encountered in Shapley value computation")
            #for idx, val in enumerate(dy):
                #shap_value = tf.tensor_scatter_nd_add(shap_value, [[orders_list[j][idx]]], [val])
            shap_value[orders[j]] += dy
            
        if i % 100 == 0:
            print(f"{i}/{sample_times // n_per_batch}")
    shap_value /= (sample_times // n_per_batch * n_per_batch)
    return shap_value


def getShapley_freq_softmax(img, label, model, sample_times, mask_size, k=0, n_per_batch=1, split_n=1,
                            static_center=False, fix_masks=False, mask_path=None):
    b, w, h, c = img.shape
    length = mask_size ** 2 + 1
    shap_value = tf.zeros((mask_size ** 2,), dtype=tf.float32)

    ########img_k = tf.expand_dims(img[k], axis=0)

    for i in range(sample_times // n_per_batch):
        mask, order = sample_mask(w, h, mask_size, mask_size, static_center=static_center)
        #base = tf.signal.fft2d(tf.cast(img[k], tf.complex64))
        
        #print("base 的维数:", base.ndim)
        #print("base 的形状:", base.shape)
        #print("mask 的维数:", mask.ndim)
        #print("mask 的形状:", mask.shape)
        #print("mask.shape[0], c, w, h:",mask.shape[0], c, w, h)
        #base = base.permute(2, 0, 1)
        #print("mask 的形状:", mask.shape)
        #print("base 的形状:", base.shape)
        #base = tf.transpose(base, perm=[1,2,0]) 
        #print("mask 的形状:", mask.shape)
        #print("base 的形状:", base.shape)
        #base = tf.tile(base, [mask.shape[0], c, w, h])
        #base = tf.broadcast_to(base, [mask.shape[0], c, w, h])
        #print("base 的维数2:", base.ndim)
        #print("base 的形状2:", base.shape)
        #base = tf.tile(base, [mask.shape[0], 1, 1, 1])
        # masked_base = base * tf.cast(mask, tf.complex64)
        #base_expanded = np.broadcast_to(base, (mask.shape[0], c, w, h))
        #print(base.dtype)
        #print(mask.dtype)
        
        base = transform_fft(img[k])
        base = tf.math.real(base)
        base = tf.cast(base, tf.float32)
        base = tf.expand_dims(base, axis=0)
        #print("base 的形状1:", base.shape)
        base = tf.broadcast_to(base, [mask.shape[0], w, h, c])
        #base = tf.broadcast_to(base, [mask.shape[0], w, h, mask_size])
        #base = tf.tile(base, [mask.shape[0], w, h, mask_size//c])
        #print("base 的形状:", base.shape)
        #print("mask 的形状:", mask.shape)

        masked_base = base * mask
        #masked_img = tf.signal.ifft2d(masked_base)
        #######masked_img = tf.clip_by_value(tf.math.real(masked_img), 0.0, 1.0)
        masked_img = transform_ifft(masked_base)
        masked_img = np.clip(masked_img, 0., 1.)
        # masked_img = tf.clip_by_value(masked_img, 0.0, 1.0)

        output = tf.nn.softmax(model(masked_img), axis=1)
        print("masked_img:", masked_img.shape)
        print("output:", output)
        # y = tf.gather(output, label[k], axis=1)
        #y = output[:, label[k]]
        y = tf.argmax(output, axis=-1)
        print("y:", y)


        yy = y[:-1]
        dy = yy - y[-1]

        # if tf.reduce_any(tf.math.is_nan(dy)):
        #    raise ValueError("Nan in dy")

        # shap_value = tf.tensor_scatter_nd_add(shap_value,tf.reshape(order, (-1, 1)),dy)
        shap_value[order] += dy

        if i % 100 == 0:
            print(f"{i}/{sample_times}")

    shap_value /= sample_times
    return shap_value


def gen_dis_list(mask_w, mask_h):
    dis_dict = dict()
    for i in range(mask_w * mask_h):
        row = i // mask_w
        col = i % mask_w
        center_row = mask_w / 2 - 0.5
        center_col = mask_h / 2 - 0.5
        dis = (row - center_row) ** 2 + (col - center_col) ** 2
        dis_key = f"{dis:.2f}"

        if dis_key not in dis_dict:
            dis_dict[dis_key] = []
        dis_dict[dis_key].append(i)

    dis = np.sort(np.array(list(dis_dict.keys()), dtype=float))
    return dis_dict, dis


def sample_mask_dict(img_w, img_h, mask_w, mask_h, dis_dict, keys):
    length = len(keys) + 1 
    order = np.random.permutation(np.arange(0, len(keys), 1))
    mask = tf.ones((length, 3, mask_w, mask_h), dtype=tf.float32)
    mask = tf.reshape(mask, (length, 3, -1))
    
    for j in range(1, length):
        points = dis_dict[f"{keys[order[j-1]]:.2f}"]
        mask[j:, :, points] = 0

    mask = tf.reshape(mask, (length, 3, mask_w, mask_h))
    mask = tf.image.resize(mask, [img_w, img_h], method='nearest')
    return mask, order


def getShapley_freq_dis(img, label, model, sample_times, mask_size, k=0):
    b, c, w, h = img.shape
    shap_value = tf.zeros((mask_size ** 2,), dtype=tf.float32)
    dis_dict, dis = gen_dis_list(mask_size, mask_size)

    for i in range(sample_times):
        mask, order = sample_mask_dict(w, h, mask_size, mask_size, dis_dict, dis)
        #base = tf.signal.fft2d(tf.cast(img[k], tf.complex64))
        base = transform_fft(img[k])
        base = tf.expand_dims(base, 0)  # 扩展维度
        base = tf.repeat(base, 3, axis=0)  # 扩展到 3 通道#########axis=1
        base = tf.tile(base, [1, 1, w, h])  # 填充到 w, h 尺寸
        masked_img = base * maskes
        #masked_img = tf.signal.ifft2d(masked_img)
        #masked_img = tf.math.real(masked_img)  ########
        masked_img = transform_ifft(masked_img)
        masked_img = np.clip(masked_img, 0.0, 1.0)
        output = model(masked_img)  # Shape: (num_masks, num_classes)

        if np.any(np.isnan(output)):
            masked_img = np.clip(masked_img, 0., 1.)
            output = model(masked_img)
            assert not np.any(np.isnan(output))
            
        y = output[:, label[k]]
        yy = y[:-1]
        dy = yy - y[1:]
#########raise ValueError("Nan in dy")
        for i in range(len(dy)):
            key = f"{dis[order[i]]:.2f}"
            shap_value[dis_dict[key]] += dy[i]

    shap_value /= sample_times
    return shap_value


def visual_shap(shap_value, w, h, img_path):
 
    max_value = tf.reduce_max(shap_value).numpy()
    min_value = tf.reduce_min(shap_value).numpy()
    maximum = max(abs(max_value), abs(min_value))

    shap_value = tf.reshape(shap_value, (w, h)).numpy()  # Reshape the tensor to match the image dimensions
    plt.figure()
    norm = Normalize(vmin=-maximum, vmax=maximum)
    plt.imshow(shap_value, norm=norm, cmap=mlt.cm.bwr)  # Blue-White-Red colormap
    plt.gca().get_yaxis().set_visible(False)  # Hide y-axis
    plt.gca().get_xaxis().set_visible(False)  # Hide x-axis
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(img_path, format='pdf')  # Save image as a PDF
    plt.close()


def visual_shap_w_tick(shap_value, w, h, img_path):
    
    max_value = tf.reduce_max(shap_value).numpy()
    min_value = tf.reduce_min(shap_value).numpy()
    maximum = max(abs(max_value), abs(min_value))

    x_spec = np.arange(w)
    x_shift_spec = np.fft.fftshift(x_spec)  # Shift the x-axis for better visualization
    y_spec = np.arange(h)
    y_shift_spec = np.fft.fftshift(y_spec)  # Shift the y-axis for better visualization

    shap_value = tf.reshape(shap_value, (w, h)).numpy()  # Reshape tensor to the appropriate shape for plotting
    plt.figure()
    norm = Normalize(vmin=-maximum, vmax=maximum)
    plt.imshow(shap_value, norm=norm, cmap=mlt.cm.bwr)  # Blue-White-Red colormap
    plt.xticks(x_spec, x_shift_spec)  # Set x-ticks
    plt.yticks(y_spec, y_shift_spec)  # Set y-ticks
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(img_path, format='svg')  # Save image as SVG
    plt.close()
















