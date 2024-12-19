import os
import tensorflow as tf
import numpy as np
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import load_model#############I referenced to https://github.com/Ytchen981/CSA/tree/main
from pathlib import Path
import time
from utils.ResNet18_trainer import std, mean
from tensorflow.keras.preprocessing.image import save_img
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mlt
import numpy as np


def transform_fft(img):
    if len(img.shape) > 3:  # Check if the input image has more than 3 dimensions (e.g., a batch of images)
        img = tf.reshape(img, [-1, img.shape[-3], img.shape[-2], img.shape[-1]])# Reshape the image to handle each individual image in the batch
        result = []
        for i in range(img.size[0]):
            tmp = np.array(img[i])
            tmp = np.fft.fft2(tmp) #Perform 2D FFT on the image
            tmp = np.fft.fftshift(tmp)# Shift the zero-frequency component to the center of the spectrum
            result.append(np.expand_dims(tmp, axis=0))# Append the transformed image (expand dims to retain shape consistency)
    else:# If the image is not in a batch (i.e., it's a single image)
        img = np.array(img)
        img = np.fft.fft2(img)# Perform 2D FFT on the image
        img = np.fft.fftshift(img) # Shift the zero-frequency component to the center of the spectrum
        return np.array(img)
    result = tf.concat(result, axis=0)# Concatenate all transformed images in the batch along the first axis (batch axis)
    return result

def transform_ifft(img):
    if len(img.shape) > 3:

        img = tf.reshape(img, [-1, img.shape[-3], img.shape[-2], img.shape[-1]])

        result = []
        for i in range(img.shape[0]):
            tmp = np.array(img[i])
            tmp = np.fft.ifft2(np.fft.ifftshift(tmp))
            if np.abs(np.sum(np.imag(tmp))) > 1e-5: # Check if the imaginary part of the transformed image is too large
                raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")
            result.append(np.real(tmp).astype(np.float32)[np.newaxis, ...]) # Append the real part of the inverse transformed image (convert to float32)

    else:
        img = np.array(img) # If the image is not in a batch (i.e., it's a single image)
        img = np.fft.ifft2(np.fft.ifftshift(img)) # Perform the inverse 2D FFT on the image (inverse of fftshift and fft2)
        if np.abs(np.sum(np.imag(img))) > 1e-5:
            raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")
        return np.real(img).astype(np.float32) #Return the real part of the inverse transformed image (convert to float32)

    result = tf.concat(result, axis=0)  # Return the batch of inverse transformed images
    return result









def sample_mask(img_w, img_h, mask_w, mask_h, static_center=False):
    length = mask_w * mask_h + 1# Calculate the total number of mask elements (height * width + 1 for the center element)
    order = np.random.permutation(np.arange(0, mask_w * mask_h, 1)) # Create a random order of indices from 0 to mask_w * mask_h - 1
    mask = np.ones((length, 3, mask_w, mask_h))# Initialize a mask with ones, shape: (length, 3, mask_w, mask_h)
    # The mask has an additional element at the beginning for the center position
    mask = mask.reshape(length, 3, -1)# Reshape the mask to (length, 3, -1) for easier manipulation

    for j in range(1, length):# Update the mask: for each position, make elements after the current one in the order 0
        mask[j:, :, order[j - 1]] = 0# Set the selected positions to 0
    mask = mask.reshape(length, 3, mask_w, mask_h) # Reshape the mask back to its original shape
    
    if static_center: # If static_center is True, set the center element of the mask to 1 (preserving the center in the mask)
        mask[:, :, mask_w // 2, mask_h // 2] = 1
    
    mask = tf.image.resize(mask, [img_w, img_h], method='nearest')# Resize the mask to match the target image size (img_w, img_h)
    mask = tf.cast(mask, tf.float32)# Convert the mask to a float32 tensor

    return mask, order


def getShapley_freq_softmax(img, label, model, sample_times, mask_size, k=0, n_per_batch=1, split_n=1,
                            static_center=False, fix_masks=False, mask_path=None):
    b, w, h, c = img.shape # Extract the shape of the input image (batch size, width, height, channels)##################b,w,h,c might be confusing. I used to many transformation so the dimensions are messy
    length = mask_size ** 2 + 1# Calculate the total number of mask elements (mask_size * mask_size + 1 for the center)
    shap_value = tf.zeros((mask_size ** 2,), dtype=tf.float32) # Initialize the Shapley value tensor to zeros, shape: (mask_size * mask_size,)

    ########img_k = tf.expand_dims(img[k], axis=0)
 # Loop for sampling multiple times (adjusted by n_per_batch)
    for i in range(sample_times // n_per_batch):# Sample a mask with shuffled elements and an optional static center
        mask, order = sample_mask(w, h, mask_size, mask_size, static_center=static_center)
        #base = tf.signal.fft2d(tf.cast(img[k], tf.complex64))
        #############some codes to check the shape
        #print("base:", base.ndim)
        #print("base :", base.shape)
        #print("mask:", mask.ndim)
        #print("mask :", mask.shape)
        #print("mask.shape[0], c, w, h:",mask.shape[0], c, w, h)
        #base = base.permute(2, 0, 1)
        #print("mask :", mask.shape)
        #print("base :", base.shape)
        #base = tf.transpose(base, perm=[1,2,0]) 
        #print("mask :", mask.shape)
        #print("base :", base.shape)
        #base = tf.tile(base, [mask.shape[0], c, w, h])
        #base = tf.broadcast_to(base, [mask.shape[0], c, w, h])
        #print("base :", base.ndim)
        #print("base :", base.shape)
        #base = tf.tile(base, [mask.shape[0], 1, 1, 1])
        # masked_base = base * tf.cast(mask, tf.complex64)
        #base_expanded = np.broadcast_to(base, (mask.shape[0], c, w, h))
        #print(base.dtype)
        #print(mask.dtype)
        
        base = transform_fft(img[k])# Transform the input image for FFT-based operations
        base = tf.math.real(base)
        base = tf.cast(base, tf.float32)# Extract the real part of the FFT result and cast it back to float32
        base = tf.expand_dims(base, axis=0)# Expand the dimensions of the base tensor (batch size dimension)
        #print("base:", base.shape)
        base = tf.broadcast_to(base, [mask.shape[0], w, h, c])# Broadcast the base tensor to match the mask's dimensions (mask.shape[0] x width x height x channels)
        #base = tf.broadcast_to(base, [mask.shape[0], w, h, mask_size])
        #base = tf.tile(base, [mask.shape[0], w, h, mask_size//c])
        #print("base :", base.shape)
        #print("mask:", mask.shape)

        masked_base = base * mask # Apply the mask to the base tensor (element-wise multiplication)
        #masked_img = tf.signal.ifft2d(masked_base)
        #######masked_img = tf.clip_by_value(tf.math.real(masked_img), 0.0, 1.0)
        masked_img = transform_ifft(masked_base)  # Transform the masked tensor back using IFFT
        masked_img = np.clip(masked_img, 0., 1.)lip the resulting image to the [0, 1] range
        # masked_img = tf.clip_by_value(masked_img, 0.0, 1.0)

        output = tf.nn.softmax(model(masked_img), axis=1)
        print("masked_img:", masked_img.shape)
        print("output:", output)
        # y = tf.gather(output, label[k], axis=1)
        #y = output[:, label[k]]
        y = tf.argmax(output, axis=-1)
        #print("y:", y)##############the reason I got (0,)in my notebook 2....


        yy = y[:-1]# Calculate the Shapley value contribution (difference between current and baseline prediction)
        dy = yy - y[-1]

        # if tf.reduce_any(tf.math.is_nan(dy)):
        #    raise ValueError("Nan in dy")

        # shap_value = tf.tensor_scatter_nd_add(shap_value,tf.reshape(order, (-1, 1)),dy)
        shap_value[order] += dy

        if i % 100 == 0:
            print(f"{i}/{sample_times}")

    shap_value /= sample_times
    return shap_value


















