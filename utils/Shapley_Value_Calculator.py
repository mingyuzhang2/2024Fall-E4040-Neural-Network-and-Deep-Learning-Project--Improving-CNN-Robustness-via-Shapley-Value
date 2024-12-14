import tensorflow as tf
import numpy as np
import os

# Fast Fourier Transform (FFT) transformation
def transform_fast_fourier_transform(img):
    img = tf.reshape(img, [-1, *img.shape[-3:]]) if len(img.shape) > 3 else np.array(img)
    if len(img.shape) > 3:
        fft_result = [
            tf.convert_to_tensor(np.fft.fftshift(np.fft.fft2(np.array(img[i]))))[None, ...]
            for i in range(img.shape[0]]
    else:
        fft_result = tf.convert_to_tensor(np.fft.fftshift(np.fft.fft2(img))
    return tf.concat(fft_result, axis=0) if len(img.shape) > 3 else fft_result

# Inverse Fast Fourier Transform (IFFT) transformation
def transform_inverse_fast_fourier_transform(img):
    img = tf.reshape(img, [-1, *img.shape[-3:]]) if len(img.shape) > 3 else np.array(img)
    ifft_result = []
    for i in range(img.shape[0] if len(img.shape) > 3 else 1):
        tmp = np.fft.ifft2(np.fft.ifftshift(np.array(img[i] if len(img.shape) > 3 else img)))
        if np.abs(np.imag(tmp).sum()) > 1e-5:
            raise ValueError(f"Imaginary part too large: {np.abs(np.imag(tmp).sum())}")
        ifft_result.append(tf.convert_to_tensor(np.real(tmp), dtype=tf.float32)[None, ...])
    return tf.concat(ifft_result, axis=0) if len(img.shape) > 3 else ifft_result[0]

# Monte Carlo Sampling for random masks
def monte_carlo_sample_mask(img_w, img_h, mask_w, mask_h, static_center=False):
    length, order = mask_w * mask_h + 1, np.random.permutation(np.arange(mask_w * mask_h))
    mask = tf.ones((length, 3, mask_w, mask_h))
    for j in range(1, length):
        mask = tf.tensor_scatter_nd_update(
            mask,
            [[i, 0, order[j - 1]] for i in range(j, length)],
            tf.zeros((length - j,))
        )
    mask = tf.image.resize(mask, [img_w, img_h], method="nearest")
    if static_center:
        mask[:, :, mask_w // 2, mask_h // 2] = 1
    return mask, order

# Calculating Shapley values by pixel
def ShapleyValue_pixel(img, label, model, sample_times, mask_size, k=0):
    shap_value = tf.zeros((mask_size ** 2))
    for _ in range(sample_times):
        mask, order = monte_carlo_sample_mask(*img.shape[2:], mask_size, mask_size)
        masked_img = tf.tile(img[k:k+1], [mask.shape[0], 1, 1, 1]) * mask
        output = model(masked_img)
        if tf.reduce_any(tf.math.is_nan(output)):
            raise ValueError("NAN in output")
        shap_value = tf.tensor_scatter_nd_add(
            shap_value, tf.expand_dims(order, axis=1), (output[:-1, label[k]] - output[1:, label[k]]).numpy()
        )
    return shap_value / sample_times

# Calculating Shapley values in the frequency domain
def ShapleyValue_frequency(img, label, model, sample_times, mask_size, k=0, n_per_batch=1, split_n=1, static_center=False, fix_masks=False, mask_path=None):
    shap_value = tf.zeros((mask_size ** 2))
    length = mask_size ** 2 + 1
    for i in range(sample_times // n_per_batch):
        masks, orders = [], []
        for _ in range(n_per_batch):
            mask, order = monte_carlo_sample_mask(*img.shape[2:], mask_size, mask_size, static_center)
            masks.append(mask)
            orders.append(order)
        masks = tf.concat(masks, axis=0)
        base = tf.tile(transform_fast_fourier_transform(img[k:k+1]), [masks.shape[0], 1, 1, 1])
        if split_n > 1:
            outputs = [
                model(tf.clip_by_value(transform_inverse_fast_fourier_transform(base[j * length:(j + 1) * length] * masks[j * length:(j + 1) * length]), 0., 1.))
                for j in range(split_n)
            ]
        else:
            outputs = [model(tf.clip_by_value(transform_inverse_fast_fourier_transform(base * masks), 0., 1.))]
        output = tf.concat(outputs, axis=0)
        for j in range(n_per_batch):
            shap_value = tf.tensor_scatter_nd_add(
                shap_value, tf.expand_dims(orders[j], axis=1),
                (output[j * length:(j + 1) * length, label[k]][:-1] - output[j * length:(j + 1) * length, label[k]][1:]).numpy()
            )
        if i % 100 == 0:
            print(f"{i}/{sample_times // n_per_batch}")
    return shap_value / sample_times



