import math
import numpy as np
from PIL import Image
from IPython.display import clear_output
import matplotlib.pyplot as plt

def images_square_grid(images, mode):

    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im


def get_noisy_target_images(images,img_shape,noise_sigma,num_img_channels):

    target_images = images[0].reshape((-1,img_shape,img_shape,num_img_channels))
    noise = np.clip(np.random.randn(*target_images.shape), -2., 2.)
    batch_images = target_images + noise_sigma * noise
    batch_images = np.clip(batch_images, 0., 1.)

    return batch_images, target_images


def show_autoencoder_output(sess, img_shape, input_images, inputs_tf, outputs_tf, data_image_mode):

    batch_images, target_images = get_noisy_target_images(input_images,img_shape,0.4,len(data_image_mode))

    clear_output()
    cmap = None if data_image_mode == 'RGB' else 'gray'

    samples = sess.run(
        outputs_tf,
        feed_dict={inputs_tf: batch_images})

    print('max_in = ', np.max(batch_images), '  max_out = ',np.max(samples))
    print('min_in = ', np.min(batch_images), '  min_out = ',np.min(samples))
    images_grid_out = images_square_grid(samples, data_image_mode)
    images_grid_in = images_square_grid(batch_images,data_image_mode)
    images_grid_all = np.concatenate((images_grid_in,images_grid_out),axis=1)
    plt.imshow(images_grid_all, cmap=cmap)
    plt.show()