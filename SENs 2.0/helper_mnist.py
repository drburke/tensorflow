import math
import numpy as np
from PIL import Image
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from urllib.request import urlretrieve
import hashlib
import shutil

def _read32(bytestream):
    """
    Read 32-bit integer from bytesteam
    :param bytestream: A bytestream
    :return: 32-bit integer
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)


def _ungzip(save_path, extract_path, database_name, _):
    """
    Unzip a gzip file and extract it to extract_path
    :param save_path: The path of the gzip files
    :param extract_path: The location to extract the data to
    :param database_name: Name of database
    :param _: HACK - Used to have to same interface as _unzip
    """
    # Get data from save_path
    with open(save_path, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number {} in file: {}'.format(magic, f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)

    # Save data to extract_path
    for image_i, image in enumerate(
            tqdm(data, unit='File', unit_scale=True, miniters=1, desc='Extracting {}'.format(database_name))):
        Image.fromarray(image, 'L').save(os.path.join(extract_path, 'image_{}.jpg'.format(image_i)))

def get_image(image_path, width, height, mode):
    """
    Read image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)

    if image.size != (width, height):  # HACK - Check if image is from the CELEBA dataset
        # Remove most pixels that aren't part of a face
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(mode))

def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch/255.


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


def get_noisy_target_images(images,img_shape,noise_sigma,num_img_channels,augment=False):

    #target_images = images.reshape((-1,img_shape,img_shape,num_img_channels))
    target_images = images.reshape((-1,img_shape,img_shape,num_img_channels))
    if augment:
        if np.random.randint(2) == 1:
            target_images = np.flip(target_images,1)
        if np.random.randint(2) == 1:
            target_images = np.flip(target_images,2)
        target_images = np.rot90(target_images,k=np.random.randint(4), axes=(1,2))

    noise = np.clip(np.random.randn(*target_images.shape), -2., 2.)
    batch_images = target_images + noise_sigma * noise
    batch_images = np.clip(batch_images, 0., 1.)



    return batch_images, target_images


def show_autoencoder_output(sess, img_shape, input_images, inputs_tf, outputs_tf, generator_output_tf, generator_input_tf, embedded_image_dim, data_image_mode,embedded_features_input_tf,label='default'):

    if len(data_image_mode) == 1:

        batch_images, target_images = get_noisy_target_images(input_images,img_shape,0.4,len(data_image_mode), augment=False)

        #clear_output()
        cmap = None if data_image_mode == 'RGB' else 'gray'

        noise_inputs = np.clip(np.random.normal(size=(25,embedded_image_dim)),0.,1.)

    
        autoencode_samples, generated_samples = sess.run([outputs_tf,generator_output_tf],
            feed_dict={inputs_tf: batch_images, generator_input_tf: noise_inputs,
            embedded_features_input_tf:np.reshape(np.arange(25),(25,1))})#np.random.randint(embedded_image_dim,size=(25,1))})

        print('max_in = ', np.max(batch_images), '  max_out = ',np.max(autoencode_samples))
        print('min_in = ', np.min(batch_images), '  min_out = ',np.min(autoencode_samples))

        images_grid_out = images_square_grid(autoencode_samples, data_image_mode)
        images_grid_in = images_square_grid(batch_images,data_image_mode)
        images_grid_gan = images_square_grid(generated_samples, data_image_mode)

        print(np.shape(images_grid_out),np.shape(images_grid_in),np.shape(images_grid_gan))
        images_grid_all = np.concatenate((images_grid_in,images_grid_out,images_grid_gan),axis=1)

        fig = plt.figure()
        plt.imshow(images_grid_all, cmap=cmap)
        plt.show()
        plt.imsave(label+'.png', images_grid_all, cmap=cmap)
        print(label)

    else:
        clear_output()
        print(np.shape(input_images))
        batch_images, target_images = get_noisy_target_images(input_images,img_shape,0.0,len(data_image_mode), augment=False)
        #batch_images = input_images
        print(np.shape(input_images))
        print(np.shape(batch_images))
        
        cmap = None if data_image_mode == 'RGB' else 'gray'
        # batch_images = input_images
        
        autoencode_samples = sess.run([outputs_tf],
            feed_dict={inputs_tf: batch_images})

        print('max_in = ', np.max(batch_images), '  max_out = ',np.max(autoencode_samples))
        print('min_in = ', np.min(batch_images), '  min_out = ',np.min(autoencode_samples))
        print(np.shape(autoencode_samples))

        autoencode_samples = np.squeeze(autoencode_samples)
        print(np.shape(autoencode_samples))

        images_grid_in = images_square_grid(batch_images,data_image_mode)
        images_grid_out = images_square_grid(autoencode_samples, data_image_mode)
        # images_grid_in = images_square_grid(batch_images,data_image_mode)

        print(np.shape(images_grid_in))
        print(np.shape(images_grid_out))
        images_grid_all = np.concatenate((images_grid_in,images_grid_out),axis=1)
        print(np.shape(images_grid_all))
        fig = plt.figure()
        plt.imshow(images_grid_all, cmap=cmap)
        plt.show()
        plt.imsave(label+'.png', images_grid_all, dpi=150)
        print(label)

def download_extract(database_name, data_path):
    """
    Download and extract database
    :param database_name: Database name
    """
    DATASET_CELEBA_NAME = 'celeba'
    DATASET_MNIST_NAME = 'mnist'

    if database_name == DATASET_CELEBA_NAME:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        extract_path = os.path.join(data_path, 'img_align_celeba')
        save_path = os.path.join(data_path, 'celeba.zip')
        extract_fn = _unzip
    elif database_name == DATASET_MNIST_NAME:
        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        hash_code = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
        extract_path = os.path.join(data_path, 'mnist')
        save_path = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
        extract_fn = _ungzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)

def download_extract(database_name, data_path):
    """
    Download and extract database
    :param database_name: Database name
    """
    DATASET_CELEBA_NAME = 'celeba'
    DATASET_MNIST_NAME = 'mnist'

    if database_name == DATASET_CELEBA_NAME:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        extract_path = os.path.join(data_path, 'img_align_celeba')
        save_path = os.path.join(data_path, 'celeba.zip')
        extract_fn = _unzip
    elif database_name == DATASET_MNIST_NAME:
        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        hash_code = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
        extract_path = os.path.join(data_path, 'mnist')
        save_path = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
        extract_fn = _ungzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)

class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


class Dataset(object):
    """
    Dataset
    """
    def __init__(self, dataset_name, data_files):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        DATASET_CELEBA_NAME = 'celeba'
        DATASET_MNIST_NAME = 'mnist'
        IMAGE_WIDTH = 28
        IMAGE_HEIGHT = 28

        if dataset_name == DATASET_CELEBA_NAME:
            self.image_mode = 'RGB'
            image_channels = 3

        elif dataset_name == DATASET_MNIST_NAME:
            self.image_mode = 'L'
            image_channels = 1

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels

    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        IMAGE_MAX_VALUE = 255

        current_index = 0
        while current_index + batch_size <= self.shape[0] - self.shape[0]*0.2:
            data_batch = get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode)

            current_index += batch_size

            yield data_batch