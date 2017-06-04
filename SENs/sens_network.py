import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.1, name='leaky_relu'):
	return tf.maximum(x, alpha * x, name=name)

# Create the model inputs
def model_inputs(image_width, image_height, image_channels, embedded_image_dim):


	
	#input images
	input_images_tf = tf.placeholder(tf.float32, \
									   shape=(None,image_width,\
											  image_height,\
											  image_channels),\
									   name='input_images')
	
	# target output images for autoencoder
	target_output_images_tf = tf.placeholder(tf.float32, \
										shape=(None,image_width,\
											  image_height,\
											  image_channels),\
										name='target_output_images')
	
	# input embedding data for generator
	embedded_image_input_tf = tf.placeholder(tf.float32, \
						   shape=(None,embedded_image_dim),\
						   name='embedded_image_input')

	
	
	# learning rate
	learning_rate_tf = tf.placeholder(tf.float32, \
								  shape=None,\
								  name='learning_rate')


	return input_images_tf, target_output_images_tf, embedded_image_input_tf, learning_rate_tf




# encoding network
def encoder(input_images_tf, embedded_image_dim_tf, reuse=False):
	
	noise_std = 0.05
	dropout_value = 0.8

	with tf.variable_scope('encoder', reuse=reuse):
		
		with tf.variable_scope('conv_1'):
			# Input layer is 28x28x1
			conv1 = tf.layers.conv2d(input_images_tf, 64, 5, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
			conv1 += tf.truncated_normal(shape=tf.shape(conv1),mean=0.0,stddev=noise_std, dtype=tf.float32)
			relu1 = leaky_relu(conv1)
			# 14x14x64
		
		with tf.variable_scope('conv_2'):
			conv2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same',\
							kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
			bn2 = tf.layers.batch_normalization(conv2, training=True)
			bn2 += tf.truncated_normal(shape=tf.shape(bn2),mean=0.0,stddev=noise_std, dtype=tf.float32)
			relu2 = leaky_relu(bn2)
			relu2 = tf.nn.dropout(relu2,dropout_value)
			# 7x7x128
		
		with tf.variable_scope('conv_3'):
			conv3 = tf.layers.conv2d(relu2, 256, 5, strides=1, padding='same',\
									kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
			bn3 = tf.layers.batch_normalization(conv3, training=True)
			bn3 += tf.truncated_normal(shape=tf.shape(bn3),mean=0.0,stddev=noise_std, dtype=tf.float32)
			relu3 = leaky_relu(bn3)
			relu3 = tf.nn.dropout(relu3,dropout_value)
			#7x7x256
		
		with tf.variable_scope('out'):
			# Flatten it
			flat = tf.reshape(relu3, (-1, 7*7*256))
			
			# Dropout
			dropout = tf.nn.dropout(flat,dropout_value)
			
			# Logits
			logits = tf.layers.dense(dropout, embedded_image_dim_tf,\
							kernel_initializer=tf.contrib.layers.xavier_initializer())
			out = tf.sigmoid(logits)
		
		
		
	return out, logits


# Generator
def generator(embedding_tf, output_image_channel_dim, is_train=True,name='generator_default'):

	dropout_value = 0.8
	with tf.variable_scope(name, reuse=not is_train):

		with tf.variable_scope('dense'):
			# First fully connected layer
			x1 = tf.layers.dense(embedding_tf, 7*7*512,\
									kernel_initializer=tf.contrib.layers.xavier_initializer())
			
		with tf.variable_scope('deconv_1'):
			# Reshape it to start the convolutional stack
			x1 = tf.reshape(x1, (-1, 7, 7, 512))
			x1 = tf.layers.batch_normalization(x1, training=is_train)
			x1 = leaky_relu(x1)
			x1 = tf.nn.dropout(x1,dropout_value)
			# 7x7x512 now
		
		with tf.variable_scope('deconv_2'):
			x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2, padding='same',\
									kernel_initializer=tf.contrib.layers.xavier_initializer())
			x2 = tf.layers.batch_normalization(x2, training=is_train)
			x2 = leaky_relu(x2)
			x2 = tf.nn.dropout(x2,dropout_value)
			# 14x14x256 now
		
		with tf.variable_scope('deconv_3'):
			x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding='same',\
									kernel_initializer=tf.contrib.layers.xavier_initializer())
			x3 = tf.layers.batch_normalization(x3, training=is_train)
			x3 = leaky_relu(x3)
			x3 = tf.nn.dropout(x3,dropout_value)
			# 28x28x256 now
		
		with tf.variable_scope('out'):
			# Output layer
			logits = tf.layers.conv2d_transpose(x3, output_image_channel_dim, 5, strides=1, padding='same',\
									kernel_initializer=tf.contrib.layers.xavier_initializer())
			# 28x28x3 now
			
			out = tf.sigmoid(logits)
	
	return out, logits

def discriminator(embedding_tf, reuse=False):

	dropout_value = 0.8

	with tf.variable_scope('discriminator'):

		with tf.variable_scope('dense_1'):
			dense1 = tf.layers.dense(embedding_tf,256,\
				kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=reuse)
			dense1 = leaky_relu(dense1)
			dense1 = tf.nn.dropout(dense1,dropout_value)

		with tf.variable_scope('dense_2'):
			dense2 = tf.layers.dense(dense1,256,\
				kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=reuse)
			dense2 = leaky_relu(dense2)
			dense2 = tf.nn.dropout(dense2,dropout_value)

		with tf.variable_scope('out'):
			logits = tf.layers.dense(dense2,1,\
				kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=reuse)

			out = tf.sigmoid(logits)

	return out, logits


def model_loss(input_images_tf, output_image_channel_dim, target_output_images_tf, embedded_image_dim, embedded_image_input_tf):

	smooth = 0.1
	
	# create image generator (from randon embedding input)
	image_generated_output_tf, image_generated_logits_tf = generator(embedded_image_input_tf, output_image_channel_dim, is_train=True, name='generator')
	# Send to discriminator for fake and real
	image_generated_encoded_real_tf, logits_generated_encoded_real_tf = encoder(target_output_images_tf, embedded_image_dim, reuse=False)

	tf.summary.histogram("encoded_real",image_generated_encoded_real_tf)

	image_generated_encoded_fake_tf, logits_generated_encoded_fake_tf = encoder(image_generated_output_tf, embedded_image_dim, reuse=True)

	discriminator_output_fake_tf, discriminator_logits_fake_tf = discriminator(image_generated_encoded_fake_tf, reuse=False)
	discriminator_output_real_tf, discriminator_logits_real_tf = discriminator(image_generated_encoded_real_tf, reuse=True)

	# discriminator loss
	discriminator_loss_fake_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logits_fake_tf, \
                                                labels=tf.zeros_like(discriminator_output_fake_tf)))
	discriminator_loss_real_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logits_real_tf, \
                                                labels=tf.ones_like(discriminator_output_real_tf) * (1-smooth)))
	discriminator_cost_tf = discriminator_loss_fake_tf + discriminator_loss_real_tf

	generator_cost_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logits_fake_tf, labels=tf.ones_like(discriminator_output_fake_tf)))


	# Create autoencoder
	# takes image input and encodes / embeds
	image_encoder_output_tf, image_encoder_logits_tf = encoder(input_images_tf, embedded_image_dim, reuse=True)
	# takes image embedding and decodes / generates / autoencodes
	image_decoder_output_tf, image_decoder_logits_tf = generator(image_encoder_output_tf, output_image_channel_dim, is_train=False, name='generator')
	
	# autoencoder cost
	autoencoder_cost_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_output_images_tf, logits=image_decoder_logits_tf))


	
	return autoencoder_cost_tf, image_decoder_output_tf, discriminator_cost_tf, discriminator_output_real_tf, image_generated_output_tf, discriminator_loss_fake_tf, discriminator_loss_real_tf, generator_cost_tf


def model_opt(autoencoder_cost_tf, discriminator_cost_tf, generator_cost_tf, learning_rate_tf, beta1):
	# Get weights and bias to update
	t_vars = tf.trainable_variables()

	autoencoder_variables_tf = [var for var in t_vars if (var.name.startswith('generator') or var.name.startswith('encoder'))]
	discriminator_variables_tf = [var for var in t_vars if (var.name.startswith('discriminator') or var.name.startswith('encoder'))]
	generator_variables_tf = [var for var in t_vars if (var.name.startswith('generator'))]

	# Optimize
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		autoencoder_optimizer_tf = tf.train.AdamOptimizer(learning_rate_tf, beta1=beta1).minimize(autoencoder_cost_tf, var_list=autoencoder_variables_tf)
		discriminator_optimizer_tf = tf.train.AdamOptimizer(learning_rate_tf, beta1=beta1).minimize(discriminator_cost_tf, var_list=discriminator_variables_tf)
		generator_optimizer_tf = tf.train.AdamOptimizer(learning_rate_tf, beta1=beta1).minimize(generator_cost_tf, var_list=generator_variables_tf)
	return autoencoder_optimizer_tf, discriminator_optimizer_tf, generator_optimizer_tf