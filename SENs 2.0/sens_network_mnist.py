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

	
	# input for the feature picker
	embedded_feature_input_tf = tf.placeholder(tf.int32, \
							shape=(None),\
							name='embedded_feature_input')


	# learning rate
	learning_rate_tf = tf.placeholder(tf.float32, \
								  shape=None,\
								  name='learning_rate')


	return input_images_tf, target_output_images_tf, embedded_image_input_tf, embedded_feature_input_tf, learning_rate_tf




# encoding network
def encoder(input_images_tf, embedded_image_dim_tf, reuse=False, dropout_value=0.8):
	
	with tf.variable_scope('encoder', reuse=reuse):
		
		with tf.variable_scope('conv_1'):
			# Input layer is 28x28x1
			conv1 = tf.layers.conv2d(input_images_tf, 64, 5, strides=1, padding='same', \
				kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
			relu1 = leaky_relu(conv1)
			# 28x28x64
		
		with tf.variable_scope('conv_2'):
			conv2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same',\
							kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
			bn2 = tf.layers.batch_normalization(conv2, training=True)
			relu2 = leaky_relu(bn2)
			relu2 = tf.nn.dropout(relu2,dropout_value)
			# 14x14x128
		
		with tf.variable_scope('conv_3'):
			conv3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same',\
									kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
			bn3 = tf.layers.batch_normalization(conv3, training=True)
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
def decoder(embedding_tf, output_image_channel_dim, is_train=True,name='decoder',dropout_value=0.8):

	with tf.variable_scope(name, reuse=not is_train):

		with tf.variable_scope('dense'):
			# First fully connected layer
			x1 = tf.layers.dense(embedding_tf, 7*7*512,\
									kernel_initializer=tf.contrib.layers.xavier_initializer())
			# Reshape it to start the convolutional stack
			x1 = tf.reshape(x1, (-1, 7, 7, 512))
			x1 = tf.layers.batch_normalization(x1, training=is_train)
			x1 = leaky_relu(x1)
			x1 = tf.nn.dropout(x1,dropout_value)
			# 7x7x512 now
		
		with tf.variable_scope('deconv_1'):
			x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2, padding='same',\
									kernel_initializer=tf.contrib.layers.xavier_initializer())
			x2 = tf.layers.batch_normalization(x2, training=is_train)
			x2 = leaky_relu(x2)
			x2 = tf.nn.dropout(x2,dropout_value)
			# 14x14x256 now
		
		with tf.variable_scope('deconv_2'):
			x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding='same',\
									kernel_initializer=tf.contrib.layers.xavier_initializer())
			x3 = tf.layers.batch_normalization(x3, training=is_train)
			x3 = leaky_relu(x3)
			x3 = tf.nn.dropout(x3,dropout_value)
			# 28x28x256 now
		
		with tf.variable_scope('deconv_3'):
			# Output layer
			logits = tf.layers.conv2d_transpose(x3, output_image_channel_dim, 5, strides=1, padding='same',\
									kernel_initializer=tf.contrib.layers.xavier_initializer())
			# 28x28x3 now
			
			out = tf.sigmoid(logits)
	
	return out, logits


def discriminator(embedding_tf, reuse=False, dropout_value=0.8, out_size=1, name='discriminator'):

	with tf.variable_scope(name):

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
			logits = tf.layers.dense(dense2,out_size,\
				kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=reuse)

			out = tf.sigmoid(logits)

	return out, logits


def generator(generator_input_tf, feature_input_tf, embedded_image_dim, reuse=False, dropout_value=0.5):

	with tf.variable_scope('generator'):
		inputs_tf = tf.concat((tf.reshape(generator_input_tf,(-1,embedded_image_dim)),feature_input_tf),axis=1)

		print(inputs_tf)
		with tf.variable_scope('dense_1'):
			dense1 = tf.layers.dense(inputs_tf,embedded_image_dim,\
				kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=reuse)
			dense1 = leaky_relu(dense1)
			dense1 = tf.nn.dropout(dense1,dropout_value)

		with tf.variable_scope('dense_2'):
			dense2 = tf.layers.dense(dense1,embedded_image_dim,\
				kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=reuse)
			dense2 = leaky_relu(dense2)
			dense2 = tf.nn.dropout(dense2,dropout_value)

		with tf.variable_scope('out'):
			logits = tf.layers.dense(dense2,embedded_image_dim,\
				kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=reuse)

			out = tf.sigmoid(logits)

	return out, logits



def model_loss(input_images_tf, output_image_channel_dim, target_output_images_tf, embedded_image_dim, generator_image_input_tf, embedded_feature_input_tf, smooth=0.1):

	print(embedded_feature_input_tf)
	embedded_feature_input_one_hot_tf = tf.reshape(tf.one_hot(indices=embedded_feature_input_tf, depth=embedded_image_dim),shape=(-1,embedded_image_dim))

	print(embedded_feature_input_one_hot_tf)
	print(generator_image_input_tf)
	# create image generator (from randon embedding input)
	embedded_image_input_tf, embedded_logits_input_tf = generator(generator_image_input_tf, embedded_feature_input_one_hot_tf, embedded_image_dim, reuse=False)
	image_generated_output_tf, image_generated_logits_tf = decoder(embedded_image_input_tf, output_image_channel_dim, is_train=True)
	image_generated_output_tf = tf.identity(image_generated_output_tf, name='generated_image_output')

	# Send to discriminator for fake and real
	image_generated_encoded_real_tf, logits_generated_encoded_real_tf = encoder(target_output_images_tf, embedded_image_dim, reuse=False)

	# Add distribution of encoded real images
	tf.summary.histogram("encoded_real",image_generated_encoded_real_tf)

	# Encode images from generator->decoder
	image_generated_encoded_fake_tf, logits_generated_encoded_fake_tf = encoder(image_generated_output_tf, embedded_image_dim, reuse=True)


	# Add distribution of encoded fake images
	tf.summary.histogram("encoded_fake",image_generated_encoded_fake_tf)

	# Create discriminator
	discriminator_output_fake_tf, discriminator_logits_fake_tf = discriminator(image_generated_encoded_fake_tf, reuse=False)
	discriminator_output_real_tf, discriminator_logits_real_tf = discriminator(image_generated_encoded_real_tf, reuse=True)




	# discriminator loss
	discriminator_loss_fake_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logits_fake_tf, \
                                                labels=tf.zeros_like(discriminator_output_fake_tf)))
	discriminator_loss_real_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logits_real_tf, \
                                                labels=tf.ones_like(discriminator_output_real_tf) * (1-smooth)))
	discriminator_cost_tf = discriminator_loss_fake_tf + discriminator_loss_real_tf


	# Feature detector
	feature_det_output_tf, feature_det_logits_tf = discriminator(image_generated_encoded_fake_tf, reuse=False, name='feature_detector', out_size=embedded_image_dim)

	feature_det_real_output_tf, feature_det_real_logits_tf = discriminator(image_generated_encoded_real_tf, reuse=True, name='feature_detector', out_size=embedded_image_dim)

	arg_max_det_real_output_tf = tf.argmax(feature_det_real_logits_tf,axis=1)
	one_hot_det_real_output_tf = tf.one_hot(arg_max_det_real_output_tf,depth=embedded_image_dim)

	print(feature_det_logits_tf)
	print(embedded_feature_input_one_hot_tf)
	feature_det_real_loss_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=feature_det_real_logits_tf, labels=one_hot_det_real_output_tf))
	feature_det_loss_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=feature_det_logits_tf, labels=embedded_feature_input_one_hot_tf))
	
	# Generator cost, add in the feature detector cost
	generator_loss_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logits_fake_tf, labels=tf.ones_like(discriminator_output_fake_tf)))# * (1-smooth)))
	generator_cost_tf = generator_loss_tf + feature_det_loss_tf + feature_det_real_loss_tf

	# Create autoencoder
	# takes image input and encodes / embeds
	image_encoder_output_tf, image_encoder_logits_tf = encoder(input_images_tf, embedded_image_dim, reuse=True)
	image_encoder_output_tf = tf.identity(image_encoder_output_tf, name='image_encoder_output')

	# takes image embedding and decodes / generates / autoencodes
	image_decoder_output_tf, image_decoder_logits_tf = decoder(image_encoder_output_tf, output_image_channel_dim, is_train=False)
	image_decoder_output_tf = tf.identity(image_decoder_output_tf, name='image_decoder_output')
	# autoencoder cost using MSE
	autoencoder_cost_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_output_images_tf, logits=image_decoder_logits_tf))
	#autoencoder_cost_tf = tf.reduce_mean(tf.squared_difference(target_output_images_tf, image_decoder_output_tf))
	

	return autoencoder_cost_tf, image_decoder_output_tf, discriminator_cost_tf, discriminator_output_real_tf, image_generated_output_tf, discriminator_loss_fake_tf, discriminator_loss_real_tf, generator_cost_tf


def model_opt(autoencoder_cost_tf, discriminator_cost_tf, generator_cost_tf, learning_rate_tf, beta1):
	# Get weights and bias to update
	t_vars = tf.trainable_variables()

	autoencoder_variables_tf = [var for var in t_vars if (var.name.startswith('decoder') or var.name.startswith('encoder'))]
	discriminator_variables_tf = [var for var in t_vars if (var.name.startswith('discriminator') or var.name.startswith('encoder'))]
	generator_variables_tf = [var for var in t_vars if (var.name.startswith('generator') or var.name.startswith('feature_detector'))]

	# Optimize
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		autoencoder_optimizer_tf = tf.train.AdamOptimizer(learning_rate_tf, beta1=beta1).minimize(autoencoder_cost_tf, var_list=autoencoder_variables_tf)
		discriminator_optimizer_tf = tf.train.AdamOptimizer(learning_rate_tf, beta1=beta1).minimize(discriminator_cost_tf, var_list=discriminator_variables_tf)
		generator_optimizer_tf = tf.train.AdamOptimizer(learning_rate_tf, beta1=beta1).minimize(generator_cost_tf, var_list=generator_variables_tf)
	return autoencoder_optimizer_tf, discriminator_optimizer_tf, generator_optimizer_tf