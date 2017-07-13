## Shared encodings/embeddings/latent vectors

In this repo I'll be playing with a few deep network architectures as well as supervised and unsupervised training.  The idea that I'd like to explore is sharing the latent vector of an autoencoder, or alternately the encoding/embedding, with several other networks.  These networks will all share the embedding layer(s) and help train how the layer is encoded to ensure that it contains all of the relevant information.

Similar to how a gan competes two networks against each other, these networks will cooperate (and compete) with each other to create the best encoding.  I'd like to see how this concept works with gans, autoencoders (or denoising autoencoders), classifiers, etc.  Later perhaps put in a variational autoencoder and see what else comes about.  I'm unaware of how much this type of network is used, their name or if they work/have any benefits but for the moment I'll call them SENs (Shared Encoding Networks).  This page will be very much a work in progress :)

Most of the work will start with MNIST, depending on how it all goes will try with more interesting data sets (like faces !).

Initial code base was from an assignment in the Udacity deep learning nanodegree.  [Udacity deep learning repo.](https://github.com/udacity/deep-learning)

Very cool blog for generating images:
[Generating Large Images from Latent Vectors.](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/)

Code in [SENs_MNIST.ipynb](https://github.com/drburke/tensorflow/blob/master/SENs/SENs_MNIST.ipynb)

#### Initial Architecture, Network Pieces

Initial architecture will be a set of four networks.  

###### Encoding Network

Inputs : B&W Image (MNIST 28x28x1)

Outputs : Single layer of nodes, sigmoid activated, embedding_dim in size

This will be a convolutional network, currently three convolutional layers followed by a single fully connected layer which is the output.

###### Decoding Network
Inputs : Embedding

Outputs : B&W Image (Same as input for encoding network)

Similar to encoding network, using deconv layers and sigmoid output.

###### Generator Network
Inputs : Random vector used to generate an embedding.  

Outputs : Single layer of embedding nodes

At the moment a couple fully connected layers.  Initially I had tried generating random embeddings and skipping the generator network entirely.  This didn't work terribly well for the GAN, but once the rest of the network is working well I may return to that.  I expect this is because the distribution that I was using for the random embedding had different properties to the embedding created by the encoder network.  This could perhaps be fixed using a variational autoencoder which would ensure a unit gaussian distribution.   

###### Discriminator Network
Inputs : Embedding

Outputs : Single node of True/False for Real MNIST image or fake generated image.

###### Full Networks
The autoencoder combines the encoder and decoder network.  Generator combines the generator, decoder, encoder and discriminator networks.  I could have also used generator to embedding to discriminator but had less success with this approach.  It may have been too easy for the discriminator in such a direct route, passing through the decoder/encoder steps may help.

#### Implementation

For the most part I am using leaky relus, these seem to be a good way to get gradients moving efficiently.  The final layers of each network depend on what their goal is but are mostly sigmoids for the moment.  Dropout is heavily used in most layers.  Batch normalization is also used between most layers of the encoder and decoder- this significantly helps the model.

I'm adding noise to the autoencoder which I am hoping will improve the robustness of the networks.  This will be explored a bit.  Also data augmentation for the autoencoder will randomly rotate and flip the MNIST digits.  The data will not be augmented for the discriminator which will get noise free and unrotated/flipped data.

#### Initial Results

The training runs through each network showing them different images.  First run is simply training the autoencoder + noise and augmentation.  Second run is for the autoencoder and generator. Third run is training the autoencoder and discriminator.  This will be counted as one iteration of a single batch.  

For MNIST results I found using very small batches worked best with the GAN, currently using size of 4 which is very nearly online learning.  Faces seems to be the opposite.  I haven't played extensively with all of the network parameters...  After 500 batches we see the autoencoder has learned about de-noising and some cloudy numbers.  The generator is not doing very well.

First set of 25 images shows the noised images to be denoised / autoencoded.  Second set of images shows the denoised results.  Last set are the generated images.

[image1]: ./mnist_results_500_batches.png
![MNIST results after 500 iterations][image1]

After an entire epoch the autoencoder seems to have figured itself out quite nicely.  Generator numbers aren't terribly convincing although there is at least some diversity in the results.

[image2]: ./mnist_results_1_epoch.png
![MNIST results after 1 epoch][image2]

With 4 epochs down the generator is creating some pretty decent numbers.  Not too bad..

[image3]: ./mnist_results_4_epochs.png
![MNIST results after 4 epochs][image3]


#### Explore the embedding space
There are some very cool blogs showing an exploration of this embedding space.  Slowly varying the embedding can create animations with morphing outputs.  One can also embedding vectors from e.g. thin numbers to thick numbers and use this to alter or generate new images with desired features.



#### Add classifiers
It would be interesting to add a classifier into the SENs.  Using the encodings from the other network may help it identify what the object/number is.  If the generator was provided with a one hot input as well as the random vector, this input could also be compared with the classifier result of the generated image and added to the loss.  This would create a directed generator.

Having trained the autoencoder and generator/discriminator I've tried using the resulting encoder followed by a few dense layers to mnist classifier.  Giving it only 100 labelled examples it seems to get up to ~75% accuracy on 1000 examples for which it hasn't seen the label - but has been seen before by the autoencoder.  With 500 I have gotten to near ~90% accuracy testing on 5000 new examples.  Not tooo terrible.

Another thing that I've tried is feeding the network 1000 examples and giving it the correct answers as the argmax of it's final output - essentially telling the model that it is correct.  Using this on a classifier that is decently trained already can give a bit of a boost to the accuracy.  With 100 labelled examples I have achieved up to 86%.

#### Further

I would like to continue exploring some of the ideas of sharing embeddings, particularly with improving supervised learning with minimal data augmented by a large set of unsupervised learning.  
