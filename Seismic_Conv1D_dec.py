# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:48:13 2018

@author: Sediment
"""

# -*- coding: utf-8 -*-
'''
Keras implementation of deep embedder to improve clustering, inspired by:
"Unsupervised Deep Embedding for Clustering Analysis" (Xie et al, ICML 2016)

Definition can accept somewhat custom neural networks. Defaults are from paper.
'''
import sys
import numpy as np
import pandas as pd
import keras.backend as K
from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, BatchNormalization, Activation, Flatten, UpSampling1D, Reshape
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Nadam
from keras.regularizers import l2
from sklearn.preprocessing import normalize
from keras.callbacks import LearningRateScheduler
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle

class ClusteringLayer(Layer):
    '''
    Clustering layer which converts latent space Z of input layer
    into a probability vector for each cluster defined by its centre in
    Z-space. Use Kullback-Leibler divergence as loss, with a probability
    target distribution.
    # Arguments
        output_dim: int > 0. Should be same as number of clusters.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        alpha: parameter in Student's t-distribution. Default is 1.0.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        # kmeans cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0/(1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2))**2 /self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddingClustering(object):
    def __init__(self,
                 n_clusters,
                 input_dim,
                 learning_rate=0.1,
                 encoded=None,
                 decoded=None,
                 alpha=1.0,
                 pretrained_weights=None,
                 cluster_centres=None,
                 batch_size=256,
                 conv_filters=[8, 16, 32],
                 kernel_size=12,
                 Maxpooling_size=2,
                 LatentSpace_Z=25,
                 finetune_epochs=5,
                 **kwargs):

        super(DeepEmbeddingClustering, self).__init__()

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.encoded = encoded
        self.decoded = decoded
        self.alpha = alpha
        self.pretrained_weights = pretrained_weights
        self.cluster_centres = cluster_centres
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.iters_lr_update = 6000
        self.lr_change_rate = 0.1
        self.finetune_epochs = finetune_epochs
        
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.Maxpooling_size = Maxpooling_size
        self.LatentSpace_Z = LatentSpace_Z

        self.encoders = []
        self.decoders = []
        
        input_data = Input(shape=(self.input_dim, 1))

        x = Conv1D(self.conv_filters[0], (self.kernel_size), activation='relu', padding='same')(input_data)
#        x = BatchNormalization()(x)
#        x = Activation('relu')(x)
        x = MaxPooling1D((self.Maxpooling_size), padding='same')(x)
        x = Conv1D(self.conv_filters[1], (self.kernel_size), activation='relu', padding='same')(x)
#        x = BatchNormalization()(x)
#        x = Activation('relu')(x)
        x = MaxPooling1D((self.Maxpooling_size), padding='same')(x)
        x = Conv1D(self.conv_filters[2], (self.kernel_size), activation='relu', padding='same')(x)
#        x = BatchNormalization()(x)
#        x = Activation('relu')(x)
        x = MaxPooling1D((self.Maxpooling_size), padding='same')(x)
        # at this point the representation is (16 x conv_filters) i.e. 128-dimensional
        x = Flatten()(x)

        # at this point the representation is (6) i.e. 128-dimensional
        encoded = Dense(LatentSpace_Z, activation='relu')(x)
        
        # 256 = input_data / ((2^maxpool_num) * conv_fileters * 4)
        x = Dense(self.input_dim // (2**3) * self.conv_filters[2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                  bias_initializer='zeros', activation='relu')(encoded)
        x = Reshape((self.input_dim // (2**3), self.conv_filters[2]))(x)      # 16 * 2 * 2 * 2 = 128, 多少个maxpool就与多少个2相乘

        x = Conv1D(self.conv_filters[2], (self.kernel_size), activation='relu', padding='same')(x)
#        x = BatchNormalization()(x)
#        x = Activation('relu')(x)
        x = UpSampling1D((self.Maxpooling_size))(x)
        x = Conv1D(self.conv_filters[1], (self.kernel_size), activation='relu', padding='same')(x)
#        x = BatchNormalization()(x)
#        x = Activation('relu')(x)
        x = UpSampling1D((self.Maxpooling_size))(x)
        x = Conv1D(self.conv_filters[0], (1), activation='relu')(x)
#        x = BatchNormalization()(x)
#        x = Activation('relu')(x)
        x = UpSampling1D((self.Maxpooling_size))(x)
        decoded = Conv1D(1, (self.kernel_size), activation='relu', padding='same')(x)
              
        self.autoencoder = Model(input_data, decoded)
        self.autoencoder.summary()
        
        self.encoder = Model(input_data, encoded)
        
        # build the end-to-end autoencoder for finetuning
        # Note that at this point dropout is discarded

        self.encoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))

        if cluster_centres is not None:
            assert cluster_centres.shape[0] == self.n_clusters
            assert cluster_centres.shape[1] == self.encoder.layers[-1].output_dim

        if self.pretrained_weights is not None:
            self.autoencoder.load_weights(self.pretrained_weights)

    def p_mat(self, q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, X, save_autoencoder=False, finetune_iters=5000):
        if self.pretrained_weights is None:

            iters_per_epoch = int(len(X) / self.batch_size)                    
            print('layerwise pretrain')
            lr_epoch_update = max(1, self.iters_lr_update / float(iters_per_epoch))
            
            def step_decay(epoch):
                initial_rate = self.learning_rate
                factor = int(epoch / lr_epoch_update)
                lr = initial_rate / (10 ** factor)
                return lr
            lr_schedule = LearningRateScheduler(step_decay)
            #update encoder and decoder weights:
            self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=self.finetune_epochs, callbacks=[lr_schedule])

            if save_autoencoder:
                self.autoencoder.save_weights('autoencoder.h5')
        else:
            print('Loading pretrained weights for autoencoder.')
            self.autoencoder.load_weights(self.pretrained_weights)

        # update encoder, decoder
        # TODO: is this needed? Might be redundant...
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        # initialize cluster centres using k-means
        print('Initializing cluster centres with k-means.')
        if self.cluster_centres is None:
            np.random.seed(42) #随机种子，用于初始化聚类中心  
            kmeans = KMeans(n_clusters=self.n_clusters, max_iter=100, n_init=6, precompute_distances='auto', random_state=None, tol=1e-4)
            self.y_pred = kmeans.fit_predict(self.encoder.predict(X))
            self.cluster_centres = kmeans.cluster_centers_
            print ('cluster_centres:\n ', self.cluster_centres)

        # prepare DCEC model
        self.DCEC = Sequential([self.encoder,
                             ClusteringLayer(self.n_clusters,
                                                weights=self.cluster_centres,
                                                name='clustering')])

        self.DCEC.compile(loss='kullback_leibler_divergence', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
        # loss: 'mean_squared_error', 'categorical_crossentropy', 'hinge', 'squared_hinge'
        return

    def visualizeData(self, Z, labels, num_clusters, csv_filename, title):
        '''
        TSNE visualization of the points in latent space Z
        :param Z: Numpy array containing points in latent space in which clustering was performed
        :param labels: True labels - used for coloring points
        :param num_clusters: Total number of clusters
        :param title: filename where the plot should be saved
        :return: None - (side effect) saves clustering visualization plot in specified location
        '''
        print ('Start visualizing Data')
        labels = labels.astype(int)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Z_tsne = tsne.fit_transform(Z)
        fig = plt.figure()
        plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
        plt.colorbar(ticks=range(num_clusters))
#        fig.savefig(title, dpi=fig.dpi)
        fig.savefig(title, dpi=600)
        
        # save t_sne results
        print('Save t_sne results')
        dataframe = pd.DataFrame({'Z_tsne_x':Z_tsne[:, 0], 'Z_tsne_y':Z_tsne[:, 1], 'labels':labels})
        dataframe.to_csv(csv_filename, index=False, sep=',')

    def cluster(self, X, y=None,
                tol=0.001, update_interval=None,
                iter_max=799,
                save_interval=None,
                **kwargs):

        if update_interval is None:
            # 1 epochs
            update_interval = X.shape[0]/self.batch_size
        print('Update interval', update_interval)

        if save_interval is None:
            # 50 epochs
            save_interval = X.shape[0]/self.batch_size*50
        print('Save interval', save_interval)

        assert save_interval >= update_interval

        train = True
        iteration, index = 0, 0
        self.accuracy = []

        while train:
            sys.stdout.write('\r')
            # cutoff iteration
            if iter_max < iteration:
                print('Reached maximum iteration limit. Stopping training.')
                return self.y_pred

            # update (or initialize) probability distributions and propagate weight changes
            # from DCEC model to encoder.
            if iteration % update_interval == 0:
                self.q = self.DCEC.predict(X, verbose=0)
                self.p = self.p_mat(self.q)

                y_pred = self.q.argmax(1)
                delta_label = ((y_pred == self.y_pred).sum().astype(np.float32) / y_pred.shape[0])
                if y is None:
                    print(str(np.round(delta_label*100, 5))+'% change in label assignment')

                if iteration > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    train = False
                    continue
                else:
                    self.y_pred = y_pred

                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.DCEC.layers[0].layers[i].get_weights())
                self.cluster_centres = self.DCEC.layers[-1].get_weights()[0]

            # train on batch
            sys.stdout.write('Iteration %d, ' % iteration)
            if (index+1)*self.batch_size >= X.shape[0]:
                loss = self.DCEC.train_on_batch(X[index*self.batch_size::], self.p[index*self.batch_size::])
                index = 0
                sys.stdout.write('Loss %f\n' % loss)
            else:
                loss = self.DCEC.train_on_batch(X[index*self.batch_size:(index+1) * self.batch_size],
                                               self.p[index*self.batch_size:(index+1) * self.batch_size])
                sys.stdout.write('Loss %f\n' % loss)
                index += 1

            # save intermediate
            if iteration % save_interval == 0:
                z = self.encoder.predict(X)
                pca = PCA(n_components=2).fit(z)
                z_2d = pca.transform(z)
                clust_2d = pca.transform(self.cluster_centres)
                # save states for visualization
                pickle.dump({'z_2d': z_2d, 'clust_2d': clust_2d, 'q': self.q, 'p': self.p},
                            open('c'+str(iteration)+'.pkl', 'wb'))
                # save DCEC model checkpoints
                self.DCEC.save('DCEC_model_'+str(iteration)+'.h5')

            iteration += 1
            sys.stdout.flush()
        return y_pred
