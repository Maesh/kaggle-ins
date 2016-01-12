
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from sklearn import svm, cross_validation, preprocessing, metrics

from kaggletdevries import getDummiesInplace, pdFillNAN, make_dataset, pdStandardScaler

if __name__ == '__main__':

	rs = 19683
	maxlen = 100  # cut texts after this number of words (among top max_features most common words)
	batch_size = 32

	train, test, labels = make_dataset(useDummies = True, 
    fillNANStrategy = "mean", useNormalization = True)

	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(train, labels, \
			test_size=0.2, random_state=rs)

	print('Build model...')
	# model = Graph()
	# model.add_input(name='input', input_shape=(X_train.shape[1],), dtype=int)
	# model.add_node(Embedding(X_train.shape[0], 128, input_length=X_train.shape[1]),
	#                name='embedding', input='input')
	# model.add_node(LSTM(64), name='forward', input='embedding')
	# model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
	# model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
	# model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
	# model.add_output(name='output', input='sigmoid')

	# # try using different optimizers and different optimizer configs
	# model.compile('adam', {'output': 'binary_crossentropy'})

	# print('Train...')
	# model.fit({'input': X_train, 'output': y_train},
	#           batch_size=batch_size,
	#           nb_epoch=4)
	# acc = accuracy(y_test,
	#                np.round(np.array(model.predict({'input': X_test},
	#                                                batch_size=batch_size)['output'])))
	# print('Test accuracy:', acc)

	model = Sequential()
	model.add(SimpleRNN(1, 100))
	model.add(Dense(100, 1, activation = "relu"))
	model.compile(loss="mean_squared_error", optimizer = "sgd")