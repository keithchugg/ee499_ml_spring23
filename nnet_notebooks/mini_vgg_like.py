# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


def mini_VGG_like_bn_after(width, height, depth, classes):
	model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	# first CONV => RELU => CONV => RELU => POOL layer set
	model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# second CONV => RELU => CONV => RELU => POOL layer set
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))

	# return the constructed network architecture
	return model

def mini_VGG_like_bn_before(width, height, depth, classes):
	model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	# first CONV => RELU => CONV => RELU => POOL layer set
	model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Activation("relu"))
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# second CONV => RELU => CONV => RELU => POOL layer set
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Activation("relu"))
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Dropout(0.5))

	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))

	# return the constructed network architecture
	return model

