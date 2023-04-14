# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from mini_vgg_like import mini_VGG_like_bn_after, mini_VGG_like_bn_before
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
import click

######################################################################################################
### based on: https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/
######################################################################################################


@click.command()
# @click.argument('run_name', nargs=1, type=click.STRING, help='run-name: all of the output files will be created based on this name')
@click.argument('run_name', nargs=1)
@click.option('-e', '--epochs', type=click.INT, default=25, help='number of epochs to run [default=25]')
@click.option('-lr', '--initial_learning_rate', type=click.FLOAT, default=0.01, help='initial learning rate [default=0.01]')
@click.option('-b', '--batch_size', type=click.INT, default=32, help='training batch size [default=32]')
@click.option('-bnb', '--bn_before', type=click.BOOL, default=False, help='btach normalization before activation [default=False (BN is after)]')

def main(run_name, epochs, initial_learning_rate, batch_size, bn_before):
	# grab the Fashion MNIST dataset (if this is your first time running
	# this the dataset will be automatically downloaded)
	print("[INFO] loading Fashion MNIST...")
	((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
	 
	# use "channels last" ordering, so the design
	# matrix shape should be: num_samples x rows x columns x depth
	# channels last is the default format in tf.  
	# it can be changed by editing ~/.keras/keras.json or
	# specifying the the data_format in the conv2D layers
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	 
	# scale data to the range of [0, 1]
	trainX = trainX.astype("float32") / 255.0
	testX = testX.astype("float32") / 255.0

	# one-hot encode the training and testing labels
	trainY = to_categorical(trainY, 10)
	testY = to_categorical(testY, 10)

	# initialize the label names
	labelNames = ["top", "trouser", "pullover", "dress", "coat",
		"sandal", "shirt", "sneaker", "bag", "ankle boot"]

	# initialize the optimizer and model
	print("[INFO] compiling model...")
	# opt = SGD(lr=initial_learning_rate, momentum=0.9, decay=initial_learning_rate / epochs)
	opt = Adam()
	if bn_before:
		model = mini_VGG_like_bn_before(width=28, height=28, depth=1, classes=10)
	else:
		model = mini_VGG_like_bn_after(width=28, height=28, depth=1, classes=10)

	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	model.summary()


	# train the network
	print("[INFO] training model...")
	H = model.fit(trainX, trainY,
		validation_data=(testX, testY),
		batch_size=batch_size, epochs=epochs)

	# make predictions on the test set
	preds = model.predict(testX)

	## provide some reports on the training
	plot_model(model, to_file=f'{run_name}_model.png', show_shapes=True, show_layer_names=True)
	model.save(f'{run_name}_model.hdf5')


	# plot the training loss and accuracy
	N = epochs
	# plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(f'{run_name}_learning_curve.png')

	# show a nicely formatted classification report
	print("[INFO] evaluating network...")
	print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
		target_names=labelNames))

	test_loss, test_acc = model.evaluate(testX, testY)
	print(f'\n\nTest Loss: {test_loss : 3.2f}')
	print(f'Test Accuracy: {100 * test_acc : 3.2f}%')

	# plot a confusion matrix
	classes = np.asarray(labelNames)
	testY_classes = classes[testY.argmax(axis=1)]
	testY_decision = classes[preds.argmax(axis=1)]

	cm = confusion_matrix(testY_classes, testY_decision, normalize='true')
	plt.figure(figsize=(20,20))
	ax= plt.subplot()
	sns.heatmap(cm, annot=True, ax = ax)
	ax.set_xlabel('Predicted labels')
	ax.set_ylabel('True labels')
	ax.set_title('Confusion Matrix (normalized by number of examples of true label)') 
	ax.xaxis.set_ticklabels(labelNames); ax.yaxis.set_ticklabels(labelNames)
	plt.savefig(f'{run_name}_confusion_matrix.png')


	## show some test images

	# initialize our list of output images
	images = []

	# randomly select a few testing fashion items
	for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
		# classify the clothing
		probs = model.predict(testX[np.newaxis, i])
		prediction = probs.argmax(axis=1)
		label = labelNames[prediction[0]]
	 
		image = (testX[i] * 255).astype("uint8")

		# initialize the text label color as green (correct)
		color = (0, 255, 0)

		# otherwise, the class label prediction is incorrect
		if prediction[0] != np.argmax(testY[i]):
			color = (0, 0, 255)
	 
		# merge the channels into one image and resize the image from
		# 28x28 to 96x96 so we can better see it and then draw the
		# predicted label on the image
		image = cv2.merge([image] * 3)
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
		cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
			color, 2)

		# add the image to our list of output images
		images.append(image)

	# construct the montage for the images
	montage = build_montages(images, (96, 96), (4, 4))[0]

	# show the output montage
	# cv2.imshow("Fashion MNIST", montage)
	# cv2.waitKey(0)

	cv2.imwrite('fashion_mnist.png', montage)

if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

