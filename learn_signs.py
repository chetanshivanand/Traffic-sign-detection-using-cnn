import numpy as np
import cv2
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.1
Train_SIZE = 0.7
Validate_SIZE = 0.2

cur_path = os.getcwd()

def main():

    # Check command-line arguments
    #if len(sys.argv) not in [2, 3]:
        #sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data('gtsrb')
    print(len(images))
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)

    # train is now 70% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), train_size=Train_SIZE+Validate_SIZE, test_size=TEST_SIZE, random_state=42
    )

    # test is now 10% of the initial data set
    # validation is now 20% of the initial data set
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.1, random_state=42)

    print(len(x_train))
    print(len(x_test))


    # Get a compiled neural network
    model = get_model(x_train)
    # Save model to file
    model.save("Saved_model.h5")
    # simple early stopping
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
   # mc = ModelCheckpoint(model, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='Saved_model.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    # Fit model on training data
    history =model.fit(x_train, y_train,batch_size=32, epochs=EPOCHS, validation_split=0.2, verbose=1, callbacks=my_callbacks)
    #history = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_val, y_val))



    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)



    # Save model to file
    model.save("my_model.h5")

    # evaluate the model
    saved_model = load_model('my_model.h5')

    _, train_acc = saved_model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = saved_model.evaluate(x_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    # plotting graphs for accuracy
    plt.figure()
    N = np.arange(0, EPOCHS)
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    #plt.savefig(args["plot"])
    #plt.plot(history.history['accuracy'], label='training accuracy')
    #plt.plot(history.history['val_accuracy'], label='val accuracy')
    #plt.title('Accuracy')
    #plt.xlabel('epochs')
    #plt.ylabel('accuracy')
    #plt.legend()
    plt.show()

    #plt.figure(1)
    #plt.plot(history.history['loss'], label='training loss')
    #plt.plot(history.history['val_loss'], label='val loss')
    #plt.title('Loss')
    #plt.xlabel('epochs')
    #plt.ylabel('loss')
    #plt.legend()
    #plt.show()
    # Label Overview
    classes = {'Speed limit (20km/h)',
               'Speed limit (30km/h)',
               'Speed limit (50km/h)',
               'Speed limit (60km/h)',
               'Speed limit (70km/h)',
               'Speed limit (80km/h)',
               'End of speed limit (80km/h)',
               'Speed limit (100km/h)',
               'Speed limit (120km/h)',
               'No passing',
               'No passing veh over 3.5 tons',
               'Right-of-way at intersection',
               'Priority road',
               'Yield',
               'Stop',
               'No vehicles',
               'Veh > 3.5 tons prohibited',
               'No entry',
               'General caution',
               'Dangerous curve left',
               'Dangerous curve right',
               'Double curve',
               'Bumpy road',
               'Slippery road',
               'Road narrows on the right',
               'Road work',
               'Traffic signals',
               'Pedestrians',
               'Children crossing',
               'Bicycles crossing',
               'Beware of ice/snow',
               'Wild animals crossing',
               'End speed + passing limits',
               'Turn right ahead',
               'Turn left ahead',
               'Ahead only',
               'Go straight or right',
               'Go straight or left',
               'Keep right',
               'Keep left',
               'Roundabout mandatory',
               'End of no passing',
               'End no passing veh > 3.5 tons'}
    x_test = np.array(x_test)
    pred = model.predict_classes(x_test)
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(x_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=classes))
    y_test = np.argmax(y_test, axis=1)

    #Accuracy with the test data
    #print(accuracy_score(y_test, pred))

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    data = []
    labels = []
    # Retrieving the images and their labels
    for i in range(NUM_CATEGORIES):
        path = os.path.join(cur_path, 'gtsrb', str(i))
        images = os.listdir(path)

        for a in images:
            try:
                #image = Image.open(path + '\\' + a)
                image = cv2.imread(path + '\\' + a)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), 3)
                #image = image.resize((IMG_WIDTH, IMG_HEIGHT))
                image = np.array(image)
                # (width, height , 3)
                #print(image.shape)
                # sim = Image.fromarray(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")
    # Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data,labels

def get_model(X_train):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Building the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    # Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return  model

if __name__ == "__main__":
    main()
