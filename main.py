import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard
from keras import backend as K

K.set_image_dim_ordering('tf')

tbCallBack = TensorBoard(log_dir='./logs',
                         histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)


def build():
    model = Sequential()
    model.add(Conv2D(64,
                     (3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu',
                     input_shape=(27, 18, 1)))
    model.add(Conv2D(64,
                     (3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))
    model.add(Conv2D(128,
                     (3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))
    model.add(Conv2D(128,
                     (3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))
    model.add(Flatten())
    model.add(Dense(64,
                    activation='relu'))
    model.add(Dense(7,
                    activation='softmax'))

    adadelta = Adadelta(lr=1,
                        rho=0.95,
                        epsilon=1e-08,
                        decay=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta)

    return model


def train():
    print('Loading train data!')
    x_train = np.loadtxt("Train_Input.cfg").reshape(56, 27, 18, 1)
    y_train = np.loadtxt("Train_Output.cfg")
    print('Train data loaded!')
    print('Loading test data!')
    x_test = np.loadtxt("Test_Input.cfg").reshape(14, 27, 18, 1)
    y_test = np.loadtxt("Test_Output.cfg")
    print('Test data loaded!')
    model.fit(x_train,
              y_train,
              batch_size=10,
              epochs=25,
              callbacks=[tbCallBack],
              validation_data=(x_test, y_test))
    model.save('model.h5')


def predict():
    x_test = np.loadtxt("Test_Input.cfg").reshape(14, 27, 18, 1)
    predictions = model.predict_on_batch(x_test)
    for prediction in predictions:
        max = prediction[0]
        pos = 0
        for i in range(len(prediction)):
            if prediction[i] > max:
                max = prediction[i]
                pos = i
        print 'Max=%2f Pos=%3d' % (max, pos)


fst_choice = raw_input('Please, enter needed action (load or train): ')
if fst_choice == 'load':
    model = load_model('model.h5')
elif fst_choice == 'train':
    model = build()
    train()

scnd_choice = raw_input('Model is ready! Start prediction? (y or n): ')
if scnd_choice == 'y':
    predict()
elif scnd_choice == 'n':
    exit()
