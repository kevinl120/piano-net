from preprocess import from_fingering_file

# from keras.losses import categorical_crossentropy
import keras.backend as K
from keras.layers import Input, Dense, Flatten, Conv1D, ReLU, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD

def custom_crossentropy(ytrue, ypred):
    true_reshape = K.reshape(ytrue, (-1, 20, 10))
    pred_reshape = K.reshape(ypred, (-1, 20, 10))
    return K.categorical_crossentropy(true_reshape, pred_reshape)
        

def create_model():
    inputs = Input(shape=(30, 3))
    x = Conv1D(16, 3, padding='same')(inputs)
    x = ReLU()(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    finger_outputs = []
    for i in range(20):
        finger_out = Dense(10, activation='softmax', name='note{}_finger1_dense'.format(i))(x)
        finger_outputs.append(finger_out)
    outputs = concatenate(finger_outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def train():
    model = create_model()
    x_train, y_train = from_fingering_file()
    # opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=custom_crossentropy, optimizer=Adam(), metrics=['accuracy'])

    # import pdb
    # pdb.set_trace()
    # print(model.predict(x_train[0].reshape((-1, 30, 3))))

    model.fit(x=x_train, y=y_train, batch_size=32, epochs=50)

train()