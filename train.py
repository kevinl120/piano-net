from preprocess import from_fingering_file

import keras.backend as K
from keras.layers import Input, Dense, Flatten, Conv1D, ReLU, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD


def create_model():
    inputs = Input(shape=(30, 3))
    x = inputs
    x = Conv1D(64, 3, padding='same')(x)
    x = ReLU()(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = ReLU()(x)
    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)
    finger_out = Dense(10, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=finger_out)
    return model

def train():
    model = create_model()
    x_train, y_train = from_fingering_file(start_file=0)
    opt = Adam(learning_rate=0.001)
#     opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    print(model.predict(x_train[0].reshape((-1, 30, 3))))
    # import pdb
    # pdb.set_trace()

    model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_split=0.2)
    
train()