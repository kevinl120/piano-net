from preprocess import from_fingering_file

from keras.layers import Input, Dense, Flatten, Conv1D, ReLU, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

def create_model():
    inputs = Input(shape=(2*context+1, 3))
    x = inputs
    x = Conv1D(128, 3, padding='valid')(x)
    x = ReLU()(x)
    x = Conv1D(256, 3, padding='valid')(x)
    x = ReLU()(x)
    x = Conv1D(512, 3, padding='valid')(x)
    x = ReLU()(x)
    x = Conv1D(1024, 3, padding='valid')(x)
    x = ReLU()(x)
    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    finger_out = Dense((10 if two_hand else 5), activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=finger_out)
    return model

def train():
    model = create_model()
    opt = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    print('accuracy before training:', accuracy_score(np.argmax(y_train, axis=1), np.argmax(model.predict(x_train), axis=1)))
    
#     opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#     plt.hist(np.argmax(y_train, axis=1), weights=np.ones(len(y_train)) / len(y_train))
    
    model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_split=0.2)
    
    return model
    
context = 5
two_hand = False
x_train, y_train = from_fingering_file(context=context, two_hand=two_hand)
print(x_train.shape)
print(y_train.shape)
m = train()