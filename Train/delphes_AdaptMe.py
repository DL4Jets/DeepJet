

from DeepJetCore.training.training_base import training_base
from keras.layers import Dense, Dropout, Concatenate, LocallyConnected1D, Reshape, Flatten
from keras.models import Model

from Layers import GradientReversal
from Losses import binary_crossentropy_labelweights_Delphes, binary_crossentropy_MConly_Delphes
import keras.backend as K
from keras.layers.core import Reshape



def myDomAdaptModel(Inputs,nclasses,nregclasses,dropoutRate=0.05):
    
    X = Dense(40, activation='relu') (Inputs[0])#reco inputs
    
    #kill this input for now
    #zero = Dense(1,kernel_initializer='zeros',trainable=False)(Inputs[2])
    #X=Concatenate()([X,zero])
    
    X = Dense(20, activation='relu')(X)
    X = Dense(10, activation='relu')(X)
    X = Dense(10, activation='relu')(X)
    Xa= Dense(20, activation='relu')(X)
    
    X = Dense(10, activation='relu')(Xa)
    labelpred = Dense(nclasses, activation='softmax')(X)
    
    Ad = GradientReversal()(Xa)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(10, activation='relu')(Ad)
    Ad = Dense(1,  activation='sigmoid')(Ad)
    
    #make list out of it
    Weight = Reshape((1,nclasses),name='reshape')(Inputs[1])
    # one-by-one apply weight to label
    Weight = LocallyConnected1D(1,1, activation='linear',use_bias=False, 
                                name="weight0") (Weight)
    Weight= Flatten()(Weight)
    Weight = GradientReversal()(Weight)
    
    Ad = Concatenate(name='domada0')([Ad,Weight])
   
    
    
    predictions = [labelpred,Ad]
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    
    
    
#also does all the parsing
train=training_base(testrun=False)


print 'Setting model'
train.setModel(myDomAdaptModel,dropoutRate=0.1)

train.compileModel(learningrate=0.003,
                   loss=[binary_crossentropy_MConly_Delphes,
                         binary_crossentropy_labelweights_Delphes],
                   metrics=['accuracy'],
                   loss_weights=[1.,1.])

print(train.keras_model.summary())

model,history = train.trainModel(nepochs=30, 
                                 batchsize=50, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)

exit()
