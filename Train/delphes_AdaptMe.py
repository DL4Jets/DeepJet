

from DeepJetCore.training.training_base import training_base
from keras.layers import Dense, Dropout, Concatenate, LocallyConnected1D, Reshape, Flatten
from keras.models import Model

from Layers import GradientReversal
from Losses import binary_crossentropy_labelweights_Delphes, binary_crossentropy_MConly_Delphes
import keras.backend as K
from keras.layers.core import Reshape



def myDomAdaptModel(Inputs,nclasses,nregclasses,dropoutRate=0.05):
    
    X = Dense(60, activation='relu') (Inputs[0])#reco inputs
    
    X = Dropout(dropoutRate)(X)
    X = Dense(60, activation='relu',name='classifier_dense0')(X)
    X = Dropout(dropoutRate)(X)
    X = Dense(60, activation='relu',name='classifier_dense1')(X)
    X = Dropout(dropoutRate)(X)
    X = Dense(60, activation='relu',name='classifier_dense2')(X)
    X = Dropout(dropoutRate)(X)
    Xa= Dense(20, activation='relu',name='classifier_dense3')(X)
    
    X = Dense(10, activation='relu',name='classifier_dense4')(Xa)
    labelpred = Dense(nclasses, activation='softmax',name='classifier_pred')(X)
    
    Ad = GradientReversal(name='da_gradrev0')(Xa)
    Ad = Dense(30, activation='relu',name='da_dense0')(Ad)
    X = Dropout(dropoutRate)(X)
    Ad = Dense(30, activation='relu',name='da_dense1')(Ad)
    X = Dropout(dropoutRate)(X)
    Ad = Dense(30, activation='relu',name='da_dense2')(Ad)
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

train.compileModel(learningrate=0.00003,
                   loss=[binary_crossentropy_MConly_Delphes,
                         binary_crossentropy_labelweights_Delphes],
                   metrics=['accuracy'],
                   loss_weights=[1.,0.])

print(train.keras_model.summary())

model,history = train.trainModel(nepochs=30, 
                                 batchsize=500, 
                                 maxqsize=10)


train.compileModel(learningrate=0.00001,
                   loss=[binary_crossentropy_MConly_Delphes,
                         binary_crossentropy_labelweights_Delphes],
                   metrics=['accuracy'],
                   loss_weights=[1.,1.])


model,history = train.trainModel(nepochs=30, 
                                 batchsize=500, 
                                 maxqsize=10)


