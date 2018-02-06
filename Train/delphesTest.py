

from DeepJetCore.training.training_base import training_base
from keras.layers import Dense, Dropout
from keras.models import Model



def myDomAdaptModel(Inputs,nclasses,nregclasses,dropoutRate=0.05):
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform',name='firstDense')(Inputs[0])
    x = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='predID')(x)
    return Model(inputs=Inputs, outputs=[x])
    
#also does all the parsing
train=training_base(testrun=False)


if  not train.modelSet():

    print 'Setting model'
    train.setModel(myDomAdaptModel,dropoutRate=0.1)
    
    train.compileModel(learningrate=0.003,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])


model,history = train.trainModel(nepochs=10, 
                                 batchsize=5000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)
