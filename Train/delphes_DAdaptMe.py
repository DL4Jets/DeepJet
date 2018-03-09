

from DeepJetCore.training.training_base import training_base


from keras.layers import Dense, Dropout, Concatenate, LocallyConnected1D, Reshape, Flatten
from keras.models import Model

from Layers import GradientReversal
from Losses import binary_crossentropy_labelweights_Delphes, binary_crossentropy_MConly_Delphes
import keras.backend as K
from keras.layers.core import Reshape

import keras.backend as K


def myDomAdaptModel(Inputs,nclasses,nregclasses,dropoutRate=0.05, nodemulti=2):
    
    X = Dense(nodemulti*30, activation='relu') (Inputs[0])#reco inputs
    
    X = Dropout(dropoutRate)(X)
    X = Dense(nodemulti*30, activation='relu',name='classifier_dense0')(X)
    X = Dropout(dropoutRate)(X)
    X = Dense(nodemulti*30, activation='relu',name='classifier_dense1')(X)

    Xa = Dropout(dropoutRate)(X)
    X = Dense(nodemulti*20, activation='relu',name='classifier_dense2')(Xa)
    X = Dropout(dropoutRate)(X)
    X = Dense(nodemulti*10, activation='relu',name='classifier_dense3')(X)
    X = Dropout(dropoutRate)(X)
    #three labels
    labelpred = Dense(3, activation='softmax',name='classifier_pred')(X)
    
    Ad = GradientReversal(name='da_gradrev0')(Xa)
    Ad = Dense(nodemulti*15, activation='relu',name='da_dense0')(Ad)
    Ad = Dropout(dropoutRate)(Ad)
    Ad = Dense(nodemulti*15, activation='relu',name='da_dense1')(Ad)
    Ad = Dropout(dropoutRate)(Ad)
    Ad = Dense(nodemulti*15, activation='relu',name='da_dense2')(Ad)
    Ad = Dense(1,  activation='sigmoid')(Ad)
    
    #make list out of it, three labels from truth - make weights
    Weight = Reshape((3,1),name='reshape')(Inputs[1])
    
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
#from pdb import set_trace
#set_trace()
train=training_base(testrun=False)
print 'Setting model'

learningratemulti=40
dropout=0.4
lrfactor=0.99
nodemulti=2 #this is the default model

if train.keras_model_method:
    parsedlist=[float(i) for i in train.keras_model_method.split(',')]
    
    learningratemulti=parsedlist[0]
    dropout=parsedlist[1]
    lrfactor=parsedlist[2]
    if len(parsedlist)>3:
        nodemulti=int(parsedlist[3])

print('learningratemulti: '+str(learningratemulti))
print('dropout: '+str(dropout))


train.setModel(myDomAdaptModel,dropoutRate=dropout, nodemulti=nodemulti)

train.defineCustomPredictionLabels(['prob_isB','prob_isC','prob_isUDSG',
                                    'prob_isMC',
                                    'labweight_0',
                                    'labweight_1',
                                    'labweight_2'])



train.compileModel(learningrate=learningratemulti*0.0001,
                   loss=[binary_crossentropy_MConly_Delphes,
                         binary_crossentropy_labelweights_Delphes],
                   metrics=['accuracy'],
                   loss_weights=[1.,0.])

print(train.keras_model.summary())


model,history = train.trainModel(nepochs=150, 
                                 batchsize=6000, 
                                 maxqsize=3,
                                 verbose=0,
                                 lr_patience=5,
                                 lr_factor=lrfactor
                                 )



exit()

#import os
#outFolder = "../newDA/trainOutput_daAll"
#os.system("cp %s/full_info.log  %s/full_info_Part1.log" %(outFolder, outFolder));


#exit()

train.compileModel(learningrate=0.0001,
                   loss=[binary_crossentropy_MConly_Delphes,
                         binary_crossentropy_labelweights_Delphes],
                   metrics=['accuracy'],
                   loss_weights=[1./10,1.])


model,history = train.trainModel(nepochs=30, 
                                 batchsize=2000, 
                                 maxqsize=3,
                                 verbose=0)


