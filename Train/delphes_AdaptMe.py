

from DeepJetCore.training.training_base import training_base
from keras.layers import Dense, Dropout, Concatenate, LocallyConnected1D, Reshape, Flatten
from keras.models import Model

from Layers import GradientReversal
import keras.backend as K
from keras.layers.core import Reshape

### just for fiddleing around here
def binary_crossentropy_labelweights_A(y_pred, y_true):
    """
    Depricated: Tested and working, yet you can use weighted_loss to (see above) to decorate K.binary_crossentropy instead
    
    The input needs two different samples that are one hot encoded, e.g. real data and simulation. Data and simulation can differ in the label proportions of some other quantity, e.g. bs vs gluon jets. The loss allows to change the label proportion in the "MC" dataset (or bag) via an input feature.
    """
    
    
    # the prediction if it is data or MC is in the first index (see model)
    isMCpred = y_pred[:,:1]
    
    #the weights are in the remaining parts of the vector
    Weightpred = y_pred[:,1:]
    # the truth if it is data or MC
    isMCtrue = y_true[:,:1]
    # labels: B, C, UDSG - not needed here, but maybe later
    labels_true = y_true[:,1:]

    #only apply label weight deltas to MC, for data will be 1 (+1)
    #as a result of locally connected if will be only !=0 for one label
    weightsum = K.clip(isMCtrue * K.sum(Weightpred, axis=-1) + 1, 0.2, 5)
    
    weighted_xentr = weightsum*K.binary_crossentropy(isMCpred, isMCtrue)
    
    #sum weight again over all samples
    return K.sum( weighted_xentr , axis=-1)/K.sum(weightsum, axis=-1)



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
    labelpred = Dense(nclasses, activation='sigmoid')(X)
    
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
                   loss=['binary_crossentropy',
                         binary_crossentropy_labelweights_A],
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
