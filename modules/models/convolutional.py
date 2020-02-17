from keras.layers import Dense, Dropout, Flatten,Concatenate, Lambda, Convolution2D, LSTM, Convolution1D, Conv2D,GlobalAveragePooling1D, GlobalMaxPooling1D,TimeDistributed
from keras.models import Model
import tensorflow as tf
from keras.layers import BatchNormalization
#from keras.layers.normalization import BatchNormalization

from keras import backend as K
from Layers import *
from buildingBlocks import block_deepFlavourConvolutions, block_deepFlavourDense


def model_deepFlavourReference(Inputs,dropoutRate=0.1,momentum=0.6):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(6, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


