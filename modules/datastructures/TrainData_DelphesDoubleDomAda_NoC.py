'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainDataDeepJetDelphes import TrainDataDeepJetDelphes, fileTimeOut

from TrainData_DelphesDomAda_NoC import TrainData_DelphesDomAda_NoC

# just change the inhertance to include Cs!
class TrainData_DelphesDoubleDomAda_NoC(TrainData_DelphesDomAda_NoC):
    '''
    example data structure - basis for further developments
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_DelphesDomAda_NoC.__init__(self)                                                                         
        

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        import numpy as np
        
        super(TrainData_DelphesDoubleDomAda_NoC,self).readFromRootFile(filename,TupleMeanStd, weighter)
        
        # just for reference: isB isC isUDSG - isC always 0
        labeltruth=self.y[0]
        
        #randomly define process label
        proclabel  =  np.random.randint(0,2,labeltruth.shape[0])
        
        #pick the 
        
        # add multiplier for different labelfraction
        ones=np.zeros(labeltruth.shape[0]) + 1
        #set multiplier for bs 
        multiplierb = np.array(ones)
        multiplierb[ proclabel >0 ] = 0.6
        
        #set multiplier for light 
        multiplierl = np.array(ones)
        multiplierl[ proclabel >0 ] =1 - 0.6
        
        #keep ones for the (removed) Cs, just in case for sipler scalability
        multi = np.vstack((multiplierb,ones,multiplierl))
        multi=multi.transpose()
        
        #explicitly also add the process label
        proclabel=proclabel.reshape((labeltruth.shape[0],1))
        
        #add all back to store the info
        self.y[1] = np.hstack((self.y[1],proclabel,multi))
        
        #append the same again for the third loss
        self.y.append(self.y[1])
        
        #this makes the structure as follows:
        # y[0] = (isB, isC, isUDSG)
        # y[1] = (isMC, isB, isC, isUDSG, isProcessA, multi_isB, multi_isC, multi_isUDSG)
        # y[2] = y[1]
        #
        # here, ProcessA describes the process with the adapted label fractions
        
        

