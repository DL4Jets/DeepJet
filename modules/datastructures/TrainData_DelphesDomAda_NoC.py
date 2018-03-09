'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainDataDeepJetDelphes import TrainDataDeepJetDelphes, fileTimeOut



class TrainData_DelphesDomAda_NoC(TrainDataDeepJetDelphes):
    '''
    example data structure - basis for further developments
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainDataDeepJetDelphes.__init__(self)

        # self.addBranches(['jet_pt', 'jet_eta']) #consider jet pt and eta                                                          

        # self.addBranches(['track_pt'], 5) #consider the pt of the first 6 tracks                                                  

        self.addBranches(['track_ptRel', 'track_sip3D', 'track_sip2D', 'track_pPar'], 5) #all those for the first 10 tracks       

        self.registerBranches(['isMC','isTtbar'])                                                                           
        

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        import numpy
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        mclabel=Tuple['isMC'].view(numpy.ndarray)
        mclabel=mclabel.reshape(mclabel.shape[0],1)
        proclabel=Tuple['isTtbar'].view(numpy.ndarray)
        proclabel=proclabel.reshape(mclabel.shape[0],1)

        
        weights,x_all,alltruth, notremoves =self.getFlavourClassificationData(filename,
                                                                              TupleMeanStd, 
                                                                              weighter,
                                                                              useremovehere=False)

        before=len(x_all)
        
        notremoves -= Tuple['isC'].view(numpy.ndarray)
        if self.remove:
            print('remove')
            mclabel  =  mclabel[notremoves > 0]
            proclabel=  proclabel[notremoves > 0]
            weights  =  weights[notremoves > 0]
            x_all    =  x_all[notremoves > 0]
            alltruth =  alltruth[notremoves > 0]
      
        print('reduced to ', len(x_all), '/', before)
        
        
        domaintruth_datamc=numpy.hstack((mclabel,alltruth))
        labeltruth=domaintruth_datamc
        
        self.w=[weights]
        #the label fraction weights are computed on the fly
        self.x=[x_all, alltruth]
        #the truth
        self.y=[labeltruth,domaintruth_datamc]


