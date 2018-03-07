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

        self.addBranches(['jet_pt', 'jet_eta']) #consider jet pt and eta                                                          

        self.addBranches(['track_pt'], 5) #consider the pt of the first 6 tracks                                                  

        self.addBranches(['track_ptRel', 'track_sip3D', 'track_sip2D', 'track_pPar'], 8) #all those for the first 10 tracks       

        self.registerBranches(['isMC','isTtbar'])
        #creates label weights per batch                                                                                          
        #due to normalisation, two are sufficient for 3 labels (B, C UDSG)                                                        
        #self.generatePerBatch=None #[[0.2,5.],[0.2,5.]]                                                                          
        

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        import numpy
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        mclabel=Tuple['isMC'].view(numpy.ndarray)
        mclabel=mclabel.reshape(mclabel.shape[0],1)
        proclabel=Tuple['isTtbar'].view(numpy.ndarray)
        proclabel=proclabel.reshape(mclabel.shape[0],1)

        
        weights,x_all,alltruth, notremoves =self.getFlavourClassificationData(filename,TupleMeanStd, weighter)

        #slicingfrom isB, isC, isUDSG and keep only isB, isUCSG
        alltruthNoC = alltruth[:,::2]
        alltruth = alltruthNoC
       
        print(' xalltruth= ', alltruth)        


        print('x_all=', x_all)
        print('x_all.shape=', x_all.shape)        
        
        #notremoves -= Tuple['isC']
        if self.remove:
            #print('remove')
            mclabel=mclabel[notremoves > 0]
            proclabel=proclabel[notremoves > 0]

        print('post remove')
        print('x_all=', x_all)
        print('x_all.shape=', x_all.shape)        
            
        
        
        domaintruth_datamc=numpy.hstack((mclabel,alltruth))
        labeltruth=domaintruth_datamc
        #domaintruth_ttbarqcd=numpy.hstack((proclabel,alltruth))
        
        print('alltruth.shape = ', alltruth.shape)
        
        self.w=[weights]
        #the label fraction weights are computed on the fly
        self.x=[x_all, alltruth]
        print(' x_all= ', x_all)

        
        
        #the truth
        self.y=[labeltruth,domaintruth_datamc]
        print(' y= ', labeltruth)
        print(' y= ', domaintruth_datamc)
        print('domaintruth_datamc.shape', domaintruth_datamc.shape)


