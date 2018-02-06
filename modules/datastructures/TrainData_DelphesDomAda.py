'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainDataDeepJetDelphes import TrainDataDeepJetDelphes, fileTimeOut



class TrainData_DelphesDomAda(TrainDataDeepJetDelphes):
    '''
    example data structure - basis for further developments
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainDataDeepJetDelphes.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta']) #consider jet pt and eta
       
        self.addBranches(['track_pt'], 6) #consider the pt of the first 6 tracks
        
        self.addBranches(['track_releta', 'track_sip3D', 'track_sip2D'], 10) #all those for the first 10 tracks
        

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        weights,x_all,alltruth, _ =self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]


