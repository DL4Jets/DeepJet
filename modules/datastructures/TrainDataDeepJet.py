

from TrainData import TrainData
from TrainData import fileTimeOut as tdfto
import numpy

def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)

class TrainDataDeepJet(TrainData):
    '''
    Base class for DeepJet.
    TO create own b-tagging trainings, please inherit from this class
    '''
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.undefTruth=['isUndefined']
        self.referenceclass='isB'
        self.truthclasses=['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isCC',
                           'isGCC','isUD','isS','isG','isUndefined']
        
        
        #standard branches
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['jet_pt','jet_eta'])
        
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        
        self.weight_binX = numpy.array([
                10,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )
        
        
             
        self.reduceTruth(None)
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd, weighter):
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        #print('took ', sw.getAndReset(), ' seconds for getting tree entries')
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        #print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        #print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth, notremoves
       
    
        
from preprocessing import MeanNormApply, MeanNormZeroPad

class TrainData_Flavour(TrainDataDeepJet):
    '''
    
    '''
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        weights,x_all,alltruth, _ =self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]
        
     
     
class TrainData_simpleTruth(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isC','isUDSG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            allb = b+bl+blc

            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)            
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            g = tuple_in['isG'].view(numpy.ndarray)
            l = g + uds

            return numpy.vstack((allb,bb+gbb,c+cc+gcc,l)).transpose()
    
    
    
#    
#
#  DeepJet default classes
#
#

    
class TrainData_leptTruth(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDSG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)
            
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            lepb=bl+blc
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            l = g + uds
            
            return numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,l)).transpose()  
        
        
        

class TrainData_fullTruth(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            
            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)
            
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            lepb=bl+blc
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            
            
            return numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()    
  

class TrainData_QGOnly(TrainDataDeepJet):
    def __init__(self):
        TrainDataDeepJet.__init__(self)
        self.clear()
        self.undefTruth=['isUndefined']
        
        self.referenceclass='isUD'
        
        
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isUDS','isG']
        if tuple_in is not None:
            #b = tuple_in['isB'].view(numpy.ndarray)
            #bb = tuple_in['isBB'].view(numpy.ndarray)
            #gbb = tuple_in['isGBB'].view(numpy.ndarray)
            #
            #
            #bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            #blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            #lepb=bl+blc
            #
            #c = tuple_in['isC'].view(numpy.ndarray)
            #cc = tuple_in['isCC'].view(numpy.ndarray)
            #gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            
            
            return numpy.vstack((uds,g)).transpose()    

class TrainData_quarkGluon(TrainDataDeepJet):
    def __init__(self):
        super(TrainData_quarkGluon, self).__init__()
        self.referenceclass = 'isG'
        self.reducedtruthclasses=['isQ', 'isG']
        self.clear()
        
    def reduceTruth(self, tuple_in):
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            #bb = tuple_in['isBB'].view(numpy.ndarray) #this should be gluon?
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            c = tuple_in['isC'].view(numpy.ndarray)
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            q = ud+s#+c+blc+bl+b
            
            g = tuple_in['isG'].view(numpy.ndarray)
            return numpy.vstack((q, g)).transpose()    
        else:
            print('I got an empty tuple?')
        
        