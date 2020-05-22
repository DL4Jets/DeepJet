

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
from argparse import ArgumentParser


class TrainData_DF(TrainData):
    def __init__(self):

        TrainData.__init__(self)

        
        self.description = "DeepJet training datastructure"
        
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass='isB'
        self.weight_binX = np.array([
            10,25,30,35,40,45,50,60,75,100,
            125,150,175,200,250,300,400,500,
            600,2000],dtype=float)
        
        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
        )

        self.global_branches = ['jet_pt', 'jet_eta',
                                'nCpfcand','nNpfcand',
                                'nsv','npv',
                                'TagVarCSV_trackSumJetEtRatio',
                                'TagVarCSV_trackSumJetDeltaR',
                                'TagVarCSV_vertexCategory',
                                'TagVarCSV_trackSip2dValAboveCharm',
                                'TagVarCSV_trackSip2dSigAboveCharm',
                                'TagVarCSV_trackSip3dValAboveCharm',
                                'TagVarCSV_trackSip3dSigAboveCharm',
                                'TagVarCSV_jetNSelectedTracks',
                                'TagVarCSV_jetNTracksEtaRel']
                
        
        self.cpf_branches = ['Cpfcan_BtagPf_trackEtaRel',
                             'Cpfcan_BtagPf_trackPtRel',
                             'Cpfcan_BtagPf_trackPPar',
                             'Cpfcan_BtagPf_trackDeltaR',
                             'Cpfcan_BtagPf_trackPParRatio',
                             'Cpfcan_BtagPf_trackSip2dVal',
                             'Cpfcan_BtagPf_trackSip2dSig',
                             'Cpfcan_BtagPf_trackSip3dVal',
                             'Cpfcan_BtagPf_trackSip3dSig',
                             'Cpfcan_BtagPf_trackJetDistVal',
                             'Cpfcan_ptrel',
                             'Cpfcan_drminsv',
                             'Cpfcan_VTX_ass',
                             'Cpfcan_puppiw',
                             'Cpfcan_chi2',
                             'Cpfcan_quality']
        self.n_cpf = 25

        self.npf_branches = ['Npfcan_ptrel','Npfcan_deltaR','Npfcan_isGamma','Npfcan_HadFrac','Npfcan_drminsv','Npfcan_puppiw']
        self.n_npf = 25
        
        self.vtx_branches = ['sv_pt','sv_deltaR',
                             'sv_mass',
                             'sv_ntracks',
                             'sv_chi2',
                             'sv_normchi2',
                             'sv_dxy',
                             'sv_dxysig',
                             'sv_d3d',
                             'sv_d3dsig',
                             'sv_costhetasvpv',
                             'sv_enratio',
        ]

        self.n_vtx = 4
        
        self.reduced_truth = ['isB','isBB','isLeptonicB','isC','isUDS','isG']

        
    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches
            )

        
        counter=0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "deepntuplizer/tree",
                    stop = None,
                    branches = branches
                )
                weighter.addDistributions(nparray)
                #del nparray
                counter=counter+1
                weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
        return {'weigther':weighter}
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):

        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw=stopwatch()
        swall=stopwatch()
        if not istraining:
            self.remove = False
        
        def reduceTruth(uproot_arrays):
            
            b = uproot_arrays[b'isB']
            
            bb = uproot_arrays[b'isBB']
            gbb = uproot_arrays[b'isGBB']
            
            bl = uproot_arrays[b'isLeptonicB']
            blc = uproot_arrays[b'isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays[b'isC']
            cc = uproot_arrays[b'isCC']
            gcc = uproot_arrays[b'isGCC']
            
            ud = uproot_arrays[b'isUD']
            s = uproot_arrays[b'isS']
            uds = ud+s
            
            g = uproot_arrays[b'isG']
            
            return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.global_branches],
                                   [1],self.nsamples)

        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.cpf_branches,
                                   self.n_cpf,self.nsamples)

        x_npf = MeanNormZeroPadParticles(filename,None,
                                         self.npf_branches,
                                         self.n_npf,self.nsamples)

        x_vtx = MeanNormZeroPadParticles(filename,None,
                                         self.vtx_branches,
                                         self.n_vtx,self.nsamples)

        
        
        import uproot
        urfile = uproot.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')


        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "deepntuplizer/tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')


        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')

        
        print('remove nans')
        x_global = np.where(np.isfinite(x_global), x_global, 0)
        x_cpf = np.where(np.isfinite(x_cpf), x_cpf, 0)
        x_npf = np.where(np.isfinite(x_npf), x_npf, 0)
        x_vtx = np.where(np.isfinite(x_vtx), x_vtx, 0)

        return [x_global,x_cpf,x_npf,x_vtx], [truth], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose() ) ),
                                         names='prob_isB, prob_isBB,prob_isLeptB, prob_isC,prob_isUDS,prob_isG,isB, isBB, isLeptB, isC,isUDS,isG,jet_pt, jet_eta')
        array2root(out, outfilename, 'tree')



class TrainData_DeepCSV(TrainData):
    def __init__(self):

        TrainData.__init__(self)

        self.description = "DeepCSV training datastructure"
       
        self.truth_branches = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isGCC','isCC','isUD','isS','isG']
        self.undefTruth=['isUndefined']
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        self.remove = True
        self.referenceclass='isB'
        self.weight_binX = np.array([
            10,25,30,35,40,45,50,60,75,100,
            125,150,175,200,250,300,400,500,
            600,2000],dtype=float)
        
        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
        )

        self.global_branches = ['jet_pt', 'jet_eta',
                                'TagVarCSV_trackSumJetEtRatio',
                                'TagVarCSV_trackSumJetDeltaR',
                                'TagVarCSV_vertexCategory',
                                'TagVarCSV_trackSip2dValAboveCharm',
                                'TagVarCSV_trackSip2dSigAboveCharm',
                                'TagVarCSV_trackSip3dValAboveCharm',
                                'TagVarCSV_trackSip3dSigAboveCharm',
                                'TagVarCSV_jetNSelectedTracks',
                                'TagVarCSV_jetNTracksEtaRel']

        self.track_branches = ['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal']
        self.n_track = 6
        
        self.eta_rel_branches = ['TagVarCSV_trackEtaRel']
        self.n_eta_rel = 4

        self.vtx_branches = ['TagVarCSV_vertexMass', 
                          'TagVarCSV_vertexNTracks', 
                          'TagVarCSV_vertexEnergyRatio',
                          'TagVarCSV_vertexJetDeltaR',
                          'TagVarCSV_flightDistance2dVal', 
                          'TagVarCSV_flightDistance2dSig', 
                          'TagVarCSV_flightDistance3dVal', 
                          'TagVarCSV_flightDistance3dSig']
        self.n_vtx = 1
                
        self.reduced_truth = ['isB','isBB','isC','isUDSG']

    def readTreeFromRootToTuple(self, filenames, limit=None, branches=None):
        '''
        To be used to get the initial tupel for further processing in inherting classes
        Makes sure the number of entries is properly set
        
        can also read a list of files (e.g. to produce weights/removes from larger statistics
        (not fully tested, yet)
        '''
        
        if branches is None or len(branches) == 0:
            return np.array([],dtype='float32')
            
        #print(branches)
        #remove duplicates
        usebranches=list(set(branches))
        tmpbb=[]
        for b in usebranches:
            if len(b):
                tmpbb.append(b)
        usebranches=tmpbb
            
        import ROOT
        from root_numpy import tree2array, root2array
        if isinstance(filenames, list):
            for f in filenames:
                fileTimeOut(f,120)
            print('add files')
            nparray = root2array(
                filenames, 
                treename = "deepntuplizer/tree", 
                stop = limit,
                branches = usebranches
                )
            print('done add files')
            return nparray
            print('add files')
        else:    
            fileTimeOut(filenames,120) #give eos a minute to recover
            rfile = ROOT.TFile(filenames)
            tree = rfile.Get(self.treename)
            if not self.nsamples:
                self.nsamples=tree.GetEntries()
            nparray = tree2array(tree, stop=limit, branches=usebranches)
            return nparray
        
    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches
            )

        
        counter=0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "deepntuplizer/tree",
                    stop = None,
                    branches = branches
                )
                weighter.addDistributions(nparray)
                #del nparray
                counter=counter+1
                weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)

        print("calculate means")
        from DeepJetCore.preprocessing import meanNormProd
        nparray = self.readTreeFromRootToTuple(allsourcefiles,branches=self.vtx_branches+self.eta_rel_branches+self.track_branches+self.global_branches, limit=500000)
        for a in (self.vtx_branches+self.eta_rel_branches+self.track_branches+self.global_branches):
            for b in range(len(nparray[a])):
                nparray[a][b] = np.where(nparray[a][b] < 100000.0, nparray[a][b], 0)
        means = np.array([],dtype='float32')
        if len(nparray):
            means = meanNormProd(nparray)
        return {'weigther':weighter,'means':means}
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):

        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw=stopwatch()
        swall=stopwatch()
        print(weighterobjects)
        if not istraining:
            self.remove = False
                
        def reduceTruth(uproot_arrays):
            
            b = uproot_arrays[b'isB']
            
            bb = uproot_arrays[b'isBB']
            gbb = uproot_arrays[b'isGBB']
            
            bl = uproot_arrays[b'isLeptonicB']
            blc = uproot_arrays[b'isLeptonicB_C']
            lepb = bl+blc
            
            c = uproot_arrays[b'isC']
            cc = uproot_arrays[b'isCC']
            gcc = uproot_arrays[b'isGCC']
            
            ud = uproot_arrays[b'isUD']
            s = uproot_arrays[b'isS']
            uds = ud+s
            
            g = uproot_arrays[b'isG']
            
            return np.vstack((b+lepb,bb+gbb,c+cc+gcc,uds+g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()  
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        x_global = MeanNormZeroPad(filename,weighterobjects['means'],
                                   [self.global_branches,self.track_branches,self.eta_rel_branches,self.vtx_branches],
                                   [1,self.n_track,self.n_eta_rel,self.n_vtx],self.nsamples)
                
        import uproot
        urfile = uproot.open(filename)["deepntuplizer/tree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "deepntuplizer/tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')


        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')

        
        print('remove nans')
        x_global = np.where(np.isfinite(x_global), x_global, 0)
        
        return [x_global], [truth], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose() ) ),
                                         names='prob_isB, prob_isBB, prob_isC,prob_isUDSG,isB, isBB, isC,isUDSG,jet_pt, jet_eta')
        array2root(out, outfilename, 'tree')
