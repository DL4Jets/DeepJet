from DeepJetCore.evaluation import makeROCs_async
from argparse import ArgumentParser

parser = ArgumentParser('program to convert root tuples to traindata format')
parser.add_argument("-i", help="set input sample description (output from the check.py script)", metavar="FILE")


# process options                                                                                                                                                                                               
args=parser.parse_args()
intextfile=args.i


makeROCs_async(intextfile, 
               name_list=['B vs light', 'B vs. C'],         
               probabilities_list=['prob_isB','prob_isB'], 
               truths_list=['isB','isB'],        
               vetos_list=['isUDSG','isC'],         
               colors_list='auto',        
               outpdffile='ROC.pdf',         
               cuts='jet_pt>30',            
               cmsstyle=False,     
               firstcomment='',    
               secondcomment='',   
               invalidlist='',     
               extralegend=None,   
               logY=True,          
               individual=False,   
               xaxis="b efficiency",           
               nbins=200, 
               treename="tree")          

