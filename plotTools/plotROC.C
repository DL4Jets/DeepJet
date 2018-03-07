#include <TH1F.h>
#include <TH2F.h>
#include <TF1.h>
#include <TFile.h>
#include <TCanvas.h>
#include "TRandom3.h"
#include "TLegend.h"
#include "TLatex.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <iostream>
#include <TF1.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include "TTree.h"
#include "TChain.h"
#include <vector>
#include <fstream>
#include <string>
#include "TROOT.h"
#include "TSystem.h"




//root plotTools/plotROC.C'("newDA1p5/ROC_dnnMC/", "newDA1p5/ROC_dnnDA/", "newDA1p5/ROC_daMC", "newDA1p5/ROC_daDA")'
void plotROC(std::string f1/*DNN mc*/, std::string f2/* DNN data*/, std::string f3/* DomAda mc*/, std::string f4/*DomAda data*/){

  gROOT->Macro("/afs/cern.ch/user/a/amartell/public/setStyle.C");

  std::vector<std::string> names;
  names.push_back("roccurve_0");
  names.push_back("roccurve_1");

  std::vector<std::string> nameH;
  nameH.push_back("B vs light");  
  nameH.push_back("B vs C");


  std::cout << " >>> f1 = " << f1 << " f2 = " << f2 << " f3 = " << f3 << " f4 = " << f4 << std::endl;


  std::vector<std::string> nameV;
  nameV.push_back("DNN mc");    
  nameV.push_back("DNN data");    
  nameV.push_back("DomAda mc");    
  nameV.push_back("DomAda data");    

  std::vector<std::string> fileInputNames;
  fileInputNames.push_back(f1);
  fileInputNames.push_back(f2);
  fileInputNames.push_back(f3);
  fileInputNames.push_back(f4);


  int nOptions = nameV.size();

  //  int iColors[8] = {kRed, kBlue, kRed+2, kBlue+2, kMagenta, kCyan, kOrange+7, kAzure+7};
  int iColors[7] = {kRed, kBlue+1, kRed+2, kMagenta, kCyan+1, kGreen+2, kYellow+2}; //, kCyan-1, kMagenta-2, kBlue-6, kRed-9, kGreen-9};
  
  TGraph* tgB[2][7];

  TH1D* dummyS[2];
  int type = 0;
  TFile *inF[4];
  for(int ij=0; ij<nOptions; ++ij){
    inF[ij] = TFile::Open((fileInputNames.at(ij)+"/ROC.root").c_str());
    for(int io=0; io<2; ++io){
      tgB[io][ij] = (TGraph*)inF[ij]->Get(names.at(io).c_str());
      tgB[io][ij]->SetName(Form((names.at(io)+"_v%d").c_str(), io));
      tgB[io][ij]->SetLineColor(iColors[ij]);
      tgB[io][ij]->SetLineWidth(3);
      tgB[io][ij]->SetLineStyle(io+1);
      tgB[io][ij]->SetMarkerColor(iColors[ij]);
    }
  }



  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);


  TLegend *legTGM = new TLegend(0.55,0.2,0.75,0.45,NULL,"brNDC");
  legTGM->SetTextFont(42);
  legTGM->SetTextSize(0.04);
  legTGM->SetFillStyle(0);
  legTGM->SetFillColor(kWhite);
  legTGM->SetLineColor(kWhite);
  legTGM->SetShadowColor(kWhite);
  for(int io=0; io<nOptions; ++io)
    legTGM->AddEntry(tgB[0][io], Form( (nameV.at(io)+"  " + nameH.at(0)).c_str(), io), "l");

  TLegend *legTGMb = new TLegend(0.15,0.7,0.35,0.95,NULL,"brNDC");
  legTGMb->SetTextFont(42);
  legTGMb->SetTextSize(0.04);
  legTGMb->SetFillStyle(0);
  legTGMb->SetFillColor(kWhite);
  legTGMb->SetLineColor(kWhite);
  legTGMb->SetShadowColor(kWhite);
  for(int io=0; io<nOptions; ++io)
    legTGMb->AddEntry(tgB[1][io], Form( (nameV.at(io) + "  " + nameH.at(1)).c_str(), io), "l");


  std::string outFolder = "plotsROC";


  TCanvas* cx = new TCanvas();
  gPad->SetLogy();  
  //  gPad->SetLogx();
  
  cx->cd();
  tgB[0][0]->GetXaxis()->SetRangeUser(0., 1.);
  tgB[0][0]->GetYaxis()->SetRangeUser(0.001, 1.1);

  tgB[0][0]->GetXaxis()->SetTitle("b efficiency");
  tgB[0][0]->GetYaxis()->SetTitle("misid probability");
  tgB[0][0]->Draw("apl");
  //  tgB[1][0]->Draw("pl, same");
  for(int ij=0; ij<1; ++ij){
    for(int io=1; io<nOptions; ++io){
      tgB[ij][io]->Draw("pl, same");
    }
  }
  legTGM->Draw("h, same");
  //legTGMb->Draw("h, same");
  cx->Print((outFolder+"/ROCcompare.png").c_str(), "png");
  cx->Print((outFolder+"/ROCcompare.root").c_str(), "root");
  cx->Print((outFolder+"/ROCcompare.pdf").c_str(), "pdf");



}

