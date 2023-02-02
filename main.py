import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from plotting import *
from pathlib import Path
from plasticity_model import PylediPlasticity
import os


def run_freq_single_processing(data, nmdabar_mult=1, ampabar_mult=1):
    frequency, pairings = data
    model = PylediPlasticity(nmdabar_mult=nmdabar_mult, ampabar_mult=ampabar_mult)
    model.run_freq_tests(frequency, pairings, soma_on=False, stdp_dt=10, start=300, afterstop=100)
    return (model, frequency, pairings)


#---------------------------------------------------------
#Bi & Poo
#---------------------------------------------------------



#----------------------------------------------------------
# Fig 2 STDP traces WW LTP and LTD

pairings_STDP_WW_2Columns=4
frequency_STDP_WW_2Columns=5

def Fig2_STDP_2Columns_WW():
    print('STDP_2Columns_WW')
    dtdict = dict([(i, x) for (x, i) in enumerate(range(-100, 110, 10))])
    data = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_WW_2Columns, frequency = frequency_STDP_WW_2Columns, w_mult = 10, nr2bbar_mult = 1, post_spikes = 2)
    DT = 10
    datas = [data[dtdict[DT]], data[dtdict[-DT]]] #AS
    plot_generic_2columns(datas, enable_stdp=True, filename=f"Fig2_STDP_2Columns_WW.png", options={"reduce_X_axis": 1, "dt": DT})

#----------------------------------------------------------
#Fig 3A STDP WW asymetrical

pairings_STDP_WW_60pairs=60
frequency_STDP_WW_60pairs=5 

def Fig3A_STDP_WW_60pairs():
    print('Fig3A STDP_WW_60pairs_5Hz')
    results_multiple = []
    for item in [0]:#np.arange(1,100,10):
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_WW_60pairs, frequency = frequency_STDP_WW_60pairs, w_mult = 10, post_spikes = 2, synparams = synparams)
        results_multiple.append(results)
    plot_stdp_full_WW_60pairs(results_multiple, filename="figures/Fig3A_STDP_WW_60pairs_5Hz.png")
    
#----------------------------------------------------------

#Fig 3B STDP WW LTP
pairings_STDP_WW_5pairs=5
frequency_STDP_WW_5pairs=5

def Fig3B_STDP_WW_5pairs():
    print('Fig3B STDP_WW_5pairs_5Hz')
    results_multiple = []
    for item in [0]:#np.arange(1,100,10):
        synparams = {}
        # synparams = {
        #     'dirac_trace_tau': item
        # }
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_WW_5pairs, frequency = frequency_STDP_WW_5pairs, w_mult = 10, post_spikes = 2, synparams = synparams)
        results_multiple.append(results)
    plot_stdp_full_WW_5pairs(results_multiple, filename="figures/Fig3B_STDP_WW_5pairs_5Hz.png")
    
#----------------------------------------------------------


#Fig 3C STDP WW LTD
pairings_STDP_WW_30pairs=30
frequency_STDP_WW_30pairs=1

def Fig3C_STDP_WW_30pairs():
    print('Fig3C STDP_WW_30pairs_1Hz')
    results_multiple = []
    for item in [0]:#np.arange(1,100,10):
        synparams = {}
        # synparams = {
        #     'dirac_trace_tau': item
        # }
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_WW_30pairs, frequency = frequency_STDP_WW_30pairs, w_mult = 10, post_spikes = 2, synparams = synparams)
        results_multiple.append(results)
    plot_stdp_full_WW_5pairs(results_multiple, filename="figures/Fig3C STDP_WW_30pairs_1Hz.png")

#--------------------------------------------------------


#Fig 3D Bi & Poo
#STDP Bi & Poo: 1 Pre - 1 Post
pairings_STDP_BP=60
frequency_STDP_BP=5
def Fig3D_STDP_BP():
    print('Fig3D STDP_BP')
    results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_BP, frequency = frequency_STDP_BP, w_mult = 10, nr2bbar_mult = 1, save_last_record_only=True)
    #results = PylediPlasticity.run_stdp_tests_static(pairings = 60, frequency = 5, w_mult = 10, nr2bbar_mult = 1)
    #plot_stdp(results)
    plot_stdp_full_BP(results, filename="figures/Fig3D_STDP_BP.png")

#------------------------------------------------------------------

# Number of post spikes pre-(1-4) post
frequency_SpikeCount = 5
pairings_SpikeCount = 30

def Fig4A_STDP_PostSpikes_Barplot():
    print("Fig4A Number of spikes Bar plot for pre-(1-4)post")
    results_multiple = []
    postspikes = [1, 2, 3, 4]
    for post_spike in postspikes:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_SpikeCount, frequency = frequency_SpikeCount, w_mult = 10, nr2bbar_mult=1, post_spikes = post_spike, synparams = synparams, stdp_range=[10], first_post=True, save_last_record_only=True)
        results_multiple.append(results[0]['weight'][-1])
        
   
    plt.figure()
    plt.margins(x=0, y=0) 

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    label_size=20

   

    ax = plt.subplot()
    ax.set_axisbelow(True)
    #ax.grid(color='blue', linestyle='dashed')
    ax.bar([str(f"{x}") for x in postspikes], results_multiple, width=0.4, color='blue')
    #ax.scatter(np.arange(0, 5), [190, 185, 125, 120, 75], marker="*", color=(0.6, 0.1, 0.9), s=1000, label="Experimental data")
    label_size = 20
    tick_size = 16
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([0, 2.5])
    ax.set_ylabel("Relative weight change", fontsize=label_size)
    ax.set_xlabel("# of postsynaptic spikes", fontsize=label_size)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig4A_STDP_SpikePost_Barplot.png"))

#------------------------------------------------------------------
# Freq of pre-post

def Fig4B_STDP_Freq_Barplot():
    print("Fig4B Frequency Bar plot for pre-post ")
    results_multiple = []
    frequencies = [1, 5, 10, 20, 30, 40, 50 ]# [1, 3, 5, 10] #[5, 10, 15 ,20 ] #
    for freq in frequencies:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = 20, frequency = freq, w_mult = 10, nr2bbar_mult=1, post_spikes = 1, synparams = synparams, stdp_range=[10], first_post=True, save_last_record_only=True)
        results_multiple.append(results[0]['weight'][-1])
        
    #plt.figure(figsize=(14,10))
    plt.figure()
    plt.margins(x=0, y=0) 

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    label_size=20

    ax = plt.subplot() 
    ax.set_axisbelow(True)
    
    ax.plot(frequencies, results_multiple, linewidth=4, color='blue')
    ax.scatter(frequencies, results_multiple, marker='*', s=100, color='blue')
    
  
    ax.set_xlim([0,  50])
    ax.set_ylim([0.8, 2.5])
    
    label_size = 20
    tick_size = 16
    ax.set_ylabel("Relative weight change", fontsize=label_size)
    ax.set_xlabel("Frequency of pre-post pairings, Hz", fontsize=label_size)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig4B_STDP_dt10_Freq_Line.png"))



#----------------------------------------------------------
#Fig 5 Three columns, LTP, LTP and LTD
def Fig5_Freq_Dependent_3Columns():
    print('Fig5 Freq_Dependent_3Columns()')
    (model1, _, _) = run_freq_single_processing((100, 100), nmdabar_mult=1, ampabar_mult=0.5)
    (model2, _, _) = run_freq_single_processing((100, 100), nmdabar_mult=1)
    (model3, _, _) = run_freq_single_processing((1, 3), nmdabar_mult=1)
    plot_generic_3columns([model1.run_data, model2.run_data, model3.run_data], enable_stdp=False, filename=f"Fig5_Freq_Dependent_3Columns.png")




#----------------------------------------------------------

if __name__ == '__main__':
    
    FIGURES_DIR = "figures"
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

#------------------------------
      
    
    #-------------------------------
    #Fig 2 Traces WW   
    #Fig2_STDP_2Columns_WW()

    #------------------------------
    #Fig 3 STDP WW and BP 
    #Fig3A_STDP_WW_60pairs()
    #Fig3B_STDP_WW_5pairs()
    #Fig3C_STDP_WW_30pairs()
    #Fig3D_STDP_BP()
    
    #------------------------------
    #Fig 4 f and # of spikes
    #Fig4A_STDP_PostSpikes_Barplot() 
    #Fig4B_STDP_Freq_Barplot() #Bars

    #---------------------------------------------
    #Frequency-dependent LTP and LTD
    Fig5_Freq_Dependent_3Columns() 

    
           
    print("\nDONE")    
    