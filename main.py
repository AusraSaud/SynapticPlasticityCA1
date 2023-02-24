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


#----------------------------------------------------------
# Fig 2: STDP traces LTP and LTD

pairings_STDP_2Columns=4
frequency_STDP_2Columns=5

def Fig2():
    print('Running Fig2 ...')
    dtdict = dict([(i, x) for (x, i) in enumerate(range(-100, 110, 10))])
    data = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_2Columns, frequency = frequency_STDP_2Columns, w_mult = 10, nr2bbar_mult = 1, post_spikes = 2)
    DT = 10
    datas = [data[dtdict[DT]], data[dtdict[-DT]]]
    plot_2columns(datas, filename=f"Fig2.png")

#----------------------------------------------------------
#Fig 3A: STDP LTD LTP LTD

pairings_STDP_60pairs=60
frequency_STDP_60pairs=5 

def Fig3A():
    print('Running Fig3A ...')
    results_multiple = []
    for item in [0]:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_60pairs, frequency = frequency_STDP_60pairs, w_mult = 10, post_spikes = 2, synparams = synparams)
        results_multiple.append(results)
    plot_stdp_full_60pairs(results_multiple, filename="figures/Fig3A.png")
    
#----------------------------------------------------------

#Fig 3B: STDP LTP
pairings_STDP_5pairs=5
frequency_STDP_5pairs=5

def Fig3B():
    print('Running Fig3B ...')
    results_multiple = []
    for item in [0]:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_5pairs, frequency = frequency_STDP_5pairs, w_mult = 10, post_spikes = 2, synparams = synparams)
        results_multiple.append(results)
    plot_stdp_full_5pairs(results_multiple, filename="figures/Fig3B.png")
    
#----------------------------------------------------------

#Fig 3C: STDP LTD
pairings_STDP_30pairs=30
frequency_STDP_30pairs=1

def Fig3C():
    print('Running Fig3C ...')
    results_multiple = []
    for item in [0]:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP_30pairs, frequency = frequency_STDP_30pairs, w_mult = 10, post_spikes = 2, synparams = synparams)
        results_multiple.append(results)
    plot_stdp_full_5pairs(results_multiple, filename="figures/Fig3C.png")

#--------------------------------------------------------

#Fig 3D: STDP 1 Pre - 1 Post
pairings_STDP=60
frequency_STDP=5
def Fig3D():
    print('Running Fig3D ...')
    results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_STDP, frequency = frequency_STDP, w_mult = 10, nr2bbar_mult = 1, save_last_record_only=True)
    plot_stdp_full_BP(results, filename="figures/Fig3D.png")

#------------------------------------------------------------------

#Fig 4A: Number of post spikes 1-4
frequency_SpikeCount = 5
pairings_SpikeCount = 30

def Fig4A():
    print("Running Fig4A ...")
    results_multiple = []
    postspikes = [1, 2, 3, 4]
    for post_spike in postspikes:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = pairings_SpikeCount, frequency = frequency_SpikeCount, w_mult = 10, nr2bbar_mult=1, post_spikes = post_spike, synparams = synparams, stdp_range=[10], first_post=True, save_last_record_only=True)
        results_multiple.append(results[0]['weight'][-1])
        

    plt.figure()
    ax = plt.subplot()
    plt.margins(x=0, y=0) 

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    label_size=20

    ax.set_axisbelow(True)
    ax.bar([str(f"{x}") for x in postspikes], results_multiple, width=0.4, color='blue')
    label_size = 20
    tick_size = 16
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([0, 2.5])
    ax.set_ylabel("Relative weight change", fontsize=label_size)
    ax.set_xlabel("# of postsynaptic spikes", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig4A.png"))

#------------------------------------------------------------------
# Fig 4B: Frequency of pre-post

def Fig4B():
    print("Running Fig4B ...")
    results_multiple = []
    frequencies = [1, 5, 10, 20, 30, 40, 50 ]
    for freq in frequencies:
        synparams = {}
        results = PylediPlasticity.run_stdp_tests_static(pairings = 20, frequency = freq, w_mult = 10, nr2bbar_mult=1, post_spikes = 1, synparams = synparams, stdp_range=[10], first_post=True, save_last_record_only=True)
        results_multiple.append(results[0]['weight'][-1])
        
    plt.figure()
    ax = plt.subplot() 
    plt.margins(x=0, y=0) 

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    label_size=20

    
    ax.set_axisbelow(True)
    
    ax.plot(frequencies, results_multiple, linewidth=4, color='blue')
    ax.scatter(frequencies, results_multiple, marker='*', s=100, color='blue')
    
  
    ax.set_xlim([0,  50])
    ax.set_ylim([0.8, 2.5])
    
    label_size = 20
    tick_size = 16
    ax.set_ylabel("Relative weight change", fontsize=label_size)
    ax.set_xlabel("Frequency of pre-post pairings, Hz", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig4B.png"))



#----------------------------------------------------------
#Fig 5: Frequency dependent LTP, LTP and LTD

def Fig5():
    print('Running Fig5 ...')
    (model1, _, _) = run_freq_single_processing((100, 100), nmdabar_mult=1, ampabar_mult=0.5)
    (model2, _, _) = run_freq_single_processing((100, 100), nmdabar_mult=1)
    (model3, _, _) = run_freq_single_processing((1, 3), nmdabar_mult=1)
    plot_3columns([model1.run_data, model2.run_data, model3.run_data], filename=f"Fig5_Freq_Dependent_3Columns.png")

#----------------------------------------------------------

if __name__ == '__main__':
    
    FIGURES_DIR = "figures"
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    #Plot figures 
    Fig2()
    Fig3A()
    Fig3B()
    Fig3C()
    Fig3D()
    Fig4A() 
    Fig4B() 
    Fig5() 

    print("Done")