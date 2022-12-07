import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from plotting import *
from pathlib import Path
from plasticity_model import PylediPlasticity
import os


def run_freq_single_processing(data, nmdabar_mult=1):
    frequency, pairings = data
    model = PylediPlasticity(nmdabar_mult=nmdabar_mult, ampabar_mult=1)
    model.run_freq_tests(frequency, pairings, soma_on=False, stdp_dt=10, start=300, afterstop=100)
    return (model, frequency, pairings)

def Fig1_STDP():
    results = PylediPlasticity.run_stdp_tests_static(pairings = 60, frequency = 5, w_mult=10)
    plot_stdp(results)

def Fig2_STDP_2Columns():
    dtdict = dict([(i, x) for (x, i) in enumerate(range(-100, 110, 10))])
    data = PylediPlasticity.run_stdp_tests_static(pairings = 2, frequency = 5, w_mult=10)
    datas = [data[dtdict[10]], data[dtdict[-10]]]

    plot_generic_2columns(datas, enable_stdp=True, filename=f"Fig2_STDP_2Columns.png")

def Fig3_Freq_Dependent_2Columns():
    (model1, _, _) = run_freq_single_processing((100, 100), nmdabar_mult=1)
    (model2, _, _) = run_freq_single_processing((1, 3), nmdabar_mult=1)
    plot_generic_2columns([model1.run_data, model2.run_data], enable_stdp=False, filename=f"Fig3_Freq_Dependent_2Columns.png")

def Fig4_GluN2B_Ratio_EPSP():
    weight_increases = []
    nmdas = [1.0, 0.8, 0.25, 0.2, 0.0]
    for i in nmdas:
        model = PylediPlasticity(nmdabar_mult=1, ampabar_mult=1, nr2bbar_mult=i)
        model.run_freq_tests(100, 100, soma_on=False, stdp_dt=10, start=300, afterstop=200)
        weight_increases.append(model.weight)


    epspcut_start = -1500
    epsps = []
    for i, weight in enumerate(weight_increases):
        model = PylediPlasticity(nmdabar_mult=1, ampabar_mult=1, nr2bbar_mult=i)
        model.run_freq_tests(1, 1, soma_on=False, stdp_dt=10, start=300, afterstop=100)
        epsp1 = np.max(model.run_data['v_soma'][epspcut_start:]) - np.min(model.run_data['v_soma'][epspcut_start:])

        model = PylediPlasticity(nmdabar_mult=1, ampabar_mult=1, nr2bbar_mult=i)
        model.customweight = weight
        model.run_freq_tests(1, 1, soma_on=False, stdp_dt=10, start=300, afterstop=100)
        epsp2 = np.max(model.run_data['v_soma'][epspcut_start:]) - np.min(model.run_data['v_soma'][epspcut_start:])

        epspchange = (epsp2 * 100)/epsp1
        epsps.append(epspchange)

    plt.figure(figsize=(14,10))
    ax = plt.subplot()
    print(epsps)
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    ax.bar([str(f"{x:0.2f}") for x  in nmdas], epsps, width=0.8, color=(0.2, 0.2, 0.2), label="Simulation data")
    ax.scatter(np.arange(0, 5), [190, 185, 125, 120, 75], marker="*", color=(0.6, 0.1, 0.9), s=1000, label="Experimental data")
    label_size = 28
    tick_size = 20
    ax.set_ylabel("EPSP (% of change)", fontsize=label_size)
    ax.set_xlabel("Fraction of active GluN2B-NMDAR", fontsize=label_size)
    ax.legend(fontsize=32)
    ax.tick_params(labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig4_GluN2B_Ratio_EPSP.png"))


if __name__ == '__main__':
    
    FIGURES_DIR = "figures"
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    Fig1_STDP()
    Fig2_STDP_2Columns()
    Fig3_Freq_Dependent_2Columns()
    Fig4_GluN2B_Ratio_EPSP()