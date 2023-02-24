from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms

def set_ymargin(ax, bottom=0.0, top=0.0):
    ax.set_ymargin(0)
    ax.autoscale_view()
    lim = ax.get_ylim()
    delta = np.diff(lim)
    bottom = lim[0] - delta*bottom
    top = lim[1] + delta*top
    ax.set_ylim(bottom,top)



#------------------------------------
#STDP 
#------------------------------------
def plot_2columns(datas, filename):
    linewidth = 3
    label_size = 20
    tick_size = 18
    legend_size = 19
    annotate_size = 16
    title_size = 32
    subplotlabel_size = 30
    AB_fontsize = 32
    framealpha = 1.0
    text_x = 0.11
    text_y = 1.22

    color1 = 'darkgreen'
    color2 = 'red'
    color3 = 'cyan'
    color4 = 'black'
    color5 = 'blue'
    
    #------------------------------
    # Fig 2 STDP traces LTP and LTD
    #------------------------------

    starts = [800, 900]

    #-------------
    # Plot Vmem
    #-------------
    simlen = [100*400, 100*400]
    X_ylim = (-0.001, 1)

    # A1 Vmem
    v_ylim = [(-71, 90), (-71, 90)]
    
    # A2 gnmda
    nmda_ylim = [(0.0, 150e-6), (0.0, 150e-6 )]
            
    # A3 gnmda trace or V  
    vtracelim = [(0, 1.7e-5), (0, 1.7e-5)]

    # A4 Theta Hill
    hilllimL = [(0, 2.1e-4), (0, (2.1e-4) )] # gtrace left column LTP
    hilllimR = [(0, 0.1e-2), (0, 0.1e-2 )]   # gtrace right column
    
    # A5 Weight
    weightlim = [(1, 1.05), (0.98, 1.)]


    #-----------------------------------------------
    # General 

    start_stop = [[starts[0], starts[0] + simlen[0]], [starts[1], starts[1] + simlen[1]]]
    fig, axs = plt.subplots(5, 2, figsize=(20, 20))
    fig.subplots_adjust(hspace=0)

    subplotlabel = ["A", "B"]
    column_title = ["LTP", "LTD"]

    for col in [0, 1]:
        axs[0][col].set_title(column_title[col], fontdict={'fontsize': 24, 'fontweight': 'heavy'})
        start, stop = start_stop[col]
        data = datas[col]

        #-------------------------------------------------------------------------------
        # A1
        # vmem
        #-------------------------------------------------------------------------------
        tstart_delta = int(start/10)
        t = np.array(data["t"][start:stop]) - tstart_delta

        start_ax01 = start - 50
        stop_ax01 = 2450 - start_ax01
        tstart_ax01_delta = int(start_ax01/10)
        t_ax01 = np.array(data["t"][start_ax01:stop_ax01]) - tstart_ax01_delta

        lineax01 = axs[0][col].plot(t_ax01, data["v"][start_ax01:stop_ax01], c=color5, label=r"$V_{d}$", linewidth=linewidth)
        lineax011 = axs[0][col].plot(t_ax01, data["v_soma"][start_ax01:stop_ax01], c=color4, label=r"$V_{s}$", linewidth=linewidth)
    
        pre_x = [t[np.argmax(data["dirac_trace"][start_ax01:stop_ax01])]]
        pre_y = [-60]
        axs[0][col].scatter(pre_x, pre_y, label='pre', marker="v", s=500, color=color5)

        #-------------
    
        annotate_dt = 10
        annotatey = 55
        annotatex_text_delta = [7, 7]
        annotatex_delta = pre_x[0]

        if col == 0:
            a_dt = annotate_dt
        elif col == 1:
            a_dt = -annotate_dt

        annotate_text = r"${{\Delta}}T={dt}ms$".format(dt=a_dt)

        axs[0][col].annotate("",
            xy=(annotatex_delta, annotatey),
            xytext=(annotatex_delta + a_dt, annotatey),
            xycoords='data',
            arrowprops=dict(facecolor='black', arrowstyle="<->"))

        axs[0][col].annotate(
            annotate_text,
            xy=(a_dt/2 + annotatex_delta - annotatex_text_delta[col], annotatey + 3),
            xycoords='data',
            fontsize=annotate_size)

        
        #-------------------------------------------------------------------------------
        # A2
        # gnmda
        #-------------------------------------------------------------------------------
        
        axs[1][col].plot(t_ax01, data["g_nr2a"][start_ax01:stop_ax01], c=color2, label=r"$g_{NMDA_{GluN2A}}$", linewidth=linewidth)
        axs[1][col].plot(t_ax01, data["g_nr2b"][start_ax01:stop_ax01], c=color1, label=r"$g_{NMDA_{GluN2B}}$", linewidth=linewidth)
      
        #-------------------------------------------------------------------------------
        # A3
        # gnmda traces 1 - AB LT, 2 - AB LTD, 3 - CD LTD 
        #-------------------------------------------------------------------------------

        axs[2][col].plot(t, data["g_nmda_trace2"][start:stop], c=color2, label=r"$\overline{g}_{NMDA_{-}}$", linewidth=linewidth)
        axs[2][col].plot(t, data["g_nmda_trace1"][start:stop], c=color1, label=r"$\overline{g}_{NMDA_{+}}$", linewidth=linewidth)
       
        #-------------------------------------------------------------------------------
        # A4
        # Hill 1 - AB LT, 2 - AB LTD, 
        #-------------------------------------------------------------------------------
        
        lineax31 = axs[3][col].plot(t, np.array(data["hilleq_ltp"][start:stop]), c=color1, label=r"$\Phi_{NMDA_{+}}$", linewidth=linewidth)
        ax3twin = axs[3][col].twinx()
        lineax32 = ax3twin.plot(t, np.array(data["hilleq_ltd"][start:stop]), c=color2, label=r"$\Phi_{NMDA_{-}}$", linewidth=linewidth)
        
        #-------------------------------------------------------------------------------
        # A5
        # Weights
        #-------------------------------------------------------------------------------

        axs[4][col].plot(t, np.array(data["weight"][start:stop]) - data["weight"][start]+1, c=color4, label=r"$\omega$", linewidth=linewidth)

        # labels
        axs[0][col].set_ylabel("mV", fontsize=label_size)
        axs[1][col].set_ylabel("nS", fontsize=label_size)
        lines00, labels00 = axs[0][col].get_legend_handles_labels()
        lines10, labels10 = axs[3][col].get_legend_handles_labels()
        lines11, labels11 = ax3twin.get_legend_handles_labels()

        axs[4][col].set_xlabel("time (ms)", fontsize=label_size)
        axs[0][col].xaxis.label.set_color(color4)
        axs[3][col].xaxis.label.set_color(color1)
        ax3twin.xaxis.label.set_color(color2)

        # legends
        legend_loc = 1
        bbox_to_anchor = (1.0, 1.0)
        axs[0][col].legend(lines00, labels00, fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=5, fancybox=True, shadow=True)
          
        axs[1][col].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=2, fancybox=True, shadow=True)
        axs[2][col].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=2, fancybox=True, shadow=True)
        ax3twin.legend(lines10 + lines11, labels10 + labels11, fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=2, fancybox=True, shadow=True)
        axs[4][col].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=1, fancybox=True, shadow=True)

        # ticks
        axs[0][col].tick_params(axis='y', colors=color4)
        axs[3][col].tick_params(axis='y', colors=color1)
        ax3twin.tick_params(axis='y', colors=color2)

        scilimits = (-2,3)
        for i in range(5):
            axs[i][col].tick_params(labelsize=tick_size)
            axs[i][col].ticklabel_format(axis='y', scilimits=scilimits, useMathText=True, useOffset=True)
            
        ax3twin.ticklabel_format(axis='y', scilimits=scilimits, useMathText=True, useOffset=True)

        # text
        for i in range(5):
            axs[i][col].text(text_x, text_y, f"{subplotlabel[col]}{i+1})",
                            verticalalignment='top', horizontalalignment='right',
                            transform=axs[i][col].transAxes,
                            color='black', fontsize=AB_fontsize, fontweight='bold')

        # limits

        ax3twin.tick_params(labelsize=tick_size)
        axs[0][col].set_ylim(v_ylim[col])
        axs[1][col].set_ylim(nmda_ylim[col])
        axs[2][col].set_ylim(vtracelim[col])
        axs[3][col].set_ylim(hilllimL[col])
        ax3twin.set_ylim(hilllimR[col])
        axs[4][col].set_ylim(weightlim[col])

        ymarginbottom = 0.05
        ymarginbottom2 = 0.2
        ymargintop = 0.25

        set_ymargin(axs[0][col], ymarginbottom, ymargintop)
        set_ymargin(axs[1][col], ymarginbottom, ymargintop)
        set_ymargin(axs[2][col], ymarginbottom, ymargintop)
        set_ymargin(axs[3][col], ymarginbottom, ymargintop)
        set_ymargin(ax3twin, ymarginbottom, ymargintop)
        set_ymargin(axs[4][col], ymarginbottom2, ymargintop)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")


#-------------------------------------------------------------------
# 3 Columns
#-------------------------------------------------------------------
def plot_3columns(datas, filename):
    linewidth = 3
    label_size = 20
    tick_size = 18
    legend_size = 19
    annotate_size = 16
    title_size = 32
    subplotlabel_size = 30
    AB_fontsize = 32
    framealpha = 1.0
    text_x = 0.11
    text_y = 1.22

    color1 = 'darkgreen'
    color2 = 'red'
    color3 = 'cyan'
    color4 = 'black'
    color5 = 'blue'
    
    #------------------------------
    # Fig 5 Frequency dependent LTP, LTP and LTD
    #------------------------------

    starts = [2890, 2990]
    simlen = [1500, 11000]

    # A1 vmem
    v_ylim = [(-71, 70), (-71, 70), (-71, -50)]
    X_ylim = (-0.001, 0.10)

    # A2 gnmda
    nmda_ylim = [(0.0, 4e-4), (0.0, 4e-4), (0.0, 1e-4)]

    # A3 gnmda traces
    vtracelim = [(0, 1.2e-4), (0, 1.2e-4), (0, 1e-4)]
    
    # A4 Hill
    hilllimL = [(0, 1e-1), (0, 1e-1), (0, 5e-3)]
    hilllimR = [(0, 1e-5), (0, 1e-5), (0, 1e-4)]

    # A5 weight
    weightlim = [(0.95, 1.2), (0.95, 2), (0.99997, 1.00001)]

    #-----------------------------------------------
    # General 

    subplotlabel = ["A", "B", "C"]
    column_title = ["LTP", "LTP", "LTD"]

    start_stop = [[starts[0], starts[0] + simlen[0]], [starts[0], starts[0] + simlen[0]], [starts[1], starts[1] + simlen[1]]]
    
    fig, axs = plt.subplots(5, 3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0)

    for col in [0, 1, 2]:
        axs[0][col].set_title(column_title[col], fontdict={'fontsize': 24, 'fontweight': 'heavy'})
        start, stop = start_stop[col]
        data = datas[col]

        # split view prepare
        if col == 2:
            start = start - 500
            startdelta = 1500
            stopdelta = 1500
            timestart = data["t"][start+startdelta]
            timeend = data["t"][stop-stopdelta]
            data["t"][(stop-stopdelta):] = [(x - timeend + timestart) for x in data["t"][(stop-stopdelta):]]
            for key in data:
                oldlen = len(data[key])
                data[key] = data[key][:(start+startdelta)] + data[key][(stop-stopdelta):]
                newlen = len(data[key])
            stop = stop - (oldlen - newlen)

        tstart_delta = int(start/10)
        t = np.array(data["t"][start:stop]) - tstart_delta

        #-------------------------------------------------------------------------------
        # A1
        # vmem
        #-------------------------------------------------------------------------------
        lineax01 = axs[0][col].plot(t, data["v"][start:stop], c=color5, linestyle='solid', label=r"$V_{d}$", linewidth=linewidth)
        lineax011 = axs[0][col].plot(t, data["v_soma"][start:stop], c=color4, label=r"$V_{s}$", linewidth=linewidth)
        
        if col != 2:
            pre_x = np.arange(10, 150, 10)
            pre_y = [-60] * len(pre_x)
            pre_y = [-77 for x in pre_x]
        else:
            pre_x = [50, 200]
            pre_y = [-60] * len(pre_x)
            pre_y = [-71.2 for x in pre_x]

        axs[0][col].scatter(pre_x, pre_y, label='pre', marker="v", s=60, color=color5)
        
        #-------------------------------------------------------------------------------
        # A2
        # gnmda
        #-------------------------------------------------------------------------------
        axs[1][col].plot(t, data["g_nr2a"][start:stop], c=color1, label=r"$g_{NMDA_{GluN2A}}$", linewidth=linewidth)
        axs[1][col].plot(t, data["g_nr2b"][start:stop], c=color2, label=r"$g_{NMDA_{GluN2B}}$", linewidth=linewidth)

        #-------------------------------------------------------------------------------
        # A3
        # gnmda traces 1 - AB LT, 2 - AB LTD, 3 - CD LTD
        #-------------------------------------------------------------------------------
        axs[2][col].plot(t, data["g_nmda_trace1"][start:stop], c=color1, label=r"$\overline{g}_{NMDA_{+}}$", linewidth=linewidth)
        axs[2][col].plot(t, data["g_nmda_trace2"][start:stop], c=color2, label=r"$\overline{g}_{NMDA_{-}}$", linewidth=linewidth)

        #-------------------------------------------------------------------------------
        # A4
        # Hill eq
        #-------------------------------------------------------------------------------

        lineax31 = axs[3][col].plot(t, np.array(data["hilleq_ltp"][start:stop]), c=color1, label=r"$\Phi_{NMDA_{+}}$", linewidth=linewidth)
        
        # instantiate a second axes that shares the same x-axis
        ax3twin = axs[3][col].twinx()
        lineax32 = ax3twin.plot(t, np.array(data["hilleq_ltd"][start:stop]), c=color2, label=r"$\Phi_{NMDA_{-}}$", linewidth=linewidth)
       
        #-------------------------------------------------------------------------------
        # A5
        # Weights
        #-------------------------------------------------------------------------------
        axs[4][col].plot(t, np.array(data["weight"][start:stop]) - data["weight"][start]+1, c=color4, label=r"$\omega$", linewidth=linewidth)

        # split view
        if col == 2:
            timestart -= tstart_delta
            for ax in axs:
                ax[col].axvline(timestart, c=color4)
                xticks = [0, 50, 100, 150, 200, 250, 300]
                xticks_forlabels = [0, 50, 100, 150, 1100, 1150, 1200]
                xticklabels = [str(int(x)) for x in xticks_forlabels]
                ax[col].set_xticks(xticks)
                ax[col].set_xticklabels(xticklabels)
                
        # labels
        axs[4][col].set_xlabel("time (ms)", fontsize=label_size)
        axs[0][col].set_ylabel("mV", fontsize=label_size)
        axs[1][col].set_ylabel("nS", fontsize=label_size)
        axs[0][col].xaxis.label.set_color(color4)

        # legends
        lines00, labels00 = axs[0][col].get_legend_handles_labels()
        lines10, labels10 = axs[3][col].get_legend_handles_labels()
        lines11, labels11 = ax3twin.get_legend_handles_labels()

        legend_loc = 1
        bbox_to_anchor = (1.0, 1.0)
        axs[0][col].legend(lines00, labels00, fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=5, fancybox=True, shadow=True)
          
        axs[1][col].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=2, fancybox=True, shadow=True)
        axs[2][col].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=2, fancybox=True, shadow=True)
        ax3twin.legend(lines10 + lines11, labels10 + labels11, fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=2, fancybox=True, shadow=True)
        axs[4][col].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
          ncol=1, fancybox=True, shadow=True)

        # ticks
        axs[0][col].tick_params(axis='y', colors=color4)
        axs[3][col].tick_params(axis='y', colors=color1)
        ax3twin.tick_params(axis='y', colors=color2)

        scilimits = (-2,3)
        for i in range(5):
            axs[i][col].tick_params(labelsize=tick_size)
            axs[i][col].ticklabel_format(axis='y', scilimits=scilimits, useMathText=True, useOffset=True)
        ax3twin.ticklabel_format(axis='y', scilimits=scilimits, useMathText=True, useOffset=True)
        ax3twin.tick_params(labelsize=tick_size)

        # text
        for i in range(5):
            axs[i][col].text(text_x, text_y, f"{subplotlabel[col]}{i+1})",
                            verticalalignment='top', horizontalalignment='right',
                            transform=axs[i][col].transAxes,
                            color='black', fontsize=AB_fontsize, fontweight='bold')

        # limits
        axs[0][col].set_ylim(v_ylim[col])
        axs[1][col].set_ylim(nmda_ylim[col])
        axs[2][col].set_ylim(vtracelim[col])
        axs[3][col].set_ylim(hilllimL[col])
        ax3twin.set_ylim(hilllimR[col])
        axs[4][col].set_ylim(weightlim[col])

        ymarginbottom = 0.05
        ymarginbottom2 = 0.2
        ymargintop = 0.4

        set_ymargin(axs[0][col], ymarginbottom, ymargintop)
        set_ymargin(axs[1][col], ymarginbottom, ymargintop)
        set_ymargin(axs[2][col], ymarginbottom, ymargintop)
        set_ymargin(axs[3][col], ymarginbottom, ymargintop)
        set_ymargin(ax3twin, ymarginbottom, ymargintop)
        set_ymargin(axs[4][col], ymarginbottom2, ymargintop)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")


#STDP BP 

linewidth_STDP_full=3 
linewidth_STDP_full_WW_5pairs=5
linewidth_STDP_full_BP=5 
color_STDP='blue'

def plot_stdp_full_BP(data, filename=None):
    stdp_range = range(-100, 110, 10)
    x = [a["weight"][-1] for a in data]
    plt.figure()
    plt.margins(x=0, y=0) 
    plt.vlines(0.0, 0.5, 2.0, colors=(0.7, 0.7, 0.7), linestyles='dashed') #for 5 Hz
    #plt.vlines(0.0, 0.8, 1.6, colors=(0.7, 0.7, 0.7), linestyles='dashed') #for 1 Hz
    
    plt.hlines(1.0, -100.0, 100.0, colors=(0.7, 0.7, 0.7), linestyles='dashed')
    plt.plot(stdp_range, x, c=color_STDP, linewidth=linewidth_STDP_full_BP)
    plt.xlabel(r'${\Delta}T  \:$ (ms)', fontsize=20)
    plt.ylabel("Relative weight change", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if filename == None:
        filename = f"figures/Fig1_STDP.png"
    plt.savefig(filename)


#-----------------------------------
#STDP 5 pairs
def plot_stdp_full_5pairs(datas, filename=None, plotparam='weight'):
    plt.figure()
    plt.margins(x=0, y=0) 
    
    #y axis
    #plt.vlines(0.0, 0.5, 2.0, colors=(0.7, 0.7, 0.7), linestyles='dashed') #for BP 5 Hz
    plt.vlines(0.0, 0.8, 1.6, colors=(0.7, 0.7, 0.7), linestyles='dashed') #for 1 Hz 
    plt.hlines(1.0, -100.0, 100.0, colors=(0.7, 0.7, 0.7), linestyles='dashed') 
    stdp_range = range(-100, 110, 10)
    for idx, data in enumerate(datas):
        x = [a[plotparam][-1] for a in data]

        plt.plot(stdp_range, x, c=color_STDP, linewidth=linewidth_STDP_full_WW_5pairs )
    
    plt.xlabel(r'${\Delta}T \:$ (ms)', fontsize=20)
    plt.ylabel("Relative weight change", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    if filename == None:
        filename = f"figures/Fig1_STDP.png"
    plt.savefig(filename)

#-------------------------------------
#STDP 60 pairs
def plot_stdp_full_60pairs(datas, filename=None, plotparam='weight'):
    plt.figure()
    plt.margins(x=0, y=0) 
    
    plt.vlines(0.0, 0.5, 2.0, colors=(0.7, 0.7, 0.7), linestyles='dashed')
    plt.hlines(1.0, -100.0, 100.0, colors=(0.7, 0.7, 0.7), linestyles='dashed')
    stdp_range = range(-100, 110, 10)
    for idx, data in enumerate(datas):
        x = [a[plotparam][-1] for a in data]

        plt.plot(stdp_range, x, c=color_STDP, linewidth=linewidth_STDP_full )
    
    plt.xlabel(r'${\Delta}T \:$ (ms)', fontsize=20)
    plt.ylabel("Relative weight change", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    if filename == None:
        filename = f"figures/Fig1_STDP.png"
    plt.savefig(filename)



#-----------------------------------------------
#STDP 


def plot_stdp_multiple(datas, filename=None, plotparam='weight'):
    plt.figure()
    plt.margins(x=0, y=0) 
    
    plt.vlines(0.0, 0.5, 2.0, colors=(0.7, 0.7, 0.7), linestyles='dashed')
    plt.hlines(1.0, -100.0, 100.0, colors=(0.7, 0.7, 0.7), linestyles='dashed')
    stdp_range = range(-100, 110, 10)
    for idx, data in enumerate(datas):
        x = [a[plotparam][-1] for a in data]

        plt.plot(stdp_range, x, c=color_STDP, linewidth=linewidth_STDP_full )
    
    plt.xlabel(r'${\Delta}T \:$ (ms)', fontsize=20)
    plt.ylabel("Relative weight change", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    if filename == None:
        filename = f"figures/Fig1_STDP.png"
    plt.savefig(filename)

    