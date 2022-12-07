from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms


def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta*left
    right = lim[1] + delta*right
    ax.set_xlim(left,right)

def set_ymargin(ax, bottom=0.0, top=0.0):
    ax.set_ymargin(0)
    ax.autoscale_view()
    lim = ax.get_ylim()
    delta = np.diff(lim)
    bottom = lim[0] - delta*bottom
    top = lim[1] + delta*top
    ax.set_ylim(bottom,top)

def plot_generic_2columns(datas, enable_stdp=False, filename=""):
    linewidth = 3
    label_size = 20
    tick_size = 18
    legend_size = 19
    annotate_size = 16
    title_size = 32
    subplotlabel_size = 30
    AB_fontsize = 32
    legend_loc = [1]*5
    bbox_to_anchor = [(1.0, 1.0)] * 5
    ABtext_color = (0.0, 0.0, 0.0)
    framealpha = 1.0
    text_x = 0.11
    text_y = 1.22
    
    if enable_stdp:
        
        starts = [890, 990]
        simlen = [400, 400]

        v_ylim = [(-71, 30), (-71, 30)]
        X_ylim = (-0.001, 0.10)
        nmda_ylim = [(0.0, 0.2), (0.0, 0.2 * 0.2)]
        vtracelim = [(0, 20), (0, 20)]
        hilllimL = [(0, 3e-2), (0, (3e-2) * 0.2)]
        hilllimR = [(0, 1.5), (0, 1.5 * 0.2)]
        weightlim = [(0.985, 1.03), (0.992, 1.002)]

        ymarginbottom = 0.05
        ymarginbottom2 = 0.2
        ymargintop = 0.25
    else:
        starts = [2890, 2990]
        simlen = [2000, 11000]

        v_ylim = [(-71, 70), (-71, -65)]
        X_ylim = (-0.001, 0.10)
        nmda_ylim = [(0.0, 0.4), (0.0, 0.4 * 0.1)]
        vtracelim = [(0, 25), (0, 25 * 0.1)]
        hilllimL = [(0, 1.0), (0, 1.0 * 0.2)]
        hilllimR = [(0, 1.0), (0, 1.0 * 0.2)]
        weightlim = [(0.80, 1.45), (0.99997, 1.00001)]

        ymarginbottom = 0.05
        ymarginbottom2 = 0.2
        ymargintop = 0.4
        
    st_st = [[starts[0], starts[0] + simlen[0]], [starts[1], starts[1] + simlen[1]]]
    
    annotatey = 50
    annotatex_delta = [[3.0, 13.0], [3.0, 13.0]]
    annotatex_text_delta = [3.4, 4]
    annotatey_text_delta = [13, 13]

    fig, axs = plt.subplots(5, 2, figsize=(20, 20))

    fig.subplots_adjust(hspace=0)

    subplotlabel = ["A", "B"]
    column_title = ["LTP", "LTD"]

    annotate_texts = [r"${\Delta}T=-10ms$", r"${\Delta}T=10ms$"]
    annotate_texts.reverse()
    for col in [0, 1]:
        axnum = col
        axs[0][axnum].set_title(column_title[axnum], fontdict={'fontsize': 24, 'fontweight': 'heavy'})
        start, stop = st_st[col]
        data = datas[col]
        if col == 1 and not enable_stdp:
            start = start - 500
            startdelta = 1500
            stopdelta = 1500
            timestart = data["t"][start+startdelta]
            timeend = data["t"][stop-stopdelta]
            xaxisticks = data["t"][(stop-stopdelta):stop] 
            data["t"][(stop-stopdelta):] = [(x - timeend + timestart) for x in data["t"][(stop-stopdelta):]]
            for key in data:
                oldlen = len(data[key])
                data[key] = data[key][:(start+startdelta)] + data[key][(stop-stopdelta):]
                newlen = len(data[key])
            stop = stop - (oldlen - newlen)

        tstart_delta = int(start/10)
        t = np.array(data["t"][start:stop]) - tstart_delta

        c0 = (0.8, 0.1, 0.1)  # delta trace
        c1 = (0.0, 0.0, 0.0)  # vmem
        c11 = (0.2, 0.9, 0.2)  # thresh ltp
        c12 = (0.9, 0.2, 0.2)  # thresh ltd
        c21 = (0.1, 0.8, 0.1)  # vmem trace ltp
        c22 = (0.8, 0.1, 0.1)  # vmem trace ltd
        c31 = (0.1, 0.8, 0.1)  # hill ltp
        c32 = (0.8, 0.1, 0.1)  # hill ltd
        c4 = (0.9, 0.5, 0.1)  # gnmda trace
        c5 = (0.1, 0.1, 0.1)  # weight

        c111 = [c0, c1]
        c2 = [c21, c22]
        c3 = [c31, c32]

        l0 = r"$\overline{X}$"
        l1 = r"$V_{d}$"
        l2 = r"$V_{s}$"
        l11 = r"$\theta_{+}$"
        l12 = r"$\theta_{+}$"
        l21 = r"$\overline{V}_{+}$"
        l22 = r"$\overline{V}_{-}$"
        l31 = r"$\Phi_{NMDA_{+}}$"
        l32 = r"$\Phi_{NMDA_{-}}$"
        l41 = r"$g_{NMDA_{GluN2A}}$"
        l42 = r"$g_{NMDA_{GluN2B}}$"
        l4 = r"$\overline{g}_{nmda}$"
        l5 = r"$\omega$"


        lineax01 = axs[0][axnum].plot(t, data["v"][start:stop], c=c1, linestyle='dashed', label=l1, linewidth=linewidth)
        lineax011 = axs[0][axnum].plot(t, data["v_soma"][start:stop], c=c1, label=l2, linewidth=linewidth)
        
        if enable_stdp:
            pre_x = [t[np.argmax(data["dirac_trace"][start:stop])]]
            pre_y = [-60]
            axs[0][axnum].scatter(pre_x, pre_y, label='pre', marker="v", s=500, color="red")
        else:
            if axnum == 0:
                pre_x = np.arange(10, 200, 10)
                pre_y = [-60] * len(pre_x)
                pre_y = [-77 for x in pre_x]
            else:
                pre_x = [50, 200]
                pre_y = [-60] * len(pre_x)
                pre_y = [-71.2 for x in pre_x]
            
            
            axs[0][axnum].scatter(pre_x, pre_y, label='pre', marker=".", s=60, color="red")

        
        axs[1][axnum].plot(t, data["g_nr2a"]
                         [start:stop], c=c0, label=l41, linewidth=linewidth)
        axs[1][axnum].plot(t, data["g_nr2b"]
                         [start:stop], c=c1, label=l42, linewidth=linewidth)

        axs[2][axnum].plot(t, data["v_trace1_threshed1"]
                         [start:stop], c=c21, label=l21, linewidth=linewidth)
        axs[2][axnum].plot(t, data["v_trace2_threshed2"]
                         [start:stop], c=c22, label=l22, linewidth=linewidth)
        lineax31 = axs[3][axnum].plot(t, np.array(data["hilleq_ltp"]
                         [start:stop]), c=c31, label=l31, linewidth=linewidth)
        
        if enable_stdp:
            axs[0][axnum].annotate("", xy=(annotatex_delta[col][0], annotatey), xytext=(annotatex_delta[col][1], annotatey), xycoords='data',
                        arrowprops=dict(facecolor='black', arrowstyle="<->"))
            axs[0][axnum].annotate(annotate_texts[col], xy=((annotatex_delta[col][1] - annotatex_delta[col][0])/2 + annotatex_delta[col][0] - annotatex_text_delta[col], annotatey - annotatey_text_delta[col]), xycoords='data', fontsize=annotate_size)

        # instantiate a second axes that shares the same x-axis
        ax3twin = axs[3][axnum].twinx()
        lineax32 = ax3twin.plot(t, np.array(data["hilleq_ltd"]
                                 [start:stop]), c=c32, label=l32, linewidth=linewidth)
        axs[4][axnum].plot(t, np.array(data["weight"][start:stop]) - data["weight"][start]+1, c=c5, label=l5, linewidth=linewidth)
        #axs[4][col].plot(t, np.array(data["g_nmda_trace1"][start:stop]) * 1000, c=c4, label=l4)

        if col == 1 and not enable_stdp:
            timestart -= tstart_delta
            for ax in axs:
                ax[axnum].axvline(timestart, c=(0.1, 0.1, 0.1))
                xticks = [0, 50, 100, 150, 200, 250, 300]
                xticks_forlabels = [0, 50, 100, 150, 1100, 1150, 1200]
                xticklabels = [str(int(x)) for x in xticks_forlabels]
                ax[axnum].set_xticks(xticks)
                ax[axnum].set_xticklabels(xticklabels)
        #axs[0].set_ylabel("\delta trace")
        axs[0][axnum].set_ylabel("mV", fontsize=label_size)
        axs[1][axnum].set_ylabel("nS", fontsize=label_size)
        lines00, labels00 = axs[0][axnum].get_legend_handles_labels()
        lines10, labels10 = axs[3][axnum].get_legend_handles_labels()
        lines11, labels11 = ax3twin.get_legend_handles_labels()

        lns0 = lines00
        labs0 = labels00
        
        lns3 = lines10 + lines11
        labs3 = labels10 + labels11

        axs[4][axnum].set_xlabel("time (ms)", fontsize=label_size)

        axs[0][axnum].legend(lns0, labs0, fontsize=legend_size, framealpha=framealpha, loc=legend_loc[0], bbox_to_anchor=bbox_to_anchor[0],
          ncol=5, fancybox=True, shadow=True)
          
        axs[1][axnum].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc[1], bbox_to_anchor=bbox_to_anchor[1],
          ncol=2, fancybox=True, shadow=True)
        axs[2][axnum].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc[2], bbox_to_anchor=bbox_to_anchor[2],
          ncol=2, fancybox=True, shadow=True)
        ax3twin.legend(lns3, labs3, fontsize=legend_size, framealpha=framealpha, loc=legend_loc[3], bbox_to_anchor=bbox_to_anchor[3],
          ncol=2, fancybox=True, shadow=True)
        axs[4][axnum].legend(fontsize=legend_size, framealpha=framealpha, loc=legend_loc[4], bbox_to_anchor=bbox_to_anchor[4],
          ncol=1, fancybox=True, shadow=True)

        axs[0][axnum].xaxis.label.set_color(c111[1])
        axs[0][axnum].tick_params(axis='y', colors=c111[1])
        axs[3][axnum].xaxis.label.set_color(c2[0])
        axs[3][axnum].tick_params(axis='y', colors=c2[0])
        ax3twin.xaxis.label.set_color(c2[1])
        ax3twin.tick_params(axis='y', colors=c2[1])

        for i in range(5):
            axs[i][axnum].text(text_x, text_y, f"{subplotlabel[col]}{i+1})",
                            verticalalignment='top', horizontalalignment='right',
                            transform=axs[i][axnum].transAxes,
                            color=ABtext_color, fontsize=AB_fontsize, fontweight='bold')

        scilimits = (-2,3)
        for i in range(5):
            axs[i][axnum].tick_params(labelsize=tick_size)
            axs[i][axnum].ticklabel_format(axis='y', scilimits=scilimits, useMathText=True, useOffset=True)
            
        ax3twin.ticklabel_format(axis='y', scilimits=scilimits, useMathText=True, useOffset=True)

        ax3twin.tick_params(labelsize=tick_size)
        axs[0][axnum].set_ylim(v_ylim[col])
        axs[1][axnum].set_ylim(nmda_ylim[col])
        axs[2][axnum].set_ylim(vtracelim[col])
        axs[3][axnum].set_ylim(hilllimL[col])
        ax3twin.set_ylim(hilllimR[col])
        axs[4][axnum].set_ylim(weightlim[col])


        set_ymargin(axs[0][axnum], ymarginbottom, ymargintop)
        set_ymargin(axs[1][axnum], ymarginbottom, ymargintop)
        set_ymargin(axs[2][axnum], ymarginbottom, ymargintop)
        set_ymargin(axs[3][axnum], ymarginbottom, ymargintop)

        set_ymargin(ax3twin, ymarginbottom, ymargintop)

        set_ymargin(axs[4][axnum], ymarginbottom2, ymargintop)


    plt.tight_layout()

    if (filename != ""):
        plt.savefig(f"figures/{filename}")
    else:
        plt.show(block=False)



def plot_stdp(data):
    stdp_range = range(-100, 110, 10)
    x = [a["weight"][-1] for a in data]
    print(x)
    plt.figure()
    plt.margins(x=0, y=0) 
    plt.vlines(0.0, 0.5, 2.0, colors=(0.7, 0.7, 0.7), linestyles='dashed')
    plt.hlines(1.0, -100.0, 100.0, colors=(0.7, 0.7, 0.7), linestyles='dashed')
    plt.plot(stdp_range, x, c=(0.0, 0.0, 0.1))
    plt.xlabel(r'${\Delta}T \: (ms)$', fontsize=20)
    plt.ylabel("Relative weight change", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"figures/Fig1_STDP.png")