import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def report_sum2(path='./data/report/sum2/loss'):
    graph={}
    for fname in os.listdir(path):
        if 'previous' in fname:
            continue
        ffname = os.path.join(path, fname)
        df = pd.read_csv(ffname,
                header=0,
                usecols=["Step", "Value"],
                )
        graph[fname]={'x':[],'y':[]}
        for index, row in df.iterrows():
            # print(row['Step'], row['Value'])
            graph[fname]['x'].append(row['Step'])
            graph[fname]['y'].append(row['Value'])
            # if index>=50:
            #     break

    dashList = [(5, 2), (2, 5), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2)]
    # List of Dash styles, each as integers in the format: (first line length, first space length, second line length, second space length...)
    plt.xlabel('Step')
    plt.ylabel("Loss")  # add a label to the y axis
    plots=[]
    pnames=[]
    c=0
    linestyles = ['-', '--',':','-.']
    markers = ['+','^', '*']
    for k,v in sorted(graph.items()):
        if c<len(linestyles):
            print(len(v['x']))
            print(len(v['y']))
            plots.append(plt.plot(v['x'][:91], np.convolve(v['y'],np.ones(11)/11,mode='valid')[:91],
                                  linestyle=linestyles[c%len(linestyles)]))
        else:
            plots.append(plt.plot(v['x'][:91], np.convolve(v['y'], np.ones(11) / 11, mode='valid')[:91],
                                  marker=markers[c % len(markers)]))
        pnames.append(k[:-4])
        c+=1
    plt.legend(pnames,shadow=True, fancybox=True, loc='lower right')

    plt.show()


def plot_cs_diag_proc(type='diag'):
    ys1diag = [0.44189042, 0.42628044, 0.35347179, 0.35972559,
           0.51927823, 0.29307148, 0.26074234, 0.2494036,
           0.26178563, 0.16837306]#, 0.29467991]

    ys2diag = [0.43882078, 0.37130833, 0.3899295, 0.39653572,
           0.50258982, 0.392995, 0.21722959, 0.30444258,
           0.19008981, 0.34303573]#, 0.31987008]

    ys1proc = [0.24392338, 0.18665618, 0.27594608, 0.20218483, 0.24657778]
    ys2proc = [0.88602263, 0.54703015, 0.72378427, 0.5829525, 0.39077044]

    xsdiag=[57411, 60883, 78901, 9695, 41042, 99812, 40200, 7804, 45981, 5789]#, 80420]
    xsproc=[3613, 3615, 3961, 3404, 114]
    if type=='diag':
        ys1=ys1diag
        ys2=ys2diag
        xs=xsdiag

    elif type=='proc':
        ys1=ys1proc
        ys2=ys2proc
        xs=xsproc
    else:
        ys1=ys1diag+ys1proc
        ys2=ys2diag+ys2proc
        xs=xsdiag+xsproc


    N = len(xs)
    men_means = ys1

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, men_means, width, color='r')

    women_means = ys2

    rects2 = ax.bar(ind + width, women_means, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Write gate value')
    if type=='diag':
        ax.set_xlabel('Diagnosis codes')
    else:
        ax.set_xlabel('Procedure codes')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xs)
    # ax.set_ylim([0, 1])
    ax.legend((rects1[0], rects2[0]), ('Late-fusion', 'Early-fusion'))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # report_sum2(path='./data/report/sum2_hard/acc')
    plot_cs_diag_proc(type='diag')
    # plot_cs_diag_proc(type='proc')
    # plot_cs_diag_proc(type='dp')