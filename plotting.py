import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def plot_single_traj(normal_modes, average_interval, nm_std, time_intervals):

    interval_labels = format_string(time_intervals)
    nvibs = 9
    ntints = 2
    vibs = np.arange(nvibs)+1
    time = np.arange(0, 1000.5, 0.5)
    offset = 0.25
    offsets = np.linspace(0, (offset*ntints - offset), ntints)
    time_int_str = format_string(time_intervals)

    plot_vib_bar(vibs, offsets, offset, nm_std, ntints, 'Standard Deviation', time_int_str, 'std.png')
    plot_vib_bar(vibs, offsets, offset, average_interval, ntints, 'Average Value', time_int_str, 'avg.png')
    plot_normal_mode(normal_modes, time, nvibs, 'normal_mode')
    print('hi')


def plot_normal_mode(normal_modes: npt.NDArray, time: npt.NDArray, nvibs: int, fileprefix: str):
    for i in range(nvibs):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(time, normal_modes[i, :])
        ax.set_xlabel('Time (fs)')
        ax.set_xticks(time)
        title_string = 'Normal mode %s' % (i+1)
        ax.set_title(title_string)
        fig_str = fileprefix + ('_%s.png' % i)
        plt.savefig(fig_str)



def plot_vib_bar(vibs: npt.NDArray, offsets:npt.NDArray, offset: int, y_data: npt.NDArray, ntints:int, y_label: str, legend_labels: list, filename: str):
    fig, ax = plt.subplots(figsize=(5,5))
    for i in range(ntints):
        ax.bar(vibs+offsets[i], y_data[:, i], width=offset)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Normal Mode Index')
    ax.set_xticks(vibs)
    ax.legend(labels=legend_labels)
    plt.savefig(filename)



def format_string(x):
    strs = []
    for i in x:
        entry = "%s - %s fs" % (str(int(i[0])), str(int(i[1])))
        strs.append(entry)
    return strs