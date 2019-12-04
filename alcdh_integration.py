import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import signal

from chromauto.utils import find_area, add_plot, walk_peaks, peak_detect

import csv


def ALCDH_Integrate(real_x, real_y, blank_x, blank_y, tmp_prefix, **kwargs):
    """

    Find peaks
    Walk over peaks to sort out areas
    Write out overview image
    Write out areas as csv

    :param real_x: data real x values
    :param real_y: data real y values
    :param blank_x: blank x values
    :param blank_y: blank y values
    :param tmp_prefix: prefix file name to save file
    :param kwargs: settings
    """

    data_y_win_length = 111 if 'data_y_win_length' not in kwargs else kwargs['data_y_win_length']
    blank_y_win_length = 333 if 'blank_y_win_length' not in kwargs else kwargs['blank_y_win_length']
    savgol_polyorder = 7 if 'savgol_polyorder' not in kwargs else kwargs['savgol_polyorder']
    peak_min_dist = 20 if 'peak_min_dist' not in kwargs else kwargs['peak_min_dist']
    peak_max_peak = 3 if 'peak_max_peak' not in kwargs else kwargs['peak_max_peak']
    peak_thres_range = (1, 5, .1) if 'peak_thres_range' not in kwargs else kwargs['peak_thres_range']
    show_file = False if 'show_file' not in kwargs else kwargs['show_file']

    grid = []

    react_text = ''

    max_x = blank_x.size - 1

    smooth_real_y = np.array(signal.savgol_filter(real_y, window_length=data_y_win_length, polyorder=savgol_polyorder))
    smooth_blank_y = np.array(
        signal.savgol_filter(blank_y, window_length=blank_y_win_length, polyorder=savgol_polyorder))

    indexes = peak_detect(smooth_real_y - smooth_blank_y, xran=peak_thres_range, min_dist=peak_min_dist,
                          max_peak=peak_max_peak)

    # Find peaks
    fpks = walk_peaks(indexes, real_y, blank_y, real_x, max_x, **kwargs)

    all_areas = 0.0

    csvout = [['area', 'start time', 'end time', ]]

    # append plots to grid
    # create fancy polygon to shade areas

    for idx, i in enumerate(fpks):
        startx = i[0]['x']
        endx = i[1]['x']
        a = real_x[startx]
        b = real_x[endx]
        ix = real_x[startx:endx]
        iy = real_y[startx:endx]
        start_vert = real_y[startx]
        end_vert = real_y[endx]

        # area of trapazoid below the curve
        trapazoid_area = ((real_y[startx] + real_y[endx]) / 2.0) * (real_x[endx] - real_x[startx])

        verts = [(a, start_vert)] + list(zip(ix, iy)) + [(b, end_vert)]
        poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        area = find_area(startx, endx, real_y, real_x, trapazoid_area)
        cout = ['%.3f' % area, '%.3f' % a, '%.3f' % b]
        axis_x = (real_x[startx - 150], real_x[min(endx + 150, max_x)])
        axis_y = (np.min(real_y[startx:endx]) - 0.4, np.max(real_y[startx:endx]) + .4)

        csvout.append(cout)

        all_areas += area

        grid.append(
            {'patch': poly, 'axis': [axis_x[0], axis_x[1], axis_y[0], axis_y[1]],
             'plots': [(real_x, real_y, 'b'), (real_x, blank_y, 'g')], 'txt': (0.75, 0.75, 'area: ' + '%.3f' % area)})

    # # graph everyone into overview image
    gl = len(grid) / 2 + 1 if not len(grid) % 2 else len(grid) / 2 + 2
    gl = int(gl + 1)
    #
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 20}

    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=.5, wspace=.05)

    ax1 = plt.subplot2grid((gl, 2), (0, 0), colspan=2, rowspan=2)

    overview_text = react_text
    # add overview image for first box in overview image
    add_plot(ax1, {'axis': [0, real_x[max_x], 0, np.max(real_y)],
                   'plots': [[real_x, real_y, 'b'], [blank_x, blank_y, 'g'], [blank_x, real_y - blank_y, 'r']],
                   'txt': (0.35, 0.7, overview_text)})

    for i in range(0, len(grid), 2):
        row = int((i + 1) / 2 + 2)
        if i + 1 < len(grid):
            add_plot(plt.subplot2grid((gl, 2), (row, 0), colspan=1), grid[i])
            add_plot(plt.subplot2grid((gl, 2), (row, 1), colspan=1), grid[i + 1])
        else:
            add_plot(plt.subplot2grid((gl, 2), (row, 0), colspan=2), grid[i])

    fig.tight_layout()

    # # useful when using ipython notebook for interactive viewing
    # plt.show(block=True)

    if show_file:
        plt.show()
    else:
        fig.savefig(tmp_prefix + '_overview.png')

    plt.close()
    plt.clf()

    with open(tmp_prefix + '_txt.csv', 'w') as f:
        csvlog = csv.writer(f)
        for i in csvout:
            csvlog.writerow(i)

    return csvout
