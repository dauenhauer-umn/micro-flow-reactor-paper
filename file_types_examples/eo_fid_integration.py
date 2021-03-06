# TODO:
# Split peaks ->
# df
#####Small min area for acetaldehyde
# Increase minimum area to dodge noise
# Put savgol filtering back in
# Change min slope
# Change # time steps

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import signal
from chromauto.utils import find_area, peak_detect, walk_peaks, add_plot, splice_peaks, sep_peak
import csv


def EO_Integrate(real_x, real_y, blank_x, blank_y, tmp_prefix, **kwargs):
    peak_hard_thres = .0003 if 'peak_hard_thres' not in kwargs else kwargs['peak_hard_thres']
    show_file = False if 'show_file' not in kwargs else kwargs['show_file']
    spec_range = [] if 'spec_range' not in kwargs else kwargs['spec_range']
    date_stamp = [] if 'date_stamp' not in kwargs else kwargs['date_stamp']
    time_stamp = [] if 'time_stamp' not in kwargs else kwargs['time_stamp']

    if real_y[0] - blank_y[0] < 0:
        blank_y = blank_y + (real_y[0] - blank_y[0]) - 1

    data_dict = {}
    grid = []

    interest_text = ''
    react_text = ''
    # x, y are the real run values (blue line)
    x = real_x
    y = real_y

    # bx, by are the blank run values (green line)
    bx = blank_x
    by = blank_y
    # plt.figure()

    # apply savgol_filter to (ny) real_y and (vy) blank_y These will be the values that are used from now on
    # ny-vy is the red line
    # ny = np.array(signal.savgol_filter(y, window_length=19, polyorder=7))
    # ny = np.array(signal.savgol_filter(y, window_length=5, polyorder=4))
    # vy = np.array(signal.savgol_filter(by, window_length=111, polyorder=5))
    ny = y
    vy = by

    # ny = np.array(signal.savgol_filter(y, window_length=111, polyorder=7))
    # vy = np.array(signal.savgol_filter(by, window_length=111, polyorder=5))
    # ny = y
    # vy = by

    max_x = bx.size - 1

    base_y = y - by

    indexes = peak_detect(base_y, thres=peak_hard_thres)
    # print(indexes, [x[i] for i in indexes])

    fpks = walk_peaks(indexes, ny, by, x, max_x, **kwargs)
    # print('fpks', fpks)

    peak_splice = splice_peaks(fpks, base_y, ny, x, peak_range=spec_range, **kwargs)

    # remove old peak, insert new peaks
    # print('FPKS', fpks)
    # print(peak_splice)

    for idx, i in enumerate(peak_splice):
        oldix = fpks.index(i['i'])
        fpks.remove(i['i'])
        for pidx, p in enumerate(i['new']):
            fpks.insert(pidx + oldix, p)

    all_areas = 0.0
    react_area = 0.0
    interest_area = 0.0
    # graph the first round of peak detection
    peak_areas = []

    # print('FPKS2', fpks)

    csvout = [['area', 'start time', 'end time', ]]

    for idx, i in enumerate(fpks):
        # startx = 29804
        # endx = 32900
        startx = i[0]['x']
        endx = i[1]['x']
        a = x[startx]
        b = x[endx]
        ix = x[startx:endx]
        iy = ny[startx:endx]
        multi = ''
        start_vert = ny[startx]
        end_vert = ny[endx]
        # area below the curve (xaxis to yline)
        # trapizod area calculation

        ly = (ny[startx] + ny[endx]) / 2.0 * (x[endx] - x[startx])

        if i[0]['multi']:
            if idx + 1 < len(fpks):
                # if a split peak, calculate the slope and recalc verts and area below curve
                # split peaks share next start or prev end
                if endx == fpks[idx + 1][0]['x']:
                    multi = 'end'
                    sly = i[1]['sly']
                    end_vert = sly
                    ly = (ny[startx] + sly) / 2.0 * (x[endx] - x[startx])
            if idx > 0:
                if startx == fpks[idx - 1][1]['x']:
                    multi = 'start'
                    sly = i[0]['sly']
                    start_vert = sly
                    ly = (ny[endx] + sly) / 2.0 * (x[endx] - x[startx])

        verts = [(a, start_vert)] + list(zip(ix, iy)) + [(b, end_vert)]
        poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        sa = find_area(startx, endx, ny, x, ly)
        cout = [sa, a, b]
        axis_x = (x[startx - 150], x[min(endx + 150, max_x)])
        axis_y = (np.min(ny[startx:endx]) - 0.4, np.max(ny[startx:endx]) + .4)

        # hack for special area where split peak differently
        # spec_range = [{'range': (11,13), 'thres': .0003,'area':.0001},]
        for p in spec_range:
            if x[startx] > p['range'][0] and x[endx] < p['range'][1]:
                # print("GO", i)
                sep_results = sep_peak(x, ny, base_y, idx, i, fpks, **kwargs)
                # print("sep", sep_results)
                if sep_results:
                    # print('results', sep_results)
                    poly = sep_results[0]
                    sa = sep_results[1]
                    cout = sep_results[2]
                    axis_x = sep_results[3]
                    axis_y = sep_results[4]
        #
        # attrs = {'standard': 'ndec', 'moles': 0.020252, 'volume': 507, 'catalyst': 'AlMCM41', 'cat_amount': 75.2}
        csvout.append(cout)

        all_areas += sa

        grid.append(
            {'patch': poly, 'axis': [axis_x[0], axis_x[1], axis_y[0], axis_y[1]],
             'plots': [(x, y, 'b'), (x, vy, 'g')], 'txt': (0.75, 0.75, 'area: ' + '%.3f' % sa)})

    # graph everyone into overview image
    gl = len(grid) / 2 + 1 if not len(grid) % 2 else len(grid) / 2 + 2
    gl = int(gl + 1)

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 20}

    plt.rc('font', **font)

    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=.5, wspace=.05)

    # axs = axs.ravel()
    ax1 = plt.subplot2grid((gl, 2), (0, 0), colspan=2, rowspan=2)

    overview_text = react_text
    add_plot(ax1, {'axis': [0, x[max_x], 0, np.max(ny)], 'plots': [[x, y, 'b'], [bx, by, 'g'], [bx, ny - vy, 'r']],
                   'txt': (0.35, 0.7, overview_text)})

    for i in range(0, len(grid), 2):
        row = int((i + 1) / 2 + 2)
        # print 'R', row
        if i + 1 < len(grid):
            add_plot(plt.subplot2grid((gl, 2), (row, 0), colspan=1), grid[i])
            add_plot(plt.subplot2grid((gl, 2), (row, 1), colspan=1), grid[i + 1])
        else:
            add_plot(plt.subplot2grid((gl, 2), (row, 0), colspan=2), grid[i])

    fig.tight_layout()
    if show_file:
        plt.show()
    else:
        fig.savefig(tmp_prefix + '_overview.png')
    plt.close()
    plt.clf()

    with open(tmp_prefix + '_txt.csv', 'w') as f:
        csvlog = csv.writer(f)
        csvout.append(['', '', ''])
        csvout.append(['', '', ''])
        csvout.append(['Date:', date_stamp, ''])
        csvout.append(['Time:', time_stamp, ''])
        for i in csvout:
            csvlog.writerow(i)

    return data_dict
