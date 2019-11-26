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
from utils import find_area, peak_detect, walk_peaks, add_plot, splice_peaks, sep_peak
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

    react_text = ''

    max_x = blank_x.size - 1

    base_y = real_y - blank_y

    indexes = peak_detect(base_y, thres=peak_hard_thres)
    

    fpks = walk_peaks(indexes, real_y, blank_y, real_x, max_x, **kwargs)

    peak_splice = splice_peaks(fpks, base_y, real_y, real_x, peak_range=spec_range, **kwargs)

    # remove old peak, insert new peaks

    for idx, i in enumerate(peak_splice):
        oldix = fpks.index(i['i'])
        fpks.remove(i['i'])
        for pidx, p in enumerate(i['new']):
            fpks.insert(pidx + oldix, p)

    all_areas = 0.0

    csvout = [['area', 'start time', 'end time', ]]

    for idx, i in enumerate(fpks):

        startx = i[0]['x']
        endx = i[1]['x']
        a = real_x[startx]
        b = real_x[endx]
        ix = real_x[startx:endx]
        iy = real_y[startx:endx]

        start_vert = real_y[startx]
        end_vert = real_y[endx]
        # area below the curve (xaxis to yline)
        # trapizod area calculation

        trapezoid_area = ((real_y[startx] + real_y[endx]) / 2.0) * (real_x[endx] - real_x[startx])

        if i[0]['multi']:
            if idx + 1 < len(fpks):
                # if a split peak, calculate the slope and recalc verts and area below curve
                # split peaks share next start or prev end
                if endx == fpks[idx + 1][0]['x']:
                    multi = 'end'
                    sly = i[1]['sly']
                    end_vert = sly
                    ly = (real_y[startx] + sly) / 2.0 * (real_x[endx] - real_x[startx])
            if idx > 0:
                if startx == fpks[idx - 1][1]['x']:
                    multi = 'start'
                    sly = i[0]['sly']
                    start_vert = sly
                    ly = (real_y[endx] + sly) / 2.0 * (real_x[endx] - real_x[startx])

        verts = [(a, start_vert)] + list(zip(ix, iy)) + [(b, end_vert)]
        poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        sa = find_area(startx, endx, real_y, real_x, ly)
        cout = [sa, a, b]
        axis_x = (real_x[startx - 150], real_x[min(endx + 150, max_x)])
        axis_y = (np.min(real_y[startx:endx]) - 0.4, np.max(real_y[startx:endx]) + .4)

        # hack for special area where split peak differently
        # spec_range = [{'range': (11,13), 'thres': .0003,'area':.0001},]
        for p in spec_range:
            if real_x[startx] > p['range'][0] and real_x[endx] < p['range'][1]:

                sep_results = sep_peak(real_x, real_y, base_y, idx, i, fpks, **kwargs)


                if sep_results:
                    # print('results', sep_results)
                    poly = sep_results[0]
                    sa = sep_results[1]
                    cout = sep_results[2]
                    axis_x = sep_results[3]
                    axis_y = sep_results[4]

        csvout.append(cout)

        all_areas += sa

        grid.append(
            {'patch': poly, 'axis': [axis_x[0], axis_x[1], axis_y[0], axis_y[1]],
             'plots': [(real_x, real_y, 'b'), (real_x, blank_y, 'g')], 'txt': (0.75, 0.75, 'area: ' + '%.3f' % sa)})

    # graph everyone into overview image
    gl = len(grid) / 2 + 1 if not len(grid) % 2 else len(grid) / 2 + 2
    gl = int(gl + 1)

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 20}

    plt.rc('font', **font)

    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=.5, wspace=.05)


    ax1 = plt.subplot2grid((gl, 2), (0, 0), colspan=2, rowspan=2)

    overview_text = react_text
    add_plot(ax1, {'axis': [0, real_x[max_x], 0, np.max(real_y)], 'plots': [[real_x, real_y, 'b'], [blank_x, blank_y, 'g'], [blank_x, real_y - blank_y, 'r']],
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
