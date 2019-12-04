import numpy as np

from scipy import signal

from chromauto.utils import walk_peaks, peak_detect, find_area, add_plot, splice_peaks
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import csv


def Isoprene_Integrate(real_x, real_y, blank_x, blank_y, tmp_prefix, **kwargs):
    blank_y_win_length = 333 if 'blank_y_win_length' not in kwargs else kwargs['blank_y_win_length']
    peak_min_dist = 20 if 'peak_min_dist' not in kwargs else kwargs['peak_min_dist']
    peak_max_peak = 3 if 'peak_max_peak' not in kwargs else kwargs['peak_max_peak']
    peak_thres_range = (1, 5, .1) if 'peak_thres_range' not in kwargs else kwargs['peak_thres_range']
    data_y_win_length = 111 if 'data_y_win_length' not in kwargs else kwargs['data_y_win_length']
    savgol_polyorder = 7 if 'savgol_polyorder' not in kwargs else kwargs['savgol_polyorder']

    show_file = False if 'show_file' not in kwargs else kwargs['show_file']

    data_dict = {}
    grid = []

    max_x = blank_x.size - 1

    comps = {'reactant': {'name': '2MTHF', 'duration': (13.58, 13.58)},
             'interest': {'name': '13PD',
                          'duration': (
                              {'points': [10.00, 10.15], 'seen': False}, {'points': [10.28, ], 'seen': False})},
             'catalyst': {'sial', 1}}
    # find peaks in smooth_real_y-smooth_blank_y (smoothed red line). Threshold steps from .03 to .1. Step until there are less than 10 peaks

    smooth_real_y = np.array(signal.savgol_filter(real_y, window_length=data_y_win_length, polyorder=savgol_polyorder))
    smooth_blank_y = np.array(
        signal.savgol_filter(blank_y, window_length=blank_y_win_length, polyorder=savgol_polyorder))
    base_y = smooth_real_y - smooth_blank_y

    indices = peak_detect(smooth_real_y - smooth_blank_y, xran=peak_thres_range, min_dist=peak_min_dist,
                          max_peak=peak_max_peak)

    fpks = walk_peaks(indices, smooth_real_y, blank_y, real_x, max_x, **kwargs)

    peak_splice = splice_peaks(fpks, base_y, smooth_real_y, real_x, **kwargs)

    # remove old peak, insert new peaks
    for idx, i in enumerate(peak_splice):
        oldix = fpks.index(i['i'])
        fpks.remove(i['i'])
        for pidx, p in enumerate(i['new']):
            fpks.insert(pidx + oldix, p)

    all_areas = 0.0
    react_area = 0.0
    interest_area = 0.0
    react_text = ""
    interest_text = ""
    csvout = [['area', 'start time', 'end time', ]]
    # graph the first round of peak detection
    for idx, i in enumerate(fpks):
        startx = i[0]['x']
        endx = i[1]['x']
        a = real_x[startx]
        b = real_x[endx]
        ix = real_x[startx:endx]
        iy = smooth_real_y[startx:endx]
        start_vert = smooth_real_y[startx]
        end_vert = smooth_real_y[endx]

        # area below the curve (xaxis to yline)
        # trapizod area calculation
        trapezoid_area = ((smooth_real_y[startx] + smooth_real_y[endx]) / 2.0) * (real_x[endx] - real_x[startx])
        if i[0]['multi']:
            if idx + 1 < len(fpks):
                # if a split peak, calculate the slope and recalc verts and area below curve
                # split peaks share next start or prev end
                if endx == fpks[idx + 1][0]['x']:
                    # multi end
                    sly = i[1]['sly']
                    end_vert = sly
                    ly = (smooth_real_y[startx] + sly) / 2.0 * (real_x[endx] - real_x[startx])
            if idx > 0:
                if startx == fpks[idx - 1][1]['x']:
                    # multi start
                    sly = i[0]['sly']
                    start_vert = sly
                    ly = (smooth_real_y[endx] + sly) / 2.0 * (real_x[endx] - real_x[startx])

        verts = [(a, start_vert)] + list(zip(ix, iy)) + [(b, end_vert)]

        poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')

        sa = find_area(startx, endx, smooth_real_y, real_x, trapezoid_area)
        csvout.append(['%.3f' % sa, '%.3f' % a, '%.3f' % b])
        if real_x[startx] < comps['reactant']['duration'][0] < real_x[endx]:
            react_text += '\nReactant start: ' + '%.2f' % real_x[startx]
            if comps['reactant']['duration'][0] == comps['reactant']['duration'][1]:
                react_text += '  end: ' + '%.2f' % real_x[endx]
            react_area += sa
        elif real_x[startx] < comps['reactant']['duration'][1] < real_x[endx]:
            react_text += '  end: ' + '%.2f' % real_x[endx]
            react_area += sa

        if not comps['interest']['duration'][0]['seen'] and [area_point for area_point in
                                                             comps['interest']['duration'][0]['points'] if
                                                             real_x[startx] < area_point < real_x[endx]]:
            interest_text += '\nSelectivity start: ' + '%.2f' % real_x[startx]
            if comps['interest']['duration'][0] == comps['interest']['duration'][1]:
                react_text += '  end: ' + '%.2f' % real_x[endx]
            interest_area += sa
            comps['interest']['duration'][0]['seen'] = True
        elif [area_point for area_point in
              comps['interest']['duration'][1]['points'] if real_x[startx] < area_point < real_x[endx]]:
            interest_text += '  end: ' + '%.2f' % real_x[endx]
            interest_area += sa

        all_areas += sa

        grid.append(
            {'patch': poly, 'axis': [real_x[startx - 150], real_x[min(endx + 150, max_x)], 0.9,
                                     np.max(smooth_real_y[startx:endx]) + .4],
             'plots': [(real_x, real_y, 'b'), (real_x, smooth_blank_y, 'g')],
             'txt': (0.75, 0.75, 'Area: ' + '%.3f' % sa)})

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

    react_text += '\nReact Area: ' + '%.3f' % react_area + '\nConversion % (react area): ' + '%.3f' % (
            (1 - (react_area / all_areas)) * 100) if react_area else ''
    interest_text += '\nSelectivity Area: ' + '%.3f' % interest_area + '\nSelectivity %: ' + '%.3f' % (
            (interest_area / (all_areas - react_area)) * 100) if interest_area else ''
    overview_text = 'Green: blank Red: signal-blank' + react_text + interest_text

    add_plot(ax1, {'axis': [0, real_x[max_x], 0, np.max(smooth_real_y)],
                   'plots': [[real_x, real_y, 'b'], [blank_x, blank_y, 'g'],
                             [blank_x, smooth_real_y - smooth_blank_y, 'r']],
                   'txt': (0.75, 0.75, overview_text)})

    if react_area:
        data_dict['Conversion'] = float('%.3f' % ((1 - react_area / all_areas) * 100))
    if interest_area:
        data_dict['Selectivity percentage'] = float('%.3f' % ((interest_area / all_areas) * 100))

    for i in range(0, len(grid), 2):
        row = int((i + 1) / 2 + 2)
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
        for i in csvout:
            csvlog.writerow(i)

    return data_dict
