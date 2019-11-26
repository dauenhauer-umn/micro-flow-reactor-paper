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
from scanner.models import handler_result
from scanner.parsers import add_extension
from scanner.parsers.type_ch.chemicals import peak_detect, find_area, add_plot
from scanner.parsers.type_ch.ch_parser import CHFile, MSFile
from django.utils.text import slugify
from projmon.settings import BASE_DIR
import csv
import re


def avg_difference(real_y, blank_y):
    """
    Find average difference between real y values and blank y values

    Flip real_y, find peaks, use those peaks for approximation of linear real y values

    Parameters:
    real_y: real y values
    blank_y: blank y values
    """

    # Find inverted peaks, filter them by last value of run
    # inv = peak_detect(-1.0*(real_y-blank_y), (0,3)) ,min_dist=1000
    inv = peak_detect(-1.0 * (real_y - blank_y), thres=.9, min_dist=1000)
    inv = np.array([i for i in inv if real_y[i] < real_y[-1]])

    avg_base = np.average([real_y[i] for i in inv]) - np.average(blank_y)

    return avg_base


def walk_peaks(indexes, base_y, ny, by, x, max_x):
    """
    Walk backwards and forwards from the peak index

    Parameters:
        indexes (list): List of peak indexes from peak_detect
        base_y (array): real y - blank y values
        ny: real y values
        by: blank y values
        x: x values
        max_x: max index for arrays

    """

    fpks = []

    # avg_base is the minimum distance between ny and blank polyfit. So, if ny = 4.9 and greenpolyfit = 4, then
    # .9 < 1, so it stops stepping backwards (or forwards) when finding area around peak.
    # avg_base = np.average(ny - vy)
    avg_base = avg_difference(ny, by) * 1.25

    # avg_base = 1.25
    # print("AVG", avg_base)
    minimum_area = 0.03
    # min_dist_between_peaks = 20

    # minimum distance between peaks
    min_dist_between_peaks = 0

    # minimum slope of blank line
    min_blank_slope = 0

    # step interval when walking the y values
    xinterval = 10

    for i in indexes:
        # pi is "past index". Keep stepping back while (slope of ny) > (slope of blank polyfit + .005) .005 is the minimum slope
        # also check to see if avg_base is exceeded.
        pi = i

        # pi > xinterval and (ny[pi] - ny[pi - xinterval] > pp(x[pi]) - pp(x[pi - xinterval]) + .005 or ny[pi] - pp(
        #        x[pi]) > avg_base):

        signal_slope = 1
        blank_slope = 0

        while signal_slope > blank_slope and pi > 0:
            signal_slope = ny[pi] - ny[pi - xinterval]
            blank_slope = by[pi] - by[pi - xinterval]
            pi -= xinterval

        # fi is "future index". Keep stepping back while (slope of ny) > (slope of blank polyfit + .005)
        # .005 is the minimum slope
        # OR keep stepping forward if the space between blank y line and real y line is greater than the avg difference
        # between them
        fi = i

        signal_slope = 1
        blank_slope = 0

        while fi + xinterval < max_x and (signal_slope > blank_slope or ny[fi] - by[fi] > avg_base):
            # or ny[fi] - pp(x[fi]) > avg_base
            # ):
            signal_slope = ny[fi] - ny[fi + xinterval]
            blank_slope = by[fi] - by[fi + xinterval]
            fi += xinterval

        if pi > 0 and pi < max_x and fi < max_x:

            # if we already have some areas, we need to compare min distance and other sanity checks

            if len(fpks):
                # check to make sure the distance between the current past index and future index are greater than the
                # most recent peak. They should have a min. distance of 20

                sa = find_area(pi, fi, ny, x)

                prev_start = fpks[-1][0]['x']
                prev_end = fpks[-1][1]['x']
                if pi - prev_start > min_dist_between_peaks and prev_end > min_dist_between_peaks and pi > prev_end:
                    if sa > minimum_area:
                        # print('check:', pi - fpks[-1][0]['x'], fi - fpks[-1][1]['x'])
                        fpks.append(({'x': pi, 'multi': False, 'sly': None}, {'x': fi, 'multi': False, 'sly': None}))


            # first peak is free
            elif not len(fpks):
                sa = find_area(pi, fi, ny, x)
                # if the area is greater than the minimum add the new peak.
                if sa > minimum_area and fi - pi > min_dist_between_peaks:
                    fpks.append(({'x': pi, 'multi': False, 'sly': None}, {'x': fi, 'multi': False, 'sly': None}))
    return fpks


def splice_peaks(fpks, base_y, ny, x, xran=(3, 10, .001), max_peak=4, peak_range=None):
    """
    With general peak areas sorted, now split up those areas if there are multiple peaks.

    Parameters:
        fpks: list of tuples peaks [({'x': 3695, 'sly': None, 'multi': False}, {'x': 4833, 'sly': None, 'multi': False}),]
        base_y: real y values minus blank y values
        ny: real y values
        x: x values
        xran:
        xran: range of thresholds for peak_detect()
        min_dist: minimum distance between peaks for peak_detect()
        peak_range: special peak_detect settings for a particular range of y values

    """
    peak_splice = []
    minimum_area = .005
    last_split = 0

    for idx, i in enumerate(fpks):
        start = i[0]['x']
        end = i[1]['x']

        # y_vals = base_y[start:end+1]
        y_vals = base_y[start:end]

        slope = (ny[end] - ny[start]) / (x[end] - x[start])
        # print('SL', slope)
        new_peaks = []

        # smooth the current section for peak detection
        smoothed_section = np.array(signal.savgol_filter(y_vals, window_length=31, polyorder=5))

        # (11, 13): (.8,2)
        # {'range': (11,13), 'thres': .008'}
        fidx = peak_detect(smoothed_section, min_dist=20, xran=xran, max_peak=max_peak)

        if peak_range:
            for p in peak_range:
                if x[start] > p['range'][0] and x[end] < p['range'][1]:
                    minimum_area = p['area']
                    fidx = peak_detect(smoothed_section, min_dist=30, max_peak=2, xran=(.5, 10, .001))
                    # print('min', fidx)
                    # print('peak_range', x[start], p['range'][0], x[end], p['range'][1], fidx)

        # print('FIDX', start, end, x[start], x[end], fidx)
        # print('PEAKS', [x[start+i] for i in fidx])

        if len(fidx) > 1:

            for pidx, spliced_peak_index in enumerate(fidx):

                last_newpeak = start if not len(new_peaks) else new_peaks[-1][1]['x']

                # print(pidx == len(fidx) - 1 and len(new_peaks))
                # print(pidx + 1 < len(fidx))
                if pidx == len(fidx) - 1 and len(new_peaks):

                    sly = (slope * (x[end] - x[start])) + ny[start]

                    ly = (ny[end] + sly) / 2.0 * (x[end] - x[last_newpeak])

                    peak_area = find_area(last_newpeak, end, ny, x, ly)

                    # print('peakarea LAST', x[spliced_peak_index+start], peak_area, sly, start, end, new_peaks[-1][1])
                    if peak_area > minimum_area:
                        new_peaks.append((new_peaks[-1][1], {'x': end, 'multi': True, 'sly': sly}))
                    else:
                        new_peaks[-1][1]['x'] = end

                elif pidx + 1 < len(fidx):

                    if len(new_peaks):
                        # print('last_split', last_split, spliced_peak_index)
                        bidx = np.argmin(ny[last_split + start:start + spliced_peak_index + 1]) + start
                    else:
                        bidx = np.argmin(
                            ny[start + spliced_peak_index:start + fidx[pidx + 1] + 1]) + start + spliced_peak_index

                    sly = (slope * (x[bidx] - x[start])) + ny[start]
                    ly = (ny[bidx] + sly) / 2.0 * (x[bidx] - x[last_newpeak])
                    # need custom ly here probably
                    # print('a', find_area(last_newpeak, bidx, ny, x), 'peakx', x[spliced_peak_index+start], 'min i', bidx,'min x', x[bidx], sly, ly, start, end)

                    # print('BIDX',[x[i+start] for i in fidx], new_peaks, bidx, x[bidx], last_split+start,start + spliced_peak_index +1, start + spliced_peak_index,start + fidx[pidx + 1] +1)
                    if bidx > last_newpeak and find_area(last_newpeak, bidx, ny, x) > minimum_area:
                        # print('NEW PEAKSSS', {'x': last_newpeak, 'multi': True, 'sly': sly}, {'x': bidx, 'multi': True, 'sly': sly})
                        last_split = spliced_peak_index
                        new_peaks.append(
                            ({'x': last_newpeak, 'multi': True, 'sly': sly}, {'x': bidx, 'multi': True, 'sly': sly}))

            # bidx = np.argmin(ny[i[0]+fidx[0]:i[0]+fidx[1]]) + i[0] + fidx[0]
            # print 'split_peak', i, fidx+i[0], x[fidx+i[0]], fidx
            # peak_splice.append({'idx': idx, 'i': i, 'new': [(start, bidx), (bidx,end)]})
            if len(new_peaks) > 1:
                peak_splice.append({'idx': idx, 'i': i, 'new': new_peaks})
                # print('split_peak', i)

    return peak_splice


# FPKS [({'x': 3667, 'multi': True, 'sly': 10.35741664765815},
# {'x': 4804, 'multi': True, 'sly': 10.35741664765815}),

# ({'x': 4804, 'multi': True, 'sly': 10.35741664765815},
# {'x': 7777, 'multi': True, 'sly': 10.448307291666666}),

# ({'x': 13725, 'multi': True, 'sly': 10.416220661919832},
# {'x': 14087, 'multi': True, 'sly': 10.416220661919832}),

# ({'x': 14087, 'multi': True, 'sly': 10.449277096518987},
# {'x': 14515, 'multi': True, 'sly': 10.449277096518987})]
def EO_TCD_Integrate(real_x, real_y, blank_x, blank_y, tmp_prefix, date_stamp, time_stamp, pt=.00003,):
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

    indexes = peak_detect(base_y, thres=pt)
    # print(indexes, [x[i] for i in indexes])

    fpks = walk_peaks(indexes, base_y, ny, by, x, max_x)

    spec_range = [{'range': (3, 5), 'thres': .05, 'area': .3}, ]
    # spec_range = []

    peak_splice = splice_peaks(fpks, base_y, ny, x, xran=(30, 60), max_peak=2, peak_range=spec_range)
    # peak_splice = []
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

    print('FPKS2', fpks)

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
                # sep_results = sep_peak(x, ny, base_y, idx, i, fpks)
                sep_results = []
                if sep_results:
                    # print('results', sep_results)
                    poly = sep_results[0]
                    sa = sep_results[1]
                    cout = sep_results[2]
                    axis_x = sep_results[3]
                    axis_y = sep_results[4]
                else:
                    print('no results')

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
    # plt.show(block=True)
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


eo_tcd_meta_data = {
    'name': 'EO TCD Method',
    'description': 'Gas chromatography EO method from Bhan group',
    'keys': ["Sample", "Date", "Time", ],
}


# keys are stored in kirkland source/computem/slicelib.hpp
def eo_tcd_handler(file_path, *args, **kwargs):
    data_dict = {}
    try:
        c = CHFile(file_path)
        if c.metadata['detector'] == 'back detector' and re.match(r'.*_EO_SIMS_.*', c.metadata['method']):
            b = CHFile(BASE_DIR + '/scanner/parsers/type_ch/eo_tcd_blank.ch')
            bx = np.array(b.times())
            by = np.array(b.values)
            prefix = '/'.join(file_path.split('/')[:-1]) + '/' + slugify(file_path.split('/')[-1]) + '_' + slugify(
                c.metadata['sample'])

            data_dict['Acquisition Date'] = c.metadata['datetime'].strftime("%m/%d/%Y") if c.metadata[
                'datetime'] else ''
            data_dict['Acquisition Time'] = c.metadata['datetime'].strftime("%I:%M:%S %p") if c.metadata[
                'datetime'] else ''

            data_dict.update(EO_TCD_Integrate(c.times(), c.values, bx[:len(c.times())], by[:len(c.times())], prefix, data_dict['Acquisition Date'], data_dict['Acquisition Time']))
            data_dict['Sample'] = c.metadata['sample']

            # SiAl_2MTHF_35sccm_0_7uL_min_44mg
        if data_dict:
            return handler_result(data=data_dict)
        else:
            return False
    except  Exception as e:
        raise


# http://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut2.html

def register():
    add_extension('ch', {'meta': eo_tcd_meta_data, 'handle': eo_tcd_handler})
