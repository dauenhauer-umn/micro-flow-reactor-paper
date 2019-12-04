import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import signal
from matplotlib.ticker import FormatStrFormatter
import peakutils


def find_area(startx, endx, real_y, blank_x, trapazoid_area=None):
    """
    Find area under curve using trapeszoidal rule.

    :param startx: starting index
    :param endx: ending index
    :param real_y: array of y values
    :param blank_x: array of blank_x values
    :param ly: custom trapazoid area to subtract from composite trapezoidal rule.
    :return: area beneath the curve
    """

    if not trapazoid_area:
        # area of trapazoid below the curve
        trapazoid_area = ((real_y[startx] + real_y[endx]) / 2.0) * (blank_x[endx] - blank_x[startx])

    # calculate the area, subtract the trapazoid area
    area = np.trapz(real_y[startx:endx + 1], blank_x[startx:endx + 1]) - trapazoid_area

    return area


def peak_detect(y_val, xran=(3, 10), min_dist=300, max_peak=10, thres=None):
    """

    :param y_val: array of y values to evaluate for peaks
    :param xran: range of thresholds to iterate through
    :param min_dist: minimum distance between peaks
    :param max_peak: maximum number of peaks to detect
    :param thres: manually set threshold
    :return: indices of the peaks
    """

    indices = None
    if not thres:
        # iterate through the range of thresholds until number of peaks is <= maximum number of peaks
        for i in np.arange(*xran):
            indices = peakutils.indexes(y_val, thres=float(i) / 100, min_dist=min_dist)
            if len(indices) <= max_peak:
                # print 'THRESHOLD', i
                break
    else:
        indices = peakutils.indexes(y_val, thres=thres, min_dist=min_dist)
    return indices


def add_plot(ax, i):
    """
    Convenience function for generating an overview image

    :param ax: the pyplot.ax object
    :param i: the dictionary of attributes for the plot {'axis','plots','txt','patch'}
    """
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if 'axis' in i:
        ax.axis(i['axis'])
    for a in i['plots']:
        ax.plot(a[0], a[1], a[2])
    if 'txt' in i:
        anno_opts = dict(xy=(i['txt'][0], i['txt'][1]), xycoords='axes fraction',
                         va='center', ha='center', fontsize=20)
        ax.annotate(i['txt'][2], **anno_opts)
    if 'patch' in i:
        ax.add_patch(i['patch'])


def normalize_baseline(blank_x, blank_y, data_x, data_y, **kwargs):
    """
    1.  Normalize baseline by smoothing out data & blank run y values
    2.  subtract them
    3.  inverting result
    4.  finding sparse peaks in that results
    5.  removing interpolating between those peaks.

    :param blank_x: blank y values
    :param blank_y: blank blank_x values
    :param data_x: data y values
    :param data_y: data blank_x values
    :param kwargs: settings dictionary
    :return: (normalized blank blank_x values, normalized blank y values)
    """

    # make sure the length of the areas is equal
    # blank runs typically are longer
    x_length = len(data_y) if len(blank_y) > len(data_y) else len(blank_x)

    blank_y_win_length = 333 if 'blank_y_win_length' not in kwargs else kwargs['blank_y_win_length']
    norm_thres_range = (20, 30) if 'norm_thres_range' not in kwargs else kwargs['norm_thres_range']
    norm_min_dist = 3000 if 'norm_min_dist' not in kwargs else kwargs['norm_min_dist']
    norm_polyorder = 5 if 'norm_polyorder' not in kwargs else kwargs['norm_polyorder']

    # smooth out data y values
    smooth_real_y = np.array(
        signal.savgol_filter(data_y[:x_length], window_length=blank_y_win_length, polyorder=norm_polyorder))
    # smooth out blank y values
    smooth_blank_y = np.array(
        signal.savgol_filter(blank_y[:x_length], window_length=blank_y_win_length, polyorder=norm_polyorder))

    # smooth out both together, after subtracting smooth blank from smooth y
    smooth_combo = np.array(signal.savgol_filter(smooth_real_y - smooth_blank_y, window_length=blank_y_win_length,
                                                 polyorder=norm_polyorder))

    # invert the smoothed array and find peaks
    inv = peak_detect(-1.0 * (smooth_combo), xran=norm_thres_range, min_dist=norm_min_dist)

    # exclude any peaks that are "higher" than the last data point of non inverted array
    inv = [i for i in inv if data_y[i] < data_y[-1]] + [len(data_y) - 1]

    # create a straight line from start to end data values
    norm_blank_x = np.linspace(0, data_x[-1], len(data_x))
    norm_blank_y = np.linspace(0, data_y[-1], len(data_y))

    cur_idx = 0
    for idx, p in enumerate(inv):
        #  match the slope between the beginning and end of the inverted peaks
        norm_blank_y[cur_idx:p + 1] = np.linspace(data_y[cur_idx], data_y[p], p - cur_idx + 1)
        cur_idx = p

    return norm_blank_x, norm_blank_y


def walk_peaks(indices, real_y, blank_y, blank_x, max_x, **kwargs):
    """
     Walk backwards and forwards from the peak index

    :param indices: (list) List of peak indices from peak_detect
    :param real_y (array): real y values
    :param blank_y (array): blank y values
    :param blank_x (array): blank x values
    :param max_x (int): max index for arrays
    :param kwargs: settings dictionary
    :return: 
    """

    data_y_win_length = 111 if 'data_y_win_length' not in kwargs else kwargs['data_y_win_length']
    savgol_polyorder = 7 if 'savgol_polyorder' not in kwargs else kwargs['savgol_polyorder']
    minimum_area = 0.1 if 'minimum_area' not in kwargs else kwargs['minimum_area']
    # min_dist_between_peaks = 20

    # minimum distance between peaks
    min_dist_between_peaks = 0 if 'min_dist_between_peaks' not in kwargs else kwargs['min_dist_between_peaks']

    # minimum slope of blank line
    min_blank_slope = 0 if 'min_blank_slope' not in kwargs else kwargs['min_blank_slope']

    # step interval when walking the y values
    xinterval = 10 if 'xinterval' not in kwargs else kwargs['xinterval']

    # add double peaks
    min_walking_vert = False if 'min_walking_vert' not in kwargs else kwargs['min_walking_vert']
    vert_distance = False

    areas = []

    # smooth the real y values so slope comparison is not affected by noise as much
    smooth_walking_y = np.array(
        signal.savgol_filter(real_y, window_length=data_y_win_length, polyorder=savgol_polyorder))

    for i in indices:

        signal_slope = 1
        blank_slope = 0

        # the vert_distance flag will keep steping back if the vertical difference
        # between baseline and trough between peaks is greater than min_walking_vert
        if min_walking_vert:
            vert_distance = True

        # pi is "past index". Keep stepping back while (slope of real_y) > (slope of blank polyfit + min_slope
        pi = i

        while (signal_slope > blank_slope + min_blank_slope or vert_distance) and pi > 0:
            vert_distance = smooth_walking_y[pi] - blank_y[pi] > min_walking_vert
            signal_slope = smooth_walking_y[pi] - smooth_walking_y[pi - xinterval]
            blank_slope = blank_y[pi] - blank_y[pi - xinterval]
            pi -= xinterval

        # fi is "future index". Keep stepping back while (slope of real_y) > (slope of blank polyfit + min_slope)
        # OR keep stepping forward if the space between blank y line and real y line is greater than the avg difference
        # between them

        fi = i

        signal_slope = 1
        blank_slope = 0

        while fi + xinterval < max_x and (signal_slope > blank_slope + min_blank_slope or vert_distance):
            vert_distance = smooth_walking_y[fi] - blank_y[fi] > min_walking_vert
            signal_slope = smooth_walking_y[fi] - smooth_walking_y[fi + xinterval]
            blank_slope = blank_y[fi] - blank_y[fi + xinterval]
            fi += xinterval

        # some sanity checks for index
        if pi > 0 and pi < max_x and fi < max_x:

            # this short test checks if the slope between the beginning index and ending index
            # crosses through the data y values. If it does, then change the index to where the
            # slope intercepts
            test_slope = (real_y[fi] - real_y[pi]) / (blank_x[fi] - blank_x[pi])
            ppi = i
            while ppi > pi and real_y[ppi] > test_slope * blank_x[ppi - pi] + real_y[pi]:
                ppi -= 1
            pi = ppi
            ffi = i
            while ffi < fi and real_y[ffi] > test_slope * blank_x[ffi - pi] + real_y[pi]:
                ffi += 1
            fi = ffi

            # if we already have some areas, we need to compare min distance and other sanity checks
            if len(areas):

                # check to make sure the distance between the current past index and future index are greater than the
                # most recent peak.

                prev_start = areas[-1][0]['x']
                prev_end = areas[-1][1]['x']


                if pi - prev_start > min_dist_between_peaks and prev_end > min_dist_between_peaks and pi > prev_end:
                    # make sure the area is greater than minimum area threshold
                    area = find_area(pi, fi, real_y, blank_x)
                    if area > minimum_area:
                        areas.append(({'x': pi, 'multi': False, 'sly': None}, {'x': fi, 'multi': False, 'sly': None}))

            # first peak is free
            elif not len(areas):
                area = find_area(pi, fi, real_y, blank_x)
                # if the area is greater than the minimum add the new peak.
                if area > minimum_area and fi - pi > min_dist_between_peaks:
                    areas.append(({'x': pi, 'multi': False, 'sly': None}, {'x': fi, 'multi': False, 'sly': None}))
    return areas


def splice_peaks(fpks, base_y, real_y, blank_x, peak_range=None, **kwargs):
    """
    With general peak areas sorted, now split up those areas if there are multiple peaks.
    This function splits them at the trough between two peaks

    :param fpks:  list of tuples peaks [({'blank_x': 3695, 'sly': None, 'multi': False}, {'blank_x': 4833, 'sly': None, 'multi': False}),]
    :param base_y: real y values - blank y values
    :param real_y: real y values
    :param blank_x: blank x values
    :param peak_range:  range of thresholds for peak_detect()
    :param kwargs:  special peak_detect settings for a particular range of y values
    :return: list of tuples peaks
    """

    split_min_dist = 100 if 'split_min_dist' not in kwargs else kwargs['split_min_dist']
    split_max_peak = 3 if 'split_max_peak' not in kwargs else kwargs['split_max_peak']
    split_thres_range = (1, 5, .1) if 'split_thres_range' not in kwargs else kwargs['split_thres_range']
    peak_thres_range = (1, 5, .1) if 'peak_thres_range' not in kwargs else kwargs['peak_thres_range']

    peak_splice = []
    split_min_area = .005 if 'split_min_area' not in kwargs else kwargs['split_min_area']
    last_split = 0
    split_win_length = 31 if 'split_win_length' not in kwargs else kwargs['split_win_length']
    split_polyorder = 5 if 'split_polyorder' not in kwargs else kwargs['split_polyorder']

    for idx, i in enumerate(fpks):
        start = i[0]['x']
        end = i[1]['x']

        y_vals = base_y[start:end]

        slope = (real_y[end] - real_y[start]) / (blank_x[end] - blank_x[start])

        new_peaks = []

        # smooth the current section for peak detection
        smoothed_section = signal.savgol_filter(y_vals, window_length=split_win_length, polyorder=split_polyorder)

        # {'range': (11,13), 'thres': .008'}
        fidx = peak_detect(smoothed_section, min_dist=split_min_dist, xran=peak_thres_range, max_peak=split_max_peak)


        if peak_range:
            for p in peak_range:
                if blank_x[start] > p['range'][0] and blank_x[end] < p['range'][1]:

                    fidx = peak_detect(smoothed_section, min_dist=split_min_dist, max_peak=split_max_peak,
                                       xran=split_thres_range)

        if len(fidx) > 1:

            for pidx, spliced_peak_index in enumerate(fidx):

                last_newpeak = start if not len(new_peaks) else new_peaks[-1][1]['x']

                if pidx == len(fidx) - 1 and len(new_peaks):

                    sly = (slope * (blank_x[end] - blank_x[start])) + real_y[start]

                    trapazoid_area = (real_y[end] + sly) / 2.0 * (blank_x[end] - blank_x[last_newpeak])

                    peak_area = find_area(last_newpeak, end, real_y, blank_x, trapazoid_area)

                    if peak_area > split_min_area:
                        new_peaks.append((new_peaks[-1][1], {'x': end, 'multi': True, 'sly': sly}))
                    else:
                        new_peaks[-1][1]['x'] = end

                elif pidx + 1 < len(fidx):

                    if len(new_peaks):
                        bidx = np.argmin(real_y[last_split + start:start + spliced_peak_index + 1]) + start
                    else:
                        bidx = np.argmin(
                            real_y[start + spliced_peak_index:start + fidx[pidx + 1] + 1]) + start + spliced_peak_index

                    sly = (slope * (blank_x[bidx] - blank_x[start])) + real_y[start]

                    if bidx > last_newpeak and find_area(last_newpeak, bidx, real_y, blank_x) > split_min_area:
                        last_split = spliced_peak_index
                        new_peaks.append(
                            ({'x': last_newpeak, 'multi': True, 'sly': sly}, {'x': bidx, 'multi': True, 'sly': sly}))


            if len(new_peaks) > 1:
                peak_splice.append({'idx': idx, 'i': i, 'new': new_peaks})

    return peak_splice



def sep_peak(blank_x, real_y, base_y, idx, peak_element, fpks, **kwargs):
    """

    If in a special range of split peaks, then take the first and last split peaks of area,
    then shave last peak area from first, and first area from the second.
    Only return the 'bump' of the last peak from the slope of the first peak

    :param blank_x: blank_x values
    :param real_y: eal y values
    :param base_y: real y values minus blank y values
    :param idx: index of list of tuples peaks
    :param peak_element:  element of list of tuples peaks
    :param fpks:  list of tuples peaks [({'x': 3695, 'sly': None, 'multi': False}, {'x': 4833, 'sly': None, 'multi': False}),]
    :param kwargs:  special peak_detect settings for a particular range of y values
    :return: list of tuples peaks
    """

    split_min_dist = 100 if 'split_min_dist' not in kwargs else kwargs['split_min_dist']
    split_max_peak = 3 if 'split_max_peak' not in kwargs else kwargs['split_max_peak']
    split_thres_range = (3, 10, 1) if 'split_thres_range' not in kwargs else kwargs['split_thres_range']
    split_win_length = 100 if 'split_win_length' not in kwargs else kwargs['split_win_length']
    split_polyorder = 3 if 'split_polyorder' not in kwargs else kwargs['split_polyorder']



    if peak_element[0]['multi']:
        startx = peak_element[0]['x']
        endx = peak_element[1]['x']
        second_peak = False

        if idx > 0:

            # If we are on first peak or later peaks
            if startx == fpks[idx - 1][1]['x']:
                # second split peak
                # change endx to final blank_x of split peak
                second_peak = True
                split_x = startx
                startx = fpks[idx - 1][0]['x']
            else:
                split_x = endx
                endx = fpks[idx + 1][1]['x']
                fpks[idx][0]['area'] = find_area(startx, endx, real_y, blank_x)

            smoothed_section = signal.savgol_filter(base_y[startx:endx], window_length=split_win_length,
                                                    polyorder=split_polyorder)

            # detect peaks in the range
            cv = peak_detect(smoothed_section, split_thres_range, split_min_dist, split_max_peak)

            if len(cv) == 2:

                mid_x = (cv[1] + startx - split_x) * 3 + split_x



                ly = (real_y[mid_x] - real_y[split_x]) / (blank_x[mid_x] - blank_x[split_x])

                for widx, w in enumerate(real_y[cv[1] + startx:]):
                    if w < (blank_x[cv[1] + startx + widx] - blank_x[split_x]) * ly + real_y[split_x]:
                        mid_x = cv[1] + startx + widx
                        break

                if second_peak:
                    # second split peak
                    # change endx to final blank_x of split peak
                    startx = split_x
                    endx = mid_x

                else:
                    mid_values = [(blank_x[v] - blank_x[split_x]) * ly + real_y[split_x] for v in range(split_x, mid_x)]
                    real_y = list(real_y[:split_x]) + mid_values + list(real_y[mid_x:])
                    real_y = np.array(real_y)

                ix = blank_x[startx:endx]
                iy = real_y[startx:endx]
                start_vert = real_y[startx]
                end_vert = real_y[endx]

            else:
                return False



            verts = [(blank_x[startx], start_vert)] + list(zip(ix, iy)) + [(blank_x[endx], end_vert)]
            poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
            ly = (real_y[startx] + real_y[endx]) / 2.0 * (blank_x[endx] - blank_x[startx])

            area = find_area(startx, endx, real_y, blank_x, ly)


            cout = [area, blank_x[startx], blank_x[endx]]

            axis_x = (blank_x[startx - 150], blank_x[min(endx + 150, blank_x.size - 1)])
            axis_y = (np.min(real_y[startx:endx]) - 0.4, np.max(real_y[startx:endx]) + .4)
            return [poly, area, cout, axis_x, axis_y]

        else:
            return False

    return False