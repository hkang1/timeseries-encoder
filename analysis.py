import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm_notebook

import downsample
import plot

with open('config.json', 'r') as file:
    config = json.load(file)

def tabulate (x, y, fn):
    return np.vectorize(fn)(*np.meshgrid(x, y, sparse=True))

def cos_sum (a, b):
    return (math.cos(a + b))

def normalize (x, x_range, t_range=(-1., 1.)):
    x_clip = np.clip(x, x_range[0], x_range[1])
    return (t_range[1] - t_range[0]) * (x_clip - x_range[0]) / (x_range[1] - x_range[0]) + t_range[0]

def process_data (x, y):
    # normalize y
    y_clip = np.clip(y, config['y_range'][0], config['y_range'][1])
    y_norm = normalize(y_clip, config['y_range'])
    
    # normalize x
    x_max = np.amax(x)
    x_min = np.amin(x)
    x_range = (x_min, x_max)
    x_norm = normalize(x, x_range, (0., 1.))
    
    # polarize time series
    gaf, phi, r = get_gaf(x_norm, y_norm)
    
    # calculate penalty score
    penalty = get_penalty(x_norm, x_range, y_clip)
    
    return (x_norm, y_norm, gaf, phi, r, penalty)

def get_gaf (x_norm, y_norm):
    # polarize time series
    phi = np.arccos(y_norm)
    r = x_norm
    
    # calculate gaf
    gaf = tabulate(phi, phi, cos_sum)
    
    return (gaf, phi, r)

def get_penalty (x, x_range, y):
    days = (x_range[1] - x_range[0]) / (24 * 3600 * 1000)
    y_t = config['y_target']
    y_r0 = config['y_range'][0]
    y_r1 = config['y_range'][1]
    y_th0 = config['y_thresholds'][0]
    y_th1 = config['y_thresholds'][1]
    y_clip = np.clip(y, y_r0, y_r1)
    y_abs = np.where(y_clip > y_t, (y_clip - y_t) / (y_r1 - y_t), (y_t - y) / (y_t - y_r0)) * config['penalty_default']
    y_cold = np.where(y_th0 and y_clip < y_th0, (y_th0 - y_clip) / (y_th0 - y_r0), 0) * config['penalty_cold']
    y_warm = np.where(y_th1 and y_clip > y_th1, (y_clip - y_th1) / (y_r1 - y_th1), 0) * config['penalty_warm']
    
    penalty_default = np.trapz(y_abs, x)
    penalty_cold = np.trapz(y_cold, x)
    penalty_warm = np.trapz(y_warm, x)
    penalty_total = (penalty_default + penalty_cold + penalty_warm) / config['penalty_max']
    penalty_norm = normalize(penalty_total, (0., 1.), (1., 1024.))
    score = 10. - math.log(penalty_norm) / math.log(2)
    
    return {
        'penalty_default': penalty_default / config['penalty_default'],
        'penalty_cold': penalty_cold / config['penalty_cold'],
        'penalty_warm': penalty_warm / config['penalty_warm'],
        'penalty_total': penalty_total,
        'mkt_vvm2': get_mkt(x, y, 12109),
        'mkt_vvm7': get_mkt(x, y, 13154),
        'mkt_vvm14': get_mkt(x, y, 13607),
        'mkt_vvm30': get_mkt(x, y, 14508),
        'days': days,
        'score': score
    }

def get_intervals (values):
    intervals = []
    for i in range(len(values)):
        intervals.append(0. if i == 0 else values[i] - values[i-1])
    return intervals
    
# VVM2 - 12109
# VVM7 - 13154
# VVM14 - 13607
# VVM30 - 14508
def get_mkt (x, y, dhr=13154):
    y_t = config['y_target']
    y_r1 = config['y_range'][1]
    y_clip = np.clip(y, y_t, y_r1)
    x_i = get_intervals(x)
    y_k = y_clip + 273.5
    denominator = -np.log(np.sum(x_i * np.exp(-dhr / y_k)) / np.sum(x_i))
    return dhr / denominator - 273.15

def generate_graph (config):
    y_max = config['y_max']
    y_min = config['y_min']
    y_target = config['y_target']
    y_r = y_max - y_min
    y_r = config['y_max'] - config['y_min']
    x = np.sort(np.random.uniform(low=0., high=1., size=(config['size'] - 2,)))
    x = np.insert(x, 0, 0.)
    x = np.append(x, 1.)
    y = []
    for index, xi in enumerate(x):
        volatility = random.uniform(*config['volatility'])
        rnd = random.uniform(-1., 1.)
        y_prev = y_target if index == 0 else y0
        y0 = y_prev + y_r * volatility * rnd
        y0 = min(y0, y_max)
        y0 = max(y0, y_min)
        y.append(y0)
    y = np.array(y)
    return (x, y)

def generate_data (config, count=20):
    data = []
    for i in range(count):
        x, y = generate_graph(config)
        x_norm, y_norm, gaf, phi, r, penalty = process_data(x, y)
        data.append(format_info(x, x_norm, y, y_norm, gaf, phi, r, penalty))
    return data

def format_info (x, x_norm, y, y_norm, gaf, phi, r, penalty):
    return {
        "x": x,
        "x_norm": x_norm,
        "y": y,
        "y_norm": y_norm,
        "gaf": gaf,
        "phi": phi,
        "r": r,
        "penalty": penalty
    }

def process_company_analysis (company_id, analysis, data):
    added = 0

    for device_id, device_data in tqdm_notebook(data.items(), desc='device data analyzed'):
        # Check to see if it has already been process previously
        # and that it has the minimum number of data points
        if device_id in analysis or len(device_data) < config['size']:
            continue

        # Make sure the device data has at least 70% of the specified time range
        d_start = device_data.iloc[0]['time']
        d_stop = device_data.iloc[-1]['time']
        if d_stop - d_start < 0.7 * config['days'] * 24 * 3600 * 1000:
            continue

        # Downsample the data to size limit
        d = downsample.largest_triangle_dynamic(device_data.values, config['size'])

        # Extract x (time) and y (temperature) and process the data
        x = d[:,0]
        y = d[:,1]
        x_norm, y_norm, gaf, phi, r, penalty = process_data(x, y)

        # Store the analysis results
        analysis[device_id] = {
            'x': x,
            'x_norm': x_norm,
            'y': y,
            'y_norm': y_norm,
            'gaf': gaf,
            'phi': phi,
            'r': r,
            'penalty': penalty
        }

        # Mark that a new analysis has been added
        added += 1
    
    return analysis, added

def print_company_analysis (analysis):
    # report devices
    print('total devices: %d' % len(analysis))

    report = []
    total = {
        'penalty_default': 0,
        'penalty_cold': 0,
        'penalty_warm': 0,
        'penalty_total': 0,
        'mkt_vvm2': 0,
        'mkt_vvm7': 0,
        'mkt_vvm14': 0,
        'mkt_vvm30': 0,
        'days': 0,
        'score': 0
    }
    
    # convert analysis into list
    for device_id, device_analysis in analysis.items():
        for key in total:
            total[key] += device_analysis['penalty'][key]

        report.append([
            device_id,
            device_analysis['penalty']['penalty_default'],
            device_analysis['penalty']['penalty_warm'],
            device_analysis['penalty']['penalty_cold'],
            device_analysis['penalty']['mkt_vvm2'],
            device_analysis['penalty']['mkt_vvm7'],
            device_analysis['penalty']['mkt_vvm14'],
            device_analysis['penalty']['mkt_vvm30'],
            device_analysis['penalty']['days'],
            device_analysis['penalty']['score']
        ])

    # show score and penalty averages
    for key in total:
        print('average %s: %f' % (key, total[key] / len(analysis)))

    # sort by score
    report.sort(key=lambda x: x[-1], reverse=True)

    # convert to panda dataframe
    analysis_pd = pd.DataFrame(report, columns = ['device id' , 'default penalty', 'warm penalty', 'cold penalty', 'mkt_vvm2', 'mkt_vvm7', 'mkt_vvm14', 'mkt_vvm30', 'days', 'score'])
    
    # show list
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(analysis_pd)
    
    # show mkt plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    #plot.plot_decay(np.ones(len(analysis_pd)) * 2, analysis_pd['mkt_vvm2'], ax=ax)
    #plot.plot_decay(np.ones(len(analysis_pd)) * 7, analysis_pd['mkt_vvm7'], ax=ax)
    #plot.plot_decay(np.ones(len(analysis_pd)) * 14, analysis_pd['mkt_vvm14'], ax=ax)
    plot.plot_decay(np.ones(len(analysis_pd)) * 30, analysis_pd['mkt_vvm30'], ax=ax)
    
    return analysis_pd
