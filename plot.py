import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pylab import rcParams

import analysis

with open('config.json', 'r') as file:
    config = json.load(file)

# Set the default figure size to be
rcParams['figure.figsize'] = (15, 7)
rcParams['font.size'] = 18

def plot_score (penalty):
    score = penalty['score']
    color_default = [ *config['color_default'][:-1], 0.2 ]
    color_score = config['color_score']
    colors = (color_score, color_default)
    
    fig = plt.figure(figsize=(15, 1))
    fig.suptitle('Total Score: %0.1f out of 10' % score, fontsize=16)
    
    ax = fig.add_subplot(111)
    ax.set_xlim(0., 10.)
    ax.set_ylim(0., 10.)
    ax.set_axisbelow(True)
    ax.grid(axis='x')
    ax.set_yticks([4])
    ax.set_yticklabels([''])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.broken_barh([ (0., score), (score, 10. - score) ], [1.5, 6], facecolors=colors)

def plot_penalty (penalty):
    pm = config['penalty_max']
    pd = penalty['penalty_default'] * 100. / pm
    pw = penalty['penalty_warm'] * 100. / pm
    pc = penalty['penalty_cold'] * 100. / pm
    pt = pd + pw + pc
    pdt = pd * 100. / pt
    pwt = pw * 100. / pt
    pct = pc * 100. / pt
    color_standard = config['color_standard']
    color_warm = config['color_warm']
    color_cold = config['color_cold']
    colors = (color_standard, color_warm, color_cold)
    
    fig = plt.figure(figsize=(15, 2))
    fig.suptitle('Penalty Breakdown', fontsize=16)

    ax = fig.add_subplot(111)
    ax.set_xlim(0., 100.)
    ax.set_ylim(0., 10.)
    ax.set_axisbelow(True)
    ax.grid(axis='x')
    ax.set_yticks([3])
    ax.set_yticklabels([''])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.broken_barh([ (0, pdt), (pdt, pwt), (pdt+pwt, pct) ], [1.5, 3], facecolors=colors)

    leg1 = mpatches.Patch(color=color_standard, label='%0.1f%% standard' % pdt)
    leg2 = mpatches.Patch(color=color_warm, label='%0.1f%% warm' % pwt)
    leg3 = mpatches.Patch(color=color_cold, label='%0.1f%% cold' % pct)
    ax.legend(handles=[leg1, leg2, leg3], ncol=3)

def plot_line (x, y, ax=None, area=True, alpha=1.0, title=None):
    y_t = config['y_target']
    y_r0 = config['y_range'][0]
    y_r1 = config['y_range'][1]
    y_th0 = config['y_thresholds'][0]
    y_th1 = config['y_thresholds'][1]
    
    if ax is None:
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(111)

    if title:
        ax.title.set_text(title)

    ax.set_xlabel('30 days represented by a range of 0 to 1')
    ax.set_ylabel('Celsius')
    ax.set_ylim(config['y_range'] or (np.amin(y), np.amax(y)))
    ax.plot([0., 1.], [y_t, y_t], dashes=(2, 3), c=config['color_default'])
    if area:
        ax.fill_between(x, y1=y, y2=y_t, color=config['color_standard'], alpha=0.2)
    
    if y_th0:
        y_cold = np.where(y < y_th0, y, y_th0)
        ax.plot([0., 1.], [y_th0, y_th0], dashes=(4, 4), c=config['color_cold'])
        if area:
            ax.fill_between(x, y1=y_cold, y2=y_th0, color=config['color_cold'], alpha=0.2)
    if y_th1:
        y_warm = np.where(y > y_th1, y, y_th1)
        ax.plot([0., 1.], [y_th1, y_th1], dashes=(4, 4), c=config['color_warm'])
        if area:
            ax.fill_between(x, y1=y_warm, y2=y_th1, color=config['color_warm'], alpha=0.2)
    
    ax.plot([0., 1.], [25, 25], dashes=(2, 6), c=config['color_warm'])
    ax.plot([0., 1.], [-0.5, -0.5], dashes=(2, 6), c=config['color_cold'])
    ax.plot([0., 1.], [-3, -3], dashes=(2, 6), c=config['color_cold'])
    ax.plot(x, y, c=config['color_default'], alpha=alpha)
    
    return ax

def plot_decay (y, mkt, ax=None):
    y_r0 = config['y_range'][0]
    y_r1 = config['y_range'][1]
    
    if ax is None:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(211)

    ax.set_xlim(0, 40)
    ax.set_ylim(1, 1000)
    ax.set_yscale('log')
    ax.set_xlabel('Celsius')
    ax.set_ylabel('days')

    ax.plot([0, 40], [400, 1.5], color='#1bffff')
    ax.plot([0, 40], [1150, 3.5], color='#34349a')
    ax.plot([0, 40], [4000, 8.0], color='#44ff48')
    ax.plot([0, 40], [9500, 10.9], color='#ff0000')
    ax.scatter(mkt, y, s=300, alpha=0.5, color='#ff0000')

def plot_polar (phi, r, ax=None):
    y_r = config['y_range']
    y_th0 = config['y_thresholds'][0]
    y_th1 = config['y_thresholds'][1]
    y_target_norm = analysis.normalize(config['y_target'], y_r)
    y_th0_norm = analysis.normalize(y_th0, y_r)
    y_th1_norm = analysis.normalize(y_th1, y_r)
    phi_target = np.arccos(y_target_norm)
    phi_th0 = np.arccos(y_th0_norm)
    phi_th1 = np.arccos(y_th1_norm)
    
    if ax is None:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.pi / 180. * np.linspace(180, 0, 4, endpoint=False))
    ax.fill_betweenx(r, np.pi - phi_target, np.pi - phi, color=config['color_standard'], alpha=0.2)
    ax.plot([np.pi - phi_target, np.pi - phi_target], [0, 1], dashes=(2, 3), c=config['color_default'])
    
    if y_th0:
        phi_cold = np.where(phi > phi_th0, phi, phi_th0)
        ax.fill_betweenx(r, np.pi - phi_th0, np.pi - phi_cold, color=config['color_cold'], alpha=0.2)
        ax.plot([np.pi - phi_th0, np.pi - phi_th0], [0, 1], dashes=(4, 4), c=config['color_cold'])
    if y_th1:
        phi_warm = np.where(phi < phi_th1, phi, phi_th1)
        ax.fill_betweenx(r, np.pi - phi_th1, np.pi - phi_warm, color=config['color_warm'], alpha=0.2)
        ax.plot([np.pi - phi_th1, np.pi - phi_th1], [0, 1], dashes=(4, 4), c=config['color_warm'])
    
    ax.plot(np.pi - phi, r, c=config['color_default'])
    
    return ax

def plot_gaf (gaf, ax=None):
    # Gramian Angular Field
    if ax is None:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
    ax.matshow(gaf)
    
def plot_all (data, score=False, line=False, polar=False, gaf=False):
    if score:
        plot_score(data['penalty'])    
        plot_penalty(data['penalty'])
    if line:
        plot_line(data['x_norm'], data['y'])
    if polar:
        plot_polar(data['phi'], data['r'])
    if gaf:
        plot_gaf(data['gaf'])
