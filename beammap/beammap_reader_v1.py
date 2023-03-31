import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u

import glob
import argparse
import sys
import json

if __name__ == '__main__':

    plt.ioff()

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-p', 
                        help='path to input directory')
    parser.add_argument('--save_to_file', '-s', help='save output to file', 
                        action='store_true')
    parser.add_argument('--output_path', '-o', help='path to output directory')
    parser.add_argument('--images', '-i', nargs='+',
                        help='images to plot', default=['signal_I'])
    parser.add_argument('--obsnum', '-n', help='find by obsnum in input dir', 
                        default='none')
    parser.add_argument('--good', '-g', help='plot good detectors', 
                    default=True)
    parser.add_argument('--color_param', '-c', help='what to plot as the color', 
                    default='nw')

    args = parser.parse_args()
    # get path to input directory
    input_path = str(args.input_path)
    plot_good_dets = args.good
    color_param = args.color_param
        
    # get array properties table ecsv file
    try:
        if args.obsnum == 'none':
            ecsv_file = glob.glob(input_path+'/apt*beammap*.ecsv')[0]
        else:
            ecsv_file = glob.glob(input_path+'/apt*beammap*'+args.obsnum.zfill(6)+'*.ecsv')[0]
        
        apt = Table.read(ecsv_file)
    except:
        print('no array properties table ecsv file found')
        sys.exit()

    arrays = ['a1100', 'a1400', 'a2000']

    if args.obsnum == 'none':
        obsnum = ecsv_file.split('/')[-1].split('_')[1]
    else:
        obsnum = args.obsnum
    print(obsnum)
    # get good detectors
    good = apt['flag'] == 1
    if (plot_good_dets):
        tg = apt[good]
    else:
        tg = apt

    nds = []
    for a in range(3):
        m = tg['array'] == a
        nds.append(len(tg[m]))


    # set up the big plot
    matplotlib.rcParams.update({'font.size': 8})
    cmap = ['tab20b', 'Dark2', 'Set1']
    if(plt.fignum_exists(9)):
        plt.clf()
    fig, axes = plt.subplots(num=9, ncols=3, nrows=2, figsize=(8, 5),
                             constrained_layout=True)
    plt.suptitle('APT: {} '.format(ecsv_file.split('/')[-1]))

    # first row is a plot of the arrays
    for a, ax, n, cc in zip(range(3), axes[0,:], nds, cmap):
        ax.cla()
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_title('{0:} - {1:} - {2:} detectors'.format(obsnum, arrays[a], n), fontsize=8)
        m = tg['array'] == a
        ax.scatter(tg[m]['x_t'].data, tg[m]['y_t'].data, s=5,
                   #c=tg[m]['nw'], cmap=cc)
                   c=tg[m][color_param], cmap=cc)
        ax.set_xlabel('x offset ["]')
        ax.set_ylabel('y offset ["]')

    # last row is a histogram of the fitted FWHM
    for a, ax in zip(range(3), axes[1,:]):
        ax.cla()
        ax.set_title('Histogram of Fitted FWHM', fontsize=8)
        m = tg['array'] == a
        ax.hist(tg[m]['a_fwhm'], bins=50, label='x FWHM')
        ax.hist(tg[m]['b_fwhm'], bins=50, label='y FWHM')
        ax.set_xlabel('Fitted FWHM')
        ax.set_ylabel('# of detectors')
        ax.legend()
        ax.set_xlim(3,12)


     # check if saving is requested
    if args.save_to_file:
        if args.output_path is not None:
            fig.savefig(args.output_path + '/toltec_beammap_' + obsnum + '_image.png', 
                        bbox_inches='tight')
        else:
            print('no save file output path specified')
            sys.exit()

    #plt.show()

