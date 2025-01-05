### here come all the functions that we frequently/ repeatedly used for plotting any data or images/movies
import os
import sys
import math
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread

cwd = os.getcwd()
modules = cwd.replace('/exe/IF_analysis', '')
sys.path.insert(1, modules)
import modules.configs_setup as configs
import modules.raw_IF_analysis as raw
import modules.javaThings as jv
jv.start_jvm()
jv.init_logger()


def AddScale_well(ax, scale_pix, Verticalsize, s=2, unit='$\mu$m', c='white', ratio=50):
    """
    Add a scale bar to a matplotlib axis.

    This function inserts a scale bar into the plot to visually represent the
    scale of the image or data within the provided matplotlib axis. It provides
    customization for the size, color, and unit of measurement of the scale bar,
    as well as its vertical thickness. The scale bar is positioned in the 'lower
    right' of the axis by default.

    :param ax: The matplotlib axis to which the scale bar will be added.
    :type ax: matplotlib.axes._axes.Axes
    :param scale_pix: The pixel length of the scale bar.
    :type scale_pix: float or int
    :param Verticalsize: The total vertical size of the scale bar as
        a measurement.
    :type Verticalsize: float or int
    :param s: Font size for the scale bar label, defaults to 2.
    :type s: float or int, optional
    :param unit: The unit of measurement for the scale, defaults to '$\mu$m'.
    :type unit: str, optional
    :param c: The color of the scale bar, defaults to 'white'.
    :type c: str, optional
    :param ratio: The ratio to divide `Verticalsize` to establish the vertical
        thickness of the scale bar, defaults to 50.
    :type ratio: float or int, optional
    :return: None
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=s)
    scalebar = AnchoredSizeBar(ax.transData,
                               scale_pix, None, 'lower right',
                               pad=5,
                               color=c,
                               frameon=False,
                               size_vertical=Verticalsize/ratio,
                               fontproperties=fontprops)

    ax.add_artist(scalebar)

def czi_plot_overview(df, config, param='imageID', morph_analysis=True, mark_filtered=True, filtered_gastrus=[], ncol=7,
                      save=True, cmap='binary'):
    """
    Generates an overview plot visualizing data from a given DataFrame, where images and
    their annotations, such as medial axes or contours, are displayed in a grid layout.
    The function allows for marking filtered images, custom channel selection, and analysis
    of morphological data.

    :param df:
        Input DataFrame containing image data and metadata such as image paths,
        scaling factors, and ID information.

    :param config:
        Dictionary containing configuration parameters, including channel information,
        plot settings, and save paths.

    :param param:
        The column name in the DataFrame to use as a label for the images in the
        plot grid. Defaults to 'imageID'.

    :param morph_analysis:
        Boolean flag indicating whether to display morphological analysis, such as the
        medial axis or contours, on the plot. Defaults to True.

    :param mark_filtered:
        Boolean flag indicating whether to mark filtered items in the plot using a
        specific marker. Defaults to True.

    :param filtered_gastrus:
        List of IDs corresponding to filtered images that should be highlighted in
        the plot. Defaults to an empty list.

    :param ncol:
        Number of columns in the plot grid layout. Determines how images are
        arranged in terms of rows and columns. Defaults to 7.

    :param save:
        Boolean flag indicating whether to save the generated plot to a file. If False,
        displays the plot instead of saving it. Defaults to True.

    :param cmap:
        Colormap used for displaying the images in the plot. Defaults to 'binary'.

    :return:
        None. Performs plotting and optionally saves the generated plot to disk.
    """
    ch_dict = config['dict_channel'][0]
    channel = ch_dict[config['overview_plot']['ch_to_use']]

    if len(filtered_gastrus) > 0:
        filtered = [oid for oid in df['imageID'] if oid.rsplit('_', 1)[1] in filtered_gastrus]

    ncol = ncol
    nrow = math.ceil(len(df) / ncol)

    if len(df) < ncol:
        fig, axes = plt.subplots(1, len(df), figsize=(len(df), 1))

    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))

        for i in range(nrow * ncol - len(df)):
            axes[nrow - 1, ncol - i - 1].set_visible(False)

    for ix, ax in zip(df.index, axes.flatten()):

        imagepath = df.at[ix, 'absPath']
        if os.path.exists(imagepath):
            imtiff = tifffile.imread(imagepath)
            image = np.int16(imtiff[channel, :, :])

        um_per_pixel = df.at[ix, 'um_per_pixel']
        ID = df.at[ix, 'imageID'].rsplit('_', 1)[1]

        ax.imshow(image, cmap=cmap)

        if morph_analysis:
            try:
                medax = np.array(df.at[ix, 'MedialAxis'])
                xAx, yAx = medax.T
                cnt = np.array(df.at[ix, 'Contour'])
                xCnt, yCnt = cnt.T

                ax.plot(yAx, xAx, c='r', lw=1)
                ax.plot(yCnt, xCnt, c='k', lw=1)

            except:
                ax.scatter(0.1 * image.shape[0], 0.1 * image.shape[1], c='k', marker='x', s=20)

        if mark_filtered:
            if ID in filtered_gastrus:
                df.at[ix, 'Flag'] != 'Keep'
                ax.scatter(0.2 * image.shape[0], 0.1 * image.shape[1], c='r', marker='x', s=20)

        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.set_title(f'i:{ix}\n{df.at[ix, param]}', fontsize=5, pad=0)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        if ix == 0:
            AddScale_well(ax, 100 / um_per_pixel, 100, c='k')

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)

    if save:
        folder = config['plots_folder']
        ending = config['overview_plot']['format']
        if (param == 'imageID') and (len(df.expID.unique()) == 1):
            savename = folder + f'{df.at[ix, "expID"]}_overview{ending}'
        else:
            savename = folder + f'{df.at[ix, param]}_overview{ending}'
        plt.savefig(savename, dpi=200, transparent=True)
        print('Successfully saved the overview plot under ', savename)

    else:
        plt.show()
        plt.close()

    config['overview_plot'] = {'ch_to_use': config['overview_plot']['ch_to_use'], 'format': ending, 'param': param,
                               'morph_analysis': morph_analysis, 'mark_filtered': mark_filtered,
                               'filtered_gastrus': filtered_gastrus, 'ncol': ncol, 'saved': save, 'cmap': cmap}

    outlier_dict = {key: list() for key in config['aliases']}
    config['outliers'] = outlier_dict
    configs.update_configs(config)


