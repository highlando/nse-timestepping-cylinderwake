import matplotlib.pyplot as plt
import numpy as np

__all__ = ['conv_plot',
           'paraplot',
           'print_nparray_tex']


def conv_plot(abscissa, datalist, fit=None,
              markerl=None, xlabel=None, ylabel=None,
              fititem=0, fitfac=1.,
              leglist=None, legloc=1,
              title='title not provided', fignum=None,
              ylims=None, xlims=None,
              yticks=None,
              logscale=False, logbase=10,
              tikzfile=None, showplot=True):
    """Universal function for convergence plots

    Parameters
    ----------
    fititem : integer, optional
        to which item of the data the fit is aligned, defaults to `0`
    fitfac : float, optional
        to shift the fitting lines in y-direction, defaults to `1.0`
    """

    try:
        import seaborn as sns
        sns.set(style="whitegrid")
        # mpilightgreen = '#BFDFDE'
        # mpigraygreen = '#7DA9A8'
        # sns.set_palette(sns.dark_palette(mpigraygreen, 4, reverse=True))
        # sns.set_palette(sns.dark_palette(mpilightgreen, 6, reverse=True))
        # sns.set_palette('cool', 3)
        sns.set_palette('ocean_r', 7)
    except ImportError:
        print('I recommend to install seaborn for nicer plots')

    lend = len(datalist)
    if markerl is None:
        markerl = ['']*lend
    if leglist is None:
        leglist = [None]*lend

    plt.figure(fignum)
    ax = plt.axes()

    for k, data in enumerate(datalist):
        plt.plot(abscissa, data, markerl[k], label=leglist[k])

    if fit is not None:
        fls = [':', ':']
        for i, cfit in enumerate(fit):
            abspow = []
            for ela in abscissa:
                try:
                    abspow.append((ela/abscissa[0])**(-cfit) *
                                  datalist[0][fititem]*fitfac)
                except TypeError:
                    abspow.append((ela/abscissa[0])**(-cfit) *
                                  datalist[0][0][fititem]*fitfac)
            ax.plot(abscissa, abspow, 'k'+fls[i])

    if logscale:
        ax.set_xscale('log', basex=logbase)
        ax.set_yscale('log', basey=logbase)
    if ylims is not None:
        plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)
    if yticks is not None:
        plt.yticks(yticks)
    if title is not None:
        ax.set_title(title)

    plt.legend(loc=legloc)
    plt.grid(which='major')
    _gohome(tikzfile, showplot)
    return


def paraplot(abscissa, datalist, abscissal=None, leglist=None, levels=None,
             markerl=None, xlabel=None, ylabel=None,
             usedefaultmarkers=False,
             title='title not provided', fignum=None,
             ylims=None, xlims=None, legloc='upper left',
             xticks=None,
             downsample=None, keeppointslist=None,
             logscale=None, logscaley=None,
             tikzfile=None, showplot=True,
             colorscheme=None, **kwargs):
    """plot data for several parameters

    Parameters
    ---
    markerl : iterable, optional
        list of (`matplotlib`) markers to be used,
        defaults to `None` (no markers)
    usedefaultmarkers : boolean, optional
        whether to use the `matplotlib` default markers, overrides `markerl`,
        defaults to `False`
    downsample : int, list, optional
        if not `None` whether to only plot every *i*-th data point
    keeppointslist : list, optional
        list of lists of data points that should be kept when downsampling
    """

    lend = len(datalist)
    if markerl is None:
        markerl = ['']*lend
    if usedefaultmarkers:
        import matplotlib
        markerl = matplotlib.markers.MarkerStyle().filled_markers

    if leglist is None:
        leglist = [None]*lend
    # handllist = ['lghdl{0}'.format(x) for x in range(lend)]

    plt.figure(fignum)
    ax = plt.axes()
    if xticks is not None:
        ax.set_xticks(xticks)

    # downsampling the data
    if downsample is not None:
        dsdt, dsabsc = [], []
        for k, data in enumerate(datalist):
            if abscissal is not None:
                abscissa = abscissal[k]
            try:
                samplerate = downsample[k]
            except TypeError:  # all samplings the same for all data
                samplerate = downsample
            filterfield = np.arange(0, len(abscissa), samplerate).tolist()
            if keeppointslist is not None:
                filterfield.extend(keeppointslist[k])
            dwnsmpabs = np.unique(np.array(filterfield))
            dsdt.append(np.array(data)[dwnsmpabs])
            dsabsc.append(np.array(abscissa)[dwnsmpabs])
        datalist, abscissal = dsdt, dsabsc

    leghndll = []
    for k, data in enumerate(datalist):
        labl = leglist[k]
        if abscissal is not None:
            abscissa = abscissal[k]
        if labl is None:
            plt.plot(abscissa, data, markerl[k], linewidth=3, label=leglist[k])
        else:
            # hndl = handllist[k]
            hndl, = plt.plot(abscissa, data,
                             markerl[k], linewidth=3, label=leglist[k])
            leghndll.append(hndl)

    if levels is not None:
        for lev in levels:
            ax.plot([abscissa[0], abscissa[-1]], [lev, lev], 'k')

    if logscale is not None:
        ax.set_xscale('log', basex=logscale)
        ax.set_yscale('log', basey=logscale)
    elif logscaley is not None:
        ax.set_yscale('log', basey=logscaley)
    if ylims is not None:
        plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)

    # plt.legend(handles=leghndll)
    plt.legend(loc=legloc)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # plt.grid(which='both')

    _gohome(tikzfile, showplot)

    return

para_plot = paraplot  # there was a renaming once...


def _gohome(tikzfile=None, showplot=True):
    if tikzfile is not None:
        try:
            from matplotlib2tikz import save as tikz_save
            tikz_save(tikzfile,
                      figureheight='\\figureheight',
                      figurewidth='\\figurewidth')
            print('you may want to use this command\n\\input{' +
                  tikzfile + '}\n')
        except ImportError:
            print('matplotlib2tikz need to export tikz filez')

    if showplot:
        plt.show(block=False)


def print_nparray_tex(array, math=True, fstr='.4f', name=None):
    tdarray = np.atleast_2d(array)
    if name is not None:
        print(name + ' & ')
    if math:
        print(" \\\\\n".join([" & ".join(map(('${0:' + fstr + '}$').
                                         format, line))
                             for line in tdarray]))
    else:
        print(" \\\\\n".join([" & ".join(map(('{0:' + fstr + '}').
                             format, line))
                             for line in tdarray]))
