"""Utilities"""

import tigramite.data_processing as pp

from exp.utils.gen_timeseries import get_regime_mask


def plot_timeseries_regimes_contexts(data, regime_partition, figsize=(15,3)):
    regime_mask = get_regime_mask(data.datasets[0], regime_partition)
    for n_context, ts_context in data.datasets.items():
        T, N = ts_context.shape
        var_names = [r'$X^{%d}$' % j for j in range(N)]
        ts_context_frame = pp.DataFrame(ts_context, var_names=var_names, mask=regime_mask)
        plot_timeseries_regimes(ts_context_frame, figsize=figsize)
        plt.suptitle(f"Context {n_context}")
        plt.show()

"""Adapting the TIGRAMITE plotting functionalities to the multi-context regime setting."""
import numpy as np
from matplotlib import pyplot, pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tigramite.plotting import _make_nice_axes


def plot_timeseries_regimes(
        dataframe=None,
        save_name=None,
        fig_axes=None,
        figsize=None,
        var_units=None,
        time_label="",
        grey_masked_samples=False,
        show_meanline=False,
        data_linewidth=1.0,
        skip_ticks_data_x=1,
        skip_ticks_data_y=1,
        label_fontsize=10,
        color='black',
        alpha=1.,
        tick_label_size=6,
        selected_dataset=0,
        adjust_plot=True,
        selected_variables=None,
):
    """Create and save figure of stacked panels with time series.

    Parameters
    ----------
    dataframe : data object, optional
        This is the Tigramite dataframe object. It has the attributes
        dataframe.values yielding a np array of shape (observations T,
        variables N) and optionally a mask of the same shape.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    fig_axes : subplots instance, optional (default: None)
        Figure and axes instance. If None they are created as
        fig, axes = pyplot.subplots(N,...)
    figsize : tuple of floats, optional (default: None)
        Figure size if new figure is created. If None, default pyplot figsize
        is used.
    var_units : list of str, optional (default: None)
        Units of variables.
    time_label : str, optional (default: '')
        Label of time axis.
    grey_masked_samples : bool, optional (default: False)
        Whether to mark masked samples by grey fills ('fill') or grey data
        ('data').
    show_meanline : bool, optional (default: False)
        Whether to plot a horizontal line at the mean.
    data_linewidth : float, optional (default: 1.)
        Linewidth.
    skip_ticks_data_x : int, optional (default: 1)
        Skip every other tickmark.
    skip_ticks_data_y : int, optional (default: 2)
        Skip every other tickmark.
    label_fontsize : int, optional (default: 10)
        Fontsize of variable labels.
    tick_label_size : int, optional (default: 6)
        Fontsize of tick labels.
    color : str, optional (default: black)
        Line color.
    alpha : float
        Alpha opacity.
    selected_dataset : int, optional (default: 0)
        In case of multiple datasets in dataframe, plot this one.
    selected_variables : list, optional (default: None)
        List of variables which to plot.
    """

    var_names = dataframe.var_names
    time = dataframe.datatime[selected_dataset]

    N = dataframe.N

    if selected_variables is None:
        selected_variables = list(range(N))

    nb_components_per_var = [len(dataframe.vector_vars[var]) for var in selected_variables]
    N_index = [sum(nb_components_per_var[:i]) for i, el in enumerate(nb_components_per_var)]
    nb_components = sum(nb_components_per_var)

    if var_units is None:
        var_units = ["" for i in range(N)]

    if fig_axes is None:
        fig, axes = pyplot.subplots(nb_components, sharex=True, figsize=figsize)
    else:
        fig, axes = fig_axes

    if adjust_plot:
        for i in range(nb_components):

            ax = axes[i]

            if (i == nb_components - 1):
                _make_nice_axes(
                    ax, where=["left", "bottom"], skip=(skip_ticks_data_x, skip_ticks_data_y)
                )
                ax.set_xlabel(r"%s" % time_label, fontsize=label_fontsize)
            else:
                _make_nice_axes(ax, where=["left"], skip=(skip_ticks_data_x, skip_ticks_data_y))
            # ax.get_xaxis().get_major_formatter().set_useOffset(False)

            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.label_outer()

            ax.set_xlim(time[0], time[-1])

            # trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
            if i in N_index:
                if var_units[N_index.index(i)]:
                    ax.set_ylabel(r"%s [%s]" % (var_names[N_index.index(i)], var_units[N_index.index(i)]),
                                  fontsize=label_fontsize)
                else:
                    ax.set_ylabel(r"%s" % (var_names[N_index.index(i)]), fontsize=label_fontsize)

            ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
            # ax.tick_params(axis='both', which='minor', labelsize=tick_label_size)

    _add_timeseries_regimes(
        dataframe=dataframe,
        fig_axes=(fig, axes),
        grey_masked_samples=grey_masked_samples,
        show_meanline=show_meanline,
        data_linewidth=data_linewidth,
        color=color,
        selected_dataset=selected_dataset,
        alpha=alpha,
        selected_variables=selected_variables
    )

    if adjust_plot:
        fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95, hspace=0.3)
        pyplot.tight_layout()

    if save_name is not None:
        fig.savefig(save_name)

    return fig, axes


COL_LBLUE = (0.61, 0.77, 0.89)
COL_GOLDEN = (0.85, 0.65, 0.13)
COL_GREEN = (0.52, 0.73, 0.4)
COL_DBLUE = '#034f84'
COL_VIO ='#654286'
COL_SCHEME = [COL_DBLUE, COL_GOLDEN, COL_GREEN, COL_VIO, COL_LBLUE]


def _add_timeseries_regimes(
        dataframe,
        fig_axes,
        grey_masked_samples=False,
        show_meanline=False,
        data_linewidth=1.0,
        color="black",
        alpha=1.,
        grey_alpha=1.0,
        selected_dataset=0,
        selected_variables=None,
        regime_colors=COL_SCHEME
):
    """Adds a time series plot to an axis.
    Plot of dataseries is added to axis. Allows for proper visualization of
    masked data.

    Parameters
    ----------
    fig : figure instance
        Figure instance.
    axes : axis instance
        Either gridded axis object or single axis instance.
    grey_masked_samples : bool, optional (default: False)
        Whether to mark masked samples by grey fills ('fill') or grey data
        ('data').
    show_meanline : bool
        Show mean of data as horizontal line.
    data_linewidth : float, optional (default: 1.)
        Linewidth.
    color : str, optional (default: black)
        Line color.
    alpha : float
        Alpha opacity.
    grey_alpha : float, optional (default: 1.)
        Opacity of fill_between.
    selected_dataset : int, optional (default: 0)
        In case of multiple datasets in dataframe, plot this one.
    selected_variables : list, optional (default: None)
        List of variables which to plot.
    """
    fig, axes = fig_axes

    # Read in all attributes from dataframe
    data = dataframe.values[selected_dataset]
    if dataframe.mask is not None:
        mask = dataframe.mask[selected_dataset]
    else:
        mask = None

    missing_flag = dataframe.missing_flag
    time = dataframe.datatime[selected_dataset]
    T = len(time)

    if selected_variables is None:
        selected_variables = list(range(dataframe.N))

    nb_components = sum([len(dataframe.vector_vars[var]) for var in selected_variables])

    for j in range(nb_components):

        ax = axes[j]
        dataseries = data[:, j]

        if missing_flag is not None:
            dataseries_nomissing = np.ma.masked_where(
                dataseries == missing_flag, dataseries
            )
        else:
            dataseries_nomissing = np.ma.masked_where(
                np.zeros(dataseries.shape), dataseries
            )


        if mask is not None:
            maskseries = mask[:, j]
            """ regime mask instead of gray/black"""
            for n_regime, regime_id in enumerate(np.unique(maskseries)):

                ax.plot(
                    time[maskseries==regime_id],
                    dataseries_nomissing[maskseries==regime_id],
                    color=regime_colors[n_regime],
                    marker=".",
                    markersize=data_linewidth,
                    linewidth=data_linewidth,
                    clip_on=False,
                    alpha=grey_alpha,
                )

        else:
            if show_meanline:
                ax.plot(time, dataseries_nomissing.mean() * np.ones(T), lw=data_linewidth / 2., color=color)

            ax.plot(
                time,
                dataseries_nomissing,
                color=color,
                linewidth=data_linewidth,
                clip_on=False,
                alpha=alpha,
            )

