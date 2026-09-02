""" for seaborn plotting"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def mirrorhistplot(data, x, y, hue=None, col=None, row=None,
                   order=None, hue_order=None, row_order=None, col_order=None,
                   bins=20, binrange=None, stat="probability", max_width=0.5, dodge=True,
                   height=5, aspect=None, palette=None, color=None, alpha=0.75,
                   sharex=True, sharey=True, legend=True, legend_out=True,
                   ax=None):
    """ FacetGrid-style mirror histogram plot (boxen layout, full distributions).

    Like sns.catplot, but draws symmetric horizontal histograms around a vertical
    axis at each x category.

    Colors encode hue only. When hue is not set, all histograms use a single color.

    PARAMS:
    - data, dataframe
    - x, categorical column for x-axis positions (e.g. "model")
    - y, numeric column for the distribution (e.g. "score_frac")
    - hue, optional grouping for color (dodged at each x when x is also set)
    - col, row, optional facet variables (like sns.catplot)
    - order, hue_order, row_order, col_order, category orders
    - bins, binrange, histogram binning (binrange e.g. (0, 1))
    - stat, "probability" (default; bin probabilities sum to 1, encoded in
      horizontal bar extent, not y-bin width) or "density" (peak-normalized PDF)
    - max_width, scale for horizontal bar extent (for stat="probability", the
      bin probabilities sum to max_width after scaling)
    - dodge, if True and hue is set, dodge hue groups at each x
    - height, aspect, FacetGrid size params
    - palette, seaborn palette name, list of colors, or dict mapping hue levels
      to colors (first entry used when hue is not set)
    - color, single color when hue is not set
    - ax, if provided, draw on this axes and return it (no faceting)

    RETURNS:
    - sns.FacetGrid (or matplotlib Axes if ax is provided)
    """

    ### HELPER FUNCTIONS
    def _resolve_var_order(data, var, order):
        if var is None:
            return None
        if order is not None:
            return list(order)
        if pd.api.types.is_categorical_dtype(data[var]):
            return list(data[var].cat.categories)
        return list(pd.unique(data[var]))


    def _resolve_hue_color_map(hue_order, palette):
        """Map hue levels to colors; palette may be a seaborn name, list, or dict."""
        n = len(hue_order)
        if palette is None:
            pal = sns.color_palette("deep", n_colors=n)
        elif isinstance(palette, dict):
            fallback = sns.color_palette("deep", n_colors=n)
            return {lev: palette.get(lev, fallback[i]) for i, lev in enumerate(hue_order)}
        elif isinstance(palette, str):
            pal = sns.color_palette(palette, n_colors=n)
        else:
            pal = list(palette)
            if len(pal) < n:
                extra = sns.color_palette("deep", n_colors=n - len(pal))
                pal = pal + list(extra)
        return {lev: pal[i] for i, lev in enumerate(hue_order)}


    def _compute_mirrorhist_bin_edges(data, y, bins, binrange):
        if binrange is not None:
            return np.linspace(binrange[0], binrange[1], bins + 1)
        vals = data[y].dropna().values
        if len(vals) == 0:
            return np.linspace(0, 1, bins + 1)
        return np.histogram_bin_edges(vals, bins=bins)


    def _draw_mirror_hist_bars(ax, vals, x_center, color, bin_edges, max_width,
                            stat="density", alpha=0.75, edgecolor="white",
                            linewidth=0.3):
        """Draw one symmetric mirror histogram centered at x_center."""
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_height = bin_width * 0.95

        counts, _ = np.histogram(vals, bins=bin_edges, density=False)
        if stat == "probability":
            if counts.sum() > 0:
                values = counts / counts.sum()
            else:
                values = counts.astype(float)
            values = values * max_width
        elif stat == "density":
            values, _ = np.histogram(vals, bins=bin_edges, density=True)
            if values.max() > 0:
                values = values / values.max() * max_width
        else:
            raise ValueError(f"stat must be 'density' or 'probability', got {stat!r}")

        for y, w in zip(bin_centers, values):
            if w <= 0:
                continue
            ax.barh(y, w, height=bar_height, left=x_center, color=color, alpha=alpha,
                    edgecolor=edgecolor, linewidth=linewidth)
            ax.barh(y, -w, height=bar_height, left=x_center, color=color, alpha=alpha,
                    edgecolor=edgecolor, linewidth=linewidth)


    def _mirrorhist_on_ax(ax, data, x, y, hue=None, order=None, hue_order=None,
                        bin_edges=None, max_width=0.4, dodge=True, palette=None,
                        stat="density", alpha=0.75, draw_axis_lines=True, color=None):
        """Draw mirror histograms for one facet onto ax."""
        if x is None and hue is None:
            raise ValueError("mirrorhistplot requires at least one of x or hue")

        if x is None:
            x_var = hue
            x_order = _resolve_var_order(data, hue, hue_order)
            color_var = hue
            color_order = x_order
        else:
            x_var = x
            x_order = _resolve_var_order(data, x, order)
            color_var = hue
            color_order = _resolve_var_order(data, hue, hue_order) if hue else None

        if color_var is not None and color_var == x_var and x is not None:
            raise ValueError("x and hue must be different variables")

        n_x = len(x_order)
        if color_var is None:
            if color is not None:
                bar_color = color
            elif palette is None:
                bar_color = sns.color_palette("deep")[0]
            elif isinstance(palette, str):
                bar_color = palette
            elif isinstance(palette, dict):
                bar_color = next(iter(palette.values()))
            else:
                bar_color = palette[0]
            color_map = None
        else:
            color_map = _resolve_hue_color_map(color_order, palette)
            bar_color = None

        if color_var is None:
            for i, xval in enumerate(x_order):
                vals = data.loc[data[x_var] == xval, y].dropna().values
                if len(vals) == 0:
                    continue
                _draw_mirror_hist_bars(ax, vals, i, bar_color, bin_edges,
                                    max_width, stat=stat, alpha=alpha)
                if draw_axis_lines:
                    ax.axvline(i, color="0.85", lw=1, zorder=0)
        elif x is not None:
            n_color = len(color_order)
            slot_width = 0.8 / n_color if dodge else 0.8
            group_max_width = max_width * slot_width if dodge else max_width / n_color
            for i, xval in enumerate(x_order):
                if draw_axis_lines:
                    ax.axvline(i, color="0.85", lw=1, zorder=0)
                for j, cval in enumerate(color_order):
                    vals = data.loc[(data[x_var] == xval) & (data[color_var] == cval), y].dropna().values
                    if len(vals) == 0:
                        continue
                    x_center = i + (j - (n_color - 1) / 2) * slot_width if dodge else i
                    _draw_mirror_hist_bars(ax, vals, x_center, color_map[cval], bin_edges,
                                        group_max_width, stat=stat, alpha=alpha)
        else:
            for i, cval in enumerate(color_order):
                vals = data.loc[data[x_var] == cval, y].dropna().values
                if len(vals) == 0:
                    continue
                _draw_mirror_hist_bars(ax, vals, i, color_map[cval], bin_edges,
                                    max_width, stat=stat, alpha=alpha)
                if draw_axis_lines:
                    ax.axvline(i, color="0.85", lw=1, zorder=0)

        ax.set_xlabel(x_var)
        ax.set_ylabel(y)
        if n_x == 0:
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
            return
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_order)
        ax.set_xlim(-0.5, n_x - 0.5)

    ### RUN
    order_resolved = _resolve_var_order(data, x, order) if x is not None else None
    hue_order_resolved = _resolve_var_order(data, hue, hue_order) if hue is not None else None

    if ax is not None:
        if row is not None or col is not None:
            raise ValueError("Pass either ax or row/col faceting, not both")
        bin_edges = _compute_mirrorhist_bin_edges(data, y, bins, binrange)
        _mirrorhist_on_ax(ax, data, x, y, hue=hue, order=order_resolved,
                          hue_order=hue_order_resolved,
                          bin_edges=bin_edges, max_width=max_width, dodge=dodge,
                          palette=palette, color=color, stat=stat, alpha=alpha)
        return ax

    if aspect is None:
        x_order = order_resolved if x is not None else hue_order_resolved
        aspect = max(1.0, 0.25 * len(x_order))

    bin_edges = _compute_mirrorhist_bin_edges(data, y, bins, binrange)

    def _plot_facet(data, **kwargs):
        _mirrorhist_on_ax(plt.gca(), data, x, y, hue=hue,
                          order=order_resolved, hue_order=hue_order_resolved,
                          bin_edges=bin_edges, max_width=max_width, dodge=dodge,
                          palette=palette, color=color, stat=stat, alpha=alpha)

    g = sns.FacetGrid(data, row=row, col=col, height=height, aspect=aspect,
                      row_order=row_order, col_order=col_order,
                      sharex=sharex, sharey=sharey,
                      legend_out=legend_out)
    g.map_dataframe(_plot_facet)

    if legend and hue is not None:
        color_map = _resolve_hue_color_map(hue_order_resolved, palette)
        from matplotlib.patches import Patch
        handles = [Patch(color=color_map[lev], label=str(lev)) for lev in hue_order_resolved]
        leg = g.figure.legend(handles=handles, title=hue, bbox_to_anchor=(1.02, 0.5),
                              loc="center left")
        if legend_out:
            g.figure.subplots_adjust(right=0.85)

    return g

def rotateLabel(ax, rotation=45, horizontalalignment="right"):
    """ seaborn, maek sure to add labels for catplot
    ax = sns.catplot(...)
    """

    fig = ax

    # PRoblem, this sometimes deltes it, I think
    # for a in fig.axes.flat:
    #     a.set_xticklabels(a.get_xticklabels(), rotation=rotation, 
    #         horizontalalignment=horizontalalignment)

    # This works...
    for ax in fig.axes.flat:
    #     ax.set_xticks(ax.get_xticks(), rotation=45)
        list_text = [this.get_text() for this in ax.get_xticklabels()]
        if len(list_text)>0:
            ax.set_xticklabels(list_text,rotation=rotation, horizontalalignment="right")

def addLabel(ax):
    """ seaborn, maek sure to add labels for catplot"""
    for a in ax.axes.flat:
        a.set_xticklabels(a.get_xticklabels(), rotation=45, 
            horizontalalignment="right")


def addTextLabelToPoints(ax, df, x, y, name):
    """ 
    adds on top of points labels in text.
    e.g., ax = sns.scatterplot(x="x", y="y", hue="category", data=df, hue_order=sorted(list(set(tasks_all_categories))))
    addTextLabelToPoints(ax, df, "x", "y", "name"),
    i.e,., tell me the x, y, and text strings to use 
    """
    
    # add annotations one by one with a loop
    for line in range(df.shape[0]):
        ax.text(df[x].values[line], df[y].values[line], df[name].values[line], 
                 horizontalalignment='left', size='medium', color='black',
                alpha=0.6)

def plotgood_lineplot(data, xval, yval, line_grouping, include_scatter=False,
    color_single=None, 
    lines_add_ci=False,
    rowvar=None, colvar=None, col_wrap=None, 
    height=4, aspect=1,
    include_mean = False, 
    relplot_kw = None):
    """ Flexible plotter for lineplots, where x is categorical (or small num of
    discrete vals) and y is scalar. Overlays lines on the same plot, and does summaries
    across lines. Can also overlay scatter of each datrapt.
    Common use: for each character draw a line representing its value across n epochs (x values).
    THen overlay the mean over characters.
    NOTE: This supercedes relPlotOverlayLineScatter and relplotOverlaid, which vbasically do the
    same thing, but the latter makes all lines the same color.
    PARAMS:
    - data, dataframe
    - xval, string, name of x var.
    - yval, string,
    - line_grouping, string, each unique level of this var gets its own line.
    - include_scatter, bool, if True, then lines are overlaid on scatters of each datapt.
    - color_single, either None (lines are each diff color) or string color (e..g, "k") applied to
    all lines. Scatter pts are always colored.
    - lines_add_ci, bool, if True, then each line includes shaded error bar. (default is 68% ci)
    - rowvar, colvar, str, variables that define subplots
    RETURNS:
    - handle to figure
    NOTE: see https://stackoverflow.com/questions/46598371/overlay-a-line-function-on-a-scatter-plot-seaborn
    for overlaying scatter and line.
    """

    assert line_grouping!=xval, 'a mistake sometimes made...'
    if relplot_kw is None:
        relplot_kw = {}
    if color_single:
        # then turn off legend
        legend=False
    else:
        legend = True

    # METHOD 1 - 
    # g = sns.FacetGrid(data, row=rowvar, hue=line_grouping, col=colvar,
    #                 sharex=True, sharey=True, height=height, aspect=aspect,
    #                 legend_out=True)

    # catlist = set(data[line_grouping])
    # if color_single is not None:
    #     palette = {cat:color_single for cat in catlist}
    # else:
    #     palette = None
    # # g = g.map(sns.scatterplot, xval, yval, palette=palette)
    # g = g.map(sns.lineplot, xval, yval,  ["k"], palette=palette, legend="full")

    # METHOD 2
    # catlist = set(data[line_grouping])
    # palette = {cat:color_single for cat in catlist}
    # g = sns.relplot(kind='scatter', x=xval, y=yval, data=data, 
    #     hue = line_grouping,
    #     height=height, aspect=aspect)
    # # g.map_dataframe(sns.lineplot, xval, yval, color='g')

    # if color_single is not None:
    #     palette = {cat:color_single for cat in catlist}
    # else:
    #     palette = None
    # g.map_dataframe(sns.lineplot, xval, yval, hue=line_grouping, palette=palette)

    # METHOD 3 - use axes and pass ax into axes level plotting functions.

    # METHOD 4
    catlist = set(data[line_grouping])
    if color_single is not None:
        palette = {cat:color_single for cat in catlist}
    else:
        palette = None
    
    if lines_add_ci:
        errorbar=("ci", 68)
    else:
        errorbar = None

    if rowvar:
        relplot_kw["row"] = rowvar
    if colvar:
        relplot_kw["col"] = colvar
    if col_wrap:
        relplot_kw["col_wrap"] = col_wrap

    g = sns.relplot(data=data, kind='line', x=xval, y=yval,
        hue = line_grouping, 
        height=height, aspect=aspect,
        palette=palette, errorbar=errorbar,
        legend=legend,
        **relplot_kw)
    # g = sns.relplot(data=data, kind='line', x=xval, y=yval,
    #     hue = line_grouping, 
    #     height=height, aspect=aspect,
    #     palette=palette, errorbar="se", 
    #     legend=legend,
    #     **relplot_kw)

    if include_scatter:
        # usually the dots you want to allow their orig color
        # g.map_dataframe(sns.scatterplot, xval, yval, hue=line_grouping, palette=palette)
        g.map_dataframe(sns.scatterplot, xval, yval, 
            hue=line_grouping, alpha=0.5, legend=legend)
    
    if include_mean:
        g.map_dataframe(sns.lineplot, xval, yval, alpha=0.5, legend=legend)

    return g


def relplotOverlaid(df, line_category, color, **relplotkwargs):
    """ if want to plot single lines for each cataegory in 
    line_category, and all the same color. sns I think forces you to 
    either average over all categories in line_category (i.e., one
    output per facet, or hue, etc, or they will be different colors if
    use hue=line_category. Here can make all the same color. 
    """
    # print(relplotkwargs)
    # assert False
    assert False, "superceded by plotgood_lineplot"
    catlist = set(df[line_category])
    palette = {cat:color for cat in catlist}
    relplotkwargs["legend"]= False # since all same color, legend is useulse..
    relplotkwargs["palette"]= palette
    relplotkwargs["data"] = df
    return sns.relplot(**relplotkwargs)


def relPlotOverlayLineScatter(data, x, y, hue=None, row=None, col=None, palette=None,
    height=3, aspect=3):
    """
    Overlay line and scatterplot.
    row="block"
    """
    assert False, "superceded by plotgood_lineplot"

    # g = sns.FacetGrid(data, row=row, sharex=True, sharey=True, height=3, aspect=3,
    #                  legend_out=True)
    g = sns.FacetGrid(data, row=row, hue=hue, col=col, sharex=True, sharey=True, height=height, aspect=aspect,
                     legend_out=True)

    catlist = set(data[hue])
    palette = {n:"k" for n in catlist}

    # print(x)
    # print(y)
    # print(hue)
    # print(palette)
    # assert False
    # g = g.map(sns.scatterplot, x, y, hue=hue, palette=palette)
    # g = g.map(sns.lineplot, x, y, hue=hue, palette=palette, legend="full")
    g = g.map(sns.scatterplot, x, y, palette=palette)
    g = g.map(sns.lineplot, x, y,  palette=palette, legend="full")
    return g


def pairplot_corrcoeff(data, x_vars=None, y_vars = None, hue=None, vars=None, aspect=1,
    corrver="spearman"):
    """ like sns.pairplot, but overlaying pearsons r and p.
    """
    from scipy import stats
    def corrfunc(x, y, **kws):
        if corrver=="pearson":
            r, p = stats.pearsonr(x, y)
        elif corrver=="spearman":
            r, p = stats.spearmanr(x, y)
        else:
            assert False

        ax = plt.gca()
        ax.annotate(f"r={r:.2f}|p={p:.4f}", xy=(.1, .9), xycoords=ax.transAxes)

    g = sns.pairplot(data=data, x_vars = x_vars, y_vars = y_vars, aspect=aspect,
                      kind="reg", hue=hue)
    # g.map_upper(plt.scatter, s=10)
    # g.map_diag(sns.distplot, kde=False)
    # g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map(corrfunc)
    return g

def map_function_tofacet(fig, func):
    """
    fig is a facetgrid plot, e.g., catplot
    func(ax), does stuff with ax,applies to each facet
    """

    def F(*pargs, **kws):
        ax = plt.gca()
        func(ax)
        # ax.axhline(0, alpha=0.2, **kws)

    fig.map(F)


def timecourse_overlaid(df, feat, xval="tvalfake", YLIM=None, row=None, col=None, 
    grouping=None,
    ALPHA = 0.5, doscatter=True, domean=True, squeeze_ylim=True, col_wrap=4):
    """ plot timecourse, both scatter of pts and overlay means"""
    
    if grouping:
        tmp = df[grouping].unique().tolist()
        colors = ["r", "b", "g", "k"]
        if len(tmp)<=len(colors):
            PALLETE = {t:c for t, c in zip(tmp, colors)}
        else:
            PALLETE=None
    else:
        PALLETE = None

    if YLIM is None and squeeze_ylim and doscatter:
        from .plottools import get_ylim
        YLIM = get_ylim(df[feat])

    if row is None and col is None:
        g = sns.FacetGrid(df, height=4, aspect=2, 
                          sharex=True, sharey=True, ylim=YLIM)
    else:
        g = sns.FacetGrid(df, row=row, col=col, height=4, aspect=2, 
                          sharex=True, sharey=True, ylim=YLIM, col_wrap=col_wrap)

    # Different plots, depening on if timecourse, or summaries.
    if xval=="tvalfake":
        xvalmean = "tvalday"
        scatplot = sns.scatterplot
        meanplot = sns.lineplot
    elif xval=="epoch":
        xvalmean = "epoch"
        scatplot = sns.swarmplot
        meanplot = sns.pointplot
    else:
        print(xval)
        assert False, "not sure what is mean summary for this xval"

    if domean:
        g.map(meanplot, xvalmean, feat, **{"err_style":"bars", "ci":68, "color":"k", "linewidth":2})
    if doscatter:
        if xval=="tvalfake":
            g.map(sns.scatterplot, xval, feat, **{"hue":grouping,
                                                            "marker":"x", 
                                                          "alpha":ALPHA,
                                                                  "s":40, 
                                                                  "palette":PALLETE})
        elif xval=="epoch":
            g.map(sns.swarmplot, xval, feat, **{"hue":grouping, "alpha":ALPHA,
                                                                  "s":4, 
                                                                  "palette":PALLETE})
    g.map(plt.axhline, **{"color":[0.7, 0.7, 0.7]})

    return g


def get_xticklabels(fig):
    """ Returns teh labels in order for each axis in fig
    PARAMS:
    - fig, a facetgrid object, e.g,, output from sns.catplot
    RETURNS:
    - list of labels, where list is len num axes, and each inner list is 
    len num x labels, in order.
    """
    out = []
    for ax in fig.axes.flatten():
        out.append([lab.get_text() for lab in ax.get_xticklabels()])
    return out
            

def heatmap_mat(datamat, ax=None, annotate_heatmap=True, zlims=(None, None),
        robust=False, diverge=False, labels_row=None, labels_col=None,
                rotation=90, rotation_y=0, cbar=True, continuous_axes=False):
    """
    Plot heatmap, given datamat shape (nrow, ncols).
    """
    df = pd.DataFrame(datamat)
    return heatmap(df, ax, annotate_heatmap, zlims,
                   robust, diverge, labels_row, labels_col, rotation, rotation_y,
                   cbar=cbar, continuous_axes=continuous_axes)

def heatmap(df, ax=None, annotate_heatmap=True, zlims=(None, None),
            robust=False, diverge=False, labels_row=None, labels_col=None,
            rotation=90, rotation_y=0, SHAPE="square", norm_method=None,
            cbar=True, continuous_axes=False, diverge_center_dark=False):
    """ 
    Plot a heatmap dictated by cols and rows of df, where the cells correspond to values
    in df
    PARAMS:
    - df, wideform dataframe to plot, should be in 2d shape, with rows and columns, the sahpe of 
    the resulting plot. df.Index are rows (from top to bottom), and df.columns are columns
    (left to right). See pandastools.convert_to_2d_dataframe to convert from long-form
    to this wideform.
    - annotate_heatmap, bool, whether puyt text in cell indicating the values
    - diverge, if True, then centers the heat
    - continuous_axes, bool, if True, then axes values match the x and y va;lues (they are not just labels). 
    This requires numerical labels_row and labels_col
    RETURNS:
    - fig, 
    - ax, 
    - rgba_values, (nrows, ncols, 4), where rgba_values[0,1], means rgba value for row 0 col 1.
    """

    # NOTE, from neural plot heatmap..
    # sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
    #    robust=robust, vmin=zlims[0], vmax=zlims[1])

    if zlims is None:
        zlims = (None, None)
    # make a copy, with these columns
    if labels_row is None:
        list_cat_1 = df.index.tolist()
    else:
        list_cat_1 = labels_row

    if labels_col is None:
        list_cat_2 = df.columns.tolist()
    else:
        list_cat_2 = labels_col

    if SHAPE == "rect":
        if len(list_cat_2)>10:
            w = len(list_cat_2)/10*3.5
        else:
            w = 5
        h = 5
    elif SHAPE == "square":
        if len(list_cat_2)>10:
            w = len(list_cat_2)/10*3.5
        else:
            w = 5
        h = w
    else:
        assert False

    # Clip to maximum size.
    SIZEMAX = 20
    aspect = h/w
    if h>SIZEMAX:
        h = SIZEMAX
        w = h/aspect
    if w>SIZEMAX:
        w = SIZEMAX
        h = aspect*w
    
    # print(SHAPE, w,h, ax is None)

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(w, h))
    else:
        fig = None

    # print("SIZE:", w,h)

    dfthis = df
    if norm_method in ["col_div", "row_div", "all_div"]:
        assert np.all(dfthis.values[np.isfinite(dfthis.values)] >= 0), "Found negative finite values"
        # if np.all(dfthis==0):
        #     dfthis += 0.0001

    if norm_method=="all_sub":
        # minus mean over all cells
        dfthis = dfthis - dfthis.mean().mean()
        diverge = True
    elif norm_method=="col_div":
        # normalize so that for each col, the sum across rows is 1
        if False:
            assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
            dfthis = dfthis.div(dfthis.sum(axis=0), axis=1)
        else:
            col_sums = dfthis.sum(axis=0, skipna=True)
            col_sums[col_sums == 0.0] = 0.001
            dfthis = dfthis.div(col_sums, axis=1)
    elif norm_method=="row_div":
        # same, but for rows
        if False: # fails if any nans
            assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
            dfthis = dfthis.div(dfthis.sum(axis=1), axis=0)
        else:
            row_sums = dfthis.sum(axis=1, skipna=True) # one value for each row
            row_sums[row_sums==0.]=0.001 # So no divide by 0.
            dfthis = dfthis.div(row_sums, axis=0) 
    elif norm_method in ["all_div", "div_all"]:
        # divide by sum of all counts
        if True:
            assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
            dfthis = dfthis/dfthis.sum().sum()
        else:
            print("---")
            print(dfthis)
            global_mean = np.nansum(dfthis.values.flatten())
            if global_mean == 0.0:
                global_mean = 0.001
            print(global_mean)
            dfthis = dfthis / global_mean
            print(dfthis)
            
    elif norm_method=="col_sub":
        # normalize so by subtracting from each column its mean across rows
        dfthis = dfthis.subtract(dfthis.mean(axis=0), axis=1)
        diverge = True
    elif norm_method=="col_sub_notdiverge":
        # normalize so by subtracting from each column its mean across rows
        dfthis = dfthis.subtract(dfthis.mean(axis=0), axis=1)
        diverge = False
    elif norm_method=="row_sub":
        # normalize so by subtracting from each column its mean across rows
        dfthis = dfthis.subtract(dfthis.mean(axis=1), axis=0)
        diverge = True
    elif norm_method=="row_sub_firstcol":
        # for each item in a given row, subtract the value of the first colum in that row.
        dfthis = dfthis.subtract(dfthis.iloc[:,0], axis=0)
    elif norm_method is None:
        pass
    else:
        print(dfthis)
        print(norm_method)
        assert False
    df = dfthis

    # compute zlims here, just so you can extract colors accruately below.
    z1, z2 = zlims
    if z1 is None:
        z1 = df.min().min()
    if z2 is None:
        z2 = df.max().max()
    if diverge:
        # then center at 0
        z = np.max(np.abs([z1, z2]))
        z1 = -z
        z2 = z
    # Make sure z1 is less than z2
    if z1>z2:
        print(z1, z2)
        assert False, "how is this possible.."
    else:
        # Make sure z1 is less than z2
        z1 = np.min([z2-0.001, z1])

    if diverge:
        # center at 0, and use diverging palletee
        # 
        # center = 0
        # cmap = sns.color_palette("vlag")
        if diverge_center_dark:
            cmap = sns.diverging_palette(220, 20, center="dark", as_cmap=True)
        else:
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
        lab_add = 0.5
    else:
        # center = None
        # cmap = sns.color_palette("rocket", as_cmap=True)
        cmap = sns.color_palette("rocket", as_cmap=True).reversed() # lower numbers are lighter hue (and thus pops out), this is useful ebcuase I often plot distance scores, and lower is important.
        # cmap = sns.color_palette
        lab_add = 0.5

    if not continuous_axes:
        # Original, categorical axes.
        sns.heatmap(df, annot=annotate_heatmap, ax=ax, vmin=z1, vmax=z2,
            robust=robust, cmap=cmap, cbar=cbar)
        
        # Categorical balues
        if len(list_cat_1)<400:
            # otherwise is too slow, too much text.
            ax.set_yticks([i+lab_add for i in range(len(list_cat_1))], list_cat_1, rotation=rotation_y, fontsize=6)
        if len(list_cat_2)<400:
            ax.set_xticks([i+lab_add for i in range(len(list_cat_2))], list_cat_2, rotation=rotation, fontsize=6)
    else:
        # Continuous axes.
        # If x and y are numerical and evenly spaced, then plot using actual values on x and y axis.
        # NOte: y starts from -1, so that is top to bottom, matching sns.heatmap
        X = df.values
        extent=[list_cat_2[0], list_cat_2[-1], list_cat_1[-1], list_cat_1[0]] 
        # print(extent)
        # print(X.shape)
        
        # add 1, so that the min and max aspects are OUTSIDE the heatmap        
        # e.g, if you have 28 rows, you want the y lims on plot to be 0, 29. By fdefault it would be
        # (0, 28)
        if list_cat_1[0] == X.shape[0]-1:
            extent[3] = extent[3]+1
        if list_cat_1[-1] == X.shape[0]-1:
            extent[2] = extent[2]+1

        img = ax.imshow(X, aspect='auto', extent=extent, 
                  cmap=cmap, vmin=zlims[0], vmax=zlims[1], interpolation="none")
        _ = plt.colorbar(img, orientation='vertical')

        # Alteramntive, but the above worked, so this is ignroed
        # img = ax.pcolormesh(list_cat_2, list_cat_1, X, cmap=cmap, vmin=zlims[0], vmax=zlims[1], shading='auto')
        # cbar1 = plt.colorbar(img, orientation='vertical', ax=ax)

        # print(list_cat_2)
        # ax.axvline(0)
        # ax.axvline(0.05)
        # ax.plot(0, 4, "ok")
        # assert False

    # Return the colors
    from matplotlib.colors import Normalize
    # Normalize data
    norm = Normalize(vmin=z1, vmax=z2)
    try:
        rgba_values = cmap(norm(df))
    except Exception as err:
        print(df)
        print(len(df))
        raise err

    return fig, ax, rgba_values


def scatter_color_by_value(data, ax, xvar, yvar, cvar, cmin=None, cmax=None, plot_colorbar=False):
    """
    Scatterplot, with helper to ensure range of colors.

    NOTE cmin, cmax ensures that the color mapping is fixed so that one nedpoint color represents cmin, 
    and the other cmax

    PARAMS:
    - cmin, cmax, values of cvar, which will define the min/max range for colormap. If None, then computes
    here using values of cvar [1, 99] prctile.
    """

    # For each dim make a figure, with subplot being the shapes
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl

    # ------------------------------------------------------------------
    x = data[xvar].values
    y = data[yvar].values
    c = data[cvar].values

    if cmin is None:
        cmin, cmax = np.percentile(c, [1, 99])

    # 1 .  Decide once what the colour scale should cover
    # ------------------------------------------------------------------
    norm = plt.Normalize(vmin=cmin, vmax=cmax)   # shared normalisation
    norm  = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

    # ------------------------------------------------------------------
    # 2 .  Build a palette that is light→dark
    #     ‘rocket’ is dark→light, so just reverse it
    # ------------------------------------------------------------------
    cmap = sns.color_palette("rocket", as_cmap=True).reversed()

    # ------------------------------------------------------------------
    # 2 .  Dummy mappable that *defines* the colour‑bar
    # ------------------------------------------------------------------
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])                     # nothing to plot; just carries scale

    # sns.scatterplot(x=x[::jump], y=y[::jump], c=c[::jump], alpha=0.1, size=2, ax=ax, cmap=cmap)
    sc = sns.scatterplot(
        x=x,
        y=y,
        hue=c,          # use `hue`, not `c`, for seaborn
        hue_norm=norm,          # ← identical mapping everywhere
        palette=cmap,
        alpha=0.10,
        s=20,                   # `s`, not `size`, for fixed marker size
        edgecolor="none",
        legend=False,           # suppress seaborn’s categorical legend
        ax=ax
    )

    # One colour‑bar per subplot
    if plot_colorbar:
        ax.figure.colorbar(sm, ax=ax, orientation='vertical',
                    shrink=0.8, pad=0.02, label='c value')

