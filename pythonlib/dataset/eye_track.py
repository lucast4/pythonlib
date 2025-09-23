""" 
Stores eye fixation data, usually for a single trial.
"""

import pandas as pd
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

CORRECT_FOR_TOUCHSCREEN_STROKES_LAG = True
assert CORRECT_FOR_TOUCHSCREEN_STROKES_LAG

class EyeTrack(object):
    """ 
    Stores eye fixation data, usually for a single trial, using data exported from SN
    """

    def __init__(self, D):
        """
        Loads final eye-tracking data
        """

        # (1) Load all data across all trialcode
        from pythonlib.globals import PATH_SAVE_CLUSTERFIX
        import pandas as pd
        import pickle
        animal = D.animals(True)[0]
        date = D.dates(True)[0]
        SAVEDIR = f"{PATH_SAVE_CLUSTERFIX}/FINAL_CLEAN/{animal}-{date}"

        print("Loading data pre-extracted using Session()... (fails if not found for all trials)")
        DatData = {}
        DatDffix = {}
        list_dict_event = []
        for tc in D.Dat["trialcode"]:
            SAVEDIR = f"{PATH_SAVE_CLUSTERFIX}/FINAL_CLEAN/{animal}-{date}"
            f"{SAVEDIR}/{tc}-data.pkl"
            
            with open(f"{SAVEDIR}/{tc}-data.pkl", "rb") as f:
                data = pickle.load(f)
            with open(f"{SAVEDIR}/{tc}-dict_events.pkl", "rb") as f:
                dict_events = pickle.load(f)                
            dffix = pd.read_pickle(f"{SAVEDIR}/{tc}-dffix.pkl")

            DatData[tc] = data
            DatDffix[tc] = dffix
            dict_events["trialcode"] = tc
            list_dict_event.append(dict_events)
            
        self.DatData = DatData
        self.DatDffix = DatDffix
        self.Dataset = D
        self.DatEvents = pd.DataFrame(list_dict_event)

    def plot_timecourse_single(self, ax, trialcode):
        """
        Plot timecourse, along with annotation of fixations and saccades
        Taken from: sn.events_get_clusterfix_fixation_times_and_centroids()
        """

        times_tdt, vals_tdt_calibrated, data_times_fix, data_times_sacc_pre_fix, data_centroids = self.datextract_raw(trialcode)            

        # Plot raw timecourse
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times_tdt, vals_tdt_calibrated[:,0], label="tdt_x", color="b")
        ax.plot(times_tdt, vals_tdt_calibrated[:,1], label="tdt_y", color="r")
        ax.legend()

        # Plot dots on fixation (time, loc)
        ax.plot(data_times_fix, np.stack(data_centroids)[:,0],  "ob")
        ax.plot(data_times_fix, np.stack(data_centroids)[:,1],  "or")

        # Plot lines for saccade onset
        for t in data_times_sacc_pre_fix:
            ax.axvline(t, linestyle="-", alpha=0.5, color="k")

        ax.set_title("fixations on raw (vlines=saccade; dot=fixation)")

        # - overlay trial events
        self.plotmod_overlay_trial_events(ax, trialcode)
    
    def plotwrap_timecourse(self, trialcode, plot_var="assigned_task_shape"):
        """
        Wrapper to plot timecourse and also final labeled shapes for each fixation

        PARAMS;
        - plot_var, the variable that is labeled (colored)
        Taken from: sn.events_get_clusterfix_fixation_times_and_centroids()
        """

        fig, axes1d = plt.subplots(2,1, figsize=(20,6), squeeze=False, sharex=True)

        ax = axes1d.flatten()[0]
        self.plot_timecourse_single(ax, trialcode)

        # Also put line underneath coloring the shapes
        ax = axes1d.flatten()[1]
        self.plot_timecourse_fixation_label_bars(ax, trialcode, plot_var=plot_var)

        return fig

    def plot_timecourse_fixation_label_bars(self, ax, trialcode, map_shape_to_y=None, 
                                                    map_shape_to_col=None,
                                                    yplot=0, plot_vlines=True, vlines_alpha=0.5,
                                                    plot_var="assigned_task_shape"):
        """
        Extract each fixation's assigned shape, and overlay this on any plot (ax) as horiz lines at bottom of
        plot, whos y-coord and color indicates the matched shape, and where x marks fixations, and lines mark the times 
        of ongoing fixation. 
        I used this for moment by mmoment decoding.
        PARAMS:
        - map_shape_to_y, dict mappiong from shape label to y location.
        - map_shape_to_col, dict mappiong from shape label to color (4-d array)

        From SN.beh_eye_fixation_task_shape_overlay_plot()
        """
        from pythonlib.tools.plottools import color_make_map_discrete_labels
        from pythonlib.tools.plottools import legend_add_manual

        # Extract fixation data
        dffix = self.datextract_dffix(trialcode)
        shapes_exist = dffix[plot_var].unique().tolist()

        # Decode how to map shape labels to y and to color
        if map_shape_to_y is None:
            map_shape_to_y = {sh:yplot for sh in shapes_exist}

        if map_shape_to_col is None:
            map_shape_to_col = color_make_map_discrete_labels(shapes_exist)[0]
            # map_shape_to_col["FAR_FROM_ALL_SHAPES"] = np.array([0.8, 0.8, 0.8, 1.])

        # Plot lines between each successive fixation
        dffix = dffix.sort_values("time_global")
        for i in range(len(dffix)):
            if i+1<len(dffix):    
                t1 = dffix.iloc[i]["time_global"]
                shape = dffix.iloc[i][plot_var]
                t2 = dffix.iloc[i+1]["time_global"]
                y = map_shape_to_y[shape]
                col = map_shape_to_col[shape]
                ax.plot([t1, t2], [y, y], "-x", color=col, alpha=0.8)

                if plot_vlines:
                    ax.axvline(t1, color=col, alpha=vlines_alpha, linestyle=":")

        legend_add_manual(ax, map_shape_to_col.keys(), map_shape_to_col.values())
        
        return dffix, map_shape_to_y, map_shape_to_col


    def _plot_task_image(self, ax, trialcode):
        """
        Helper to plot the task image.
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes

        strokestask = self.datasetbeh_extract_row(trialcode)["strokes_task"].values[0]
        plotDatStrokes(strokestask, ax, clean_task=True)

    def plot_image_overlay_single(self, ax, trialcode, twind, var_color = "assigned_task_shape", 
                                  alpha = 0.8, plot_raw=False):
        """
        Helper to plot overlaid fixations on top of image, coloring and texting with 
        properties.

        Optionally, plot raw coordinates (plot_raw).

        PARAMS:
        - dffix, dataframe holding fixations
        - twind, window to slice out data, by time in trial, e.g, twind = [8, 10.5]
        - var_color = "assigned_token_idx", how to color the shapes.
        --- e.g, "assigned_token_idx"

        From sn._beh_eye_fixation_overlay_plot()
        """
        from pythonlib.tools.plottools import color_make_map_discrete_labels

        dffix = self.datextract_dffix(trialcode)

        # Slice out data    
        if twind is None:
            dffix_this = dffix
        else:
            inds = (twind[0] < dffix["time_global"].values) & (dffix["time_global"].values < twind[1])
            dffix_this = dffix[inds]

        # how to color the fixations
        fixes = np.stack(dffix_this["fix_cen"])
        
        levels_color = dffix_this[var_color].tolist()
        list_idx_closest_token = dffix_this["closest_task_token_idx"].tolist()
        list_dist_to_closest_token = dffix_this["closest_task_token_dist"].tolist()
        closer_than_threshold_list = dffix_this["closer_than_threshold"].tolist()
        _, _, colors = color_make_map_discrete_labels(levels_color)

        # First, plot image
        self._plot_task_image(ax, trialcode)
        # strokestask = self.datasetbeh_extract_row(trialcode)["strokes_task"].values[0]
        # plotDatStrokes(strokestask, ax, clean_task=True)

        # Optionally, plot the raw eye tracking data
        if plot_raw:
            times_tdt, vals_tdt_calibrated, _, _, _ = self.datextract_raw(trialcode)            
            ax.plot(vals_tdt_calibrated[:,0], vals_tdt_calibrated[:,1], "-k", alpha=0.3)
            ax.scatter(vals_tdt_calibrated[:,0], vals_tdt_calibrated[:,1], c=times_tdt, alpha=alpha, marker="x")
        else:
            # Then just plot lines between fixations
            ax.plot(np.stack(fixes)[:, 0], np.stack(fixes)[:, 1], "-k", alpha=0.3)
                    
        # Second, plot fixations as circles, colored by variable of interest
        ax.scatter(np.stack(fixes)[:, 0], np.stack(fixes)[:, 1], c=colors, alpha=alpha)

        # Third, Number and other text labels
        for i, (fix_cen, idx_closest, dist_closest, closer) in enumerate(zip(fixes, list_idx_closest_token, list_dist_to_closest_token, closer_than_threshold_list)):
            ax.text(fix_cen[0], fix_cen[1], f"#{i}-dist={dist_closest:.1f}")
            if not closer:
                ax.plot(fix_cen[0], fix_cen[1], "xr")

    def datextract_dffix(self, trialcode, keep_only_if_looking_at_image=False, twind=None, prune_within_samp_laststroke=False):
        """
        Helper to return dffix for this trial, which is dataframe
        holding fixations and shape labels
        """
        dffix = self.DatDffix[trialcode].copy()

        if prune_within_samp_laststroke:
            # Then restrict to data between samp onset and end of last stroke
            delta = 0.2
            t1 = self.datextract_events_as_dict(trialcode)["samp"][0] + delta
            _, offs = self.datextract_strokes_ons_offs(trialcode)
            t2 = offs[-1] - delta
            
            dffix = dffix[(dffix["time_global"] >= t1) & (dffix["time_global"] <= t2)].reset_index(drop=True)

        if twind is not None:
            assert len(twind)==2
            t1, t2 = twind
            dffix = dffix[(dffix["time_global"] >= t1) & (dffix["time_global"] <= t2)].reset_index(drop=True)

        if keep_only_if_looking_at_image:
            dffix = dffix[dffix["assigned_token_idx"] != "FAR_FROM_ALL_SHAPES"].reset_index(drop=True)

        return dffix
    
    def datextract_raw(self, trialcode):
        """
        Return raw times and coordinates, as well as fixation and saccades times
        """
        times_tdt = self.DatData[trialcode]["times_tdt"]
        vals_tdt_calibrated = self.DatData[trialcode]["vals_tdt_calibrated"]
        data_times_fix = self.DatData[trialcode]["data_times_fix"]
        data_times_sacc_pre_fix = self.DatData[trialcode]["data_times_sacc_pre_fix"]
        data_centroids = self.DatData[trialcode]["data_centroids"]
        
        return times_tdt, vals_tdt_calibrated, data_times_fix, data_times_sacc_pre_fix, data_centroids

    def datextract_events_as_dict(self, trialcode, events=None):
        """
        Extract dict holding {event:times} for this trial.
        """
        if events is None:
            # The default events
            events = ["fixcue", "fixtch", "rulecue2", "samp", "go", 
                "first_raise", "seqon", "doneb", 
                "post", "reward_all"]

        tmp = self.DatEvents[self.DatEvents["trialcode"] == trialcode]
        assert len(tmp)==1
        return {ev:tmp[ev].values[0] for ev in events}

    def datextract_strokes(self, trialcode):
        """
        Extract strokes (beh) for this trial
        """
        strokes = self.datasetbeh_extract_row(trialcode)["strokes_beh"].values[0]
        if CORRECT_FOR_TOUCHSCREEN_STROKES_LAG:
            # Apply timing correction
            # - recorded times of strokes on touhcscreen are too late. Solve this by subtracting delta from stroke times
            delta = 0.043 # value determined using from camera hand tracking
            strokes = [s.copy() for s in strokes]
            for s in strokes:
                s[:, 2] -= delta
        return strokes
    
    def datextract_strokes_ons_offs(self, trialcode):
        """
        Extract onset and offset.
        RETURNS:
        - ons, offs
        """
        strokes = self.datextract_strokes(trialcode)
        ons = [s[0, 2] for s in strokes]
        offs = [s[-1, 2] for s in strokes]
        return ons, offs

    def datasetbeh_trialcode_to_idx(self, trialcode):
        """ Get the row in Dataset.Dat, for this trialcode
        RETURNS:
        - int
        """
        return self.datasetbeh_extract_row(trialcode).index.tolist()[0]
    
    def datasetbeh_extract_row(self, trialcode):
        """
        Return dataframe row (view). Fails except if there is exactly one row.
        """
        tmp = self.Dataset.Dat[self.Dataset.Dat["trialcode"] == trialcode]
        assert len(tmp)==1
        return tmp
    
    def get_trialcodes(self):
        """ REturn list of all trialcodes
        """
        return self.Dataset.Dat["trialcode"].tolist()
    
    def plotmod_overlay_trial_events(self, ax, trialcode, strokes_patches=True, 
            alignto_time=None, only_on_edge=None, YLIM=None, alpha = 0.15,
            which_events=("key_events_correct", "strokes"), include_text=True, 
            text_yshift = 0., xmin=None, xmax=None):
        """ Overlines trial events in vertical lines
        Time is rel trial onset (ml2 code 9)
        Run this after everything else, so get propoer YLIM.
        PARAMS:
        - strokes_patches, bool, then fills in patches for each stroke. otherwise places vert lines
        at onset and ofset.
        - only_on_edge, whethe and how to plot on edge, without blocking the figure, either:
        --- None, skip
        --- "top", or "bottom"
        - YLIM, to explicitly define, useful if doing one per raster line.
        - which_events, list of str, control whhic to plot
        - text_yshift, shfit text by this amnt.
        """

        # Strokes should be lower alpha, so not obscure spikes.
        alpha_st = alpha
        if alpha_st>0.2:
            alpha_st = 0.2 

        for ev in which_events:
            assert ev in ["key_events_correct", "strokes"], "doesnt exist..."

        # list_codes = []
        times_codes = np.array([])
        names_codes = [] 
        colors_codes = []

        ###### 2) key events, determeined using actual voltage clock signals or touch, etc.
        if "key_events_correct" in which_events:
            color_map = {
                "fixcue":"g",
                "fixtch":"m",
                "samp":"r",
                "go":"b",
                "first_raise":"c",
                "seqon":"y",
                "doneb":"g",
                "post":"m", 
                "rew":"k",
                "reward_all":"b"
            }
            dict_events = self.datextract_events_as_dict(trialcode)
            
            # collect all event times into single arrays
            for ev in dict_events:
                for t in dict_events[ev]:
                    times_codes = np.append(times_codes, t)
                    names_codes.append(ev)
                    if ev in color_map:
                        colors_codes.append(color_map[ev])
                    else:
                        # use a default color
                        colors_codes.append("r")            

        ###### 3) Also include times of strokes
        if strokes_patches==False and "strokes" in which_events:
            ons, offs = self.datextract_strokes_ons_offs(trialcode)
            times_codes = np.append(times_codes, ons)
            times_codes = np.append(times_codes, offs)
            names_codes.extend(["Son" for _ in range(len(ons))])
            colors_codes.extend(["b" for _ in range(len(ons))])
            names_codes.extend(["Soff" for _ in range(len(offs))])
            colors_codes.extend(["m" for _ in range(len(offs))])
        
        if alignto_time is not None:
            times_codes = times_codes - alignto_time

        if not YLIM:
            YLIM = ax.get_ylim()

        ############## Plot marker for each event
        for time, name, col in zip(times_codes, names_codes, colors_codes):
            if np.isnan(time):
                continue
            if xmin is not None and time<xmin:
                continue
            if xmax is not None and  time>xmax:
                continue
            if only_on_edge:
                if only_on_edge=="top":
                    ax.plot(time, YLIM[1], "v", color=col, alpha=alpha)
                    y_text = YLIM[1]
                elif only_on_edge=="bottom":
                    ax.plot(time, YLIM[0], "^", color=col, alpha=alpha)
                    y_text = YLIM[0]
                else:
                    assert False
            else:
                ax.axvline(time, color=col, ls="--", alpha=0.7)
                y_text = YLIM[0]
            if include_text:
                y_text = y_text + text_yshift
                ax.text(time, y_text, name, rotation="vertical", fontsize=10, color="m", alpha=0.5)

        # color in stroke times
        if strokes_patches and "strokes" in which_events:
            from matplotlib.patches import Rectangle
            ons, offs = self.datextract_strokes_ons_offs(trialcode)
            if alignto_time:
                ons = [o - alignto_time for o in ons]
                offs = [o - alignto_time for o in offs]

            for on, of in zip(ons, offs):
                if only_on_edge:
                    if only_on_edge=="top":
                        ax.hlines(YLIM[1], on, of, color="r", alpha=alpha)
                    elif only_on_edge=="bottom":
                        ax.hlines(YLIM[0], on, of, color="r", alpha=alpha)
                    else:
                        assert False
                        # ax., YLIM[1], "v", color=col)
                else:
                    rect = Rectangle((on, YLIM[0]), of-on, YLIM[1]-YLIM[0], 
                        linewidth=1, edgecolor='r',facecolor='r', alpha=alpha_st)
                    ax.add_patch(rect)

    def grammar_plot_timecourse_abstract(self, ax, trialcode):
        """
        Plot the timecourse of fixations and strokes, in an abstract manner, where the
        y axis is rank (in correct sequence) and x axis is time.

        Useful for visualizing how eye movements relate to ongoing strokes.

        Copied from: Session().beh_eye_fixation_grammar_summary_plot
        """
        # import pandas as pd
        from pythonlib.tools.plottools import color_make_map_discrete_labels
        import matplotlib.pyplot as plt

        D = self.Dataset

        # Get fixtaion data
        dffix = self.datextract_dffix(trialcode)

        # Get the correct sequence
        ind_dat = self.datasetbeh_trialcode_to_idx(trialcode)
        tokens_correct_order = D.grammarparses_task_tokens_correct_order_sequence(ind_dat, PLOT=False)

        # map from token index, to its rank (in correct sequence)
        map_taskidx_to_rank = {}
        for rank, tok in enumerate(tokens_correct_order):
            idx_orig = tok["ind_taskstroke_orig"]
            map_taskidx_to_rank[idx_orig] = rank
        dffix["assigned_tok_rank_correct"] = [map_taskidx_to_rank[i] if i!="FAR_FROM_ALL_SHAPES" else -1 for i in dffix["assigned_token_idx"]]

        # To make sure plotted line stays on the fixated shape until right before next fixation (i.e,, before saccade), fake a dataframe.
        dffixtmp = dffix.iloc[1:].copy()
        dffixtmp["time_global"] -= 0.03
        dffixtmp["assigned_tok_rank_correct"] = dffix.iloc[:-1]["assigned_tok_rank_correct"].tolist()
        dffix_fake = pd.concat([dffix, dffixtmp], axis=0)
        dffix_fake = dffix_fake.sort_values("time_global", axis=0)

        # To plot strokes
        ons, offs = self.datextract_strokes_ons_offs(trialcode)
        ranks = list(range(len(ons)))

        # To color stroke by chunk
        map_shape_to_color, _, _ = color_make_map_discrete_labels(D.taskclass_shapes_extract_unique_alltrials())
        shapes = [tok["shape"] for tok in tokens_correct_order]
        colors_strokes = [map_shape_to_color[sh] for sh in shapes]

        # TODO: For each fixation, label it something useful
        # same chunk
        # next 
        # Mark events

        # Make plot where shapes are ordered by the drawing order, and the fixations and drawings are plotted overlaid on that
        ax.grid(axis='y')

        # Overlay strokes
        assert len(ons)==len(offs)==len(ranks)
        for _on, _off, _rank, _col in zip(ons, offs, ranks, colors_strokes):
            # ax.plot([_on, _off], [_rank-0.2, _rank-0.2], "k-", linewidth=8, alpha=0.5)
            ax.plot([_on, _off], [_rank, _rank], "-", color=_col, linewidth=8, alpha=0.7)
            # ax.axvline(_on)
            # ax.axvline(_off)

        # Overlay fixations
        ax.plot(dffix_fake["time_global"], dffix_fake["assigned_tok_rank_correct"], "-k", alpha=0.4)
        # ax.scatter(dffix["time_global"], dffix["assigned_tok_rank_correct"], c=dffix["assigned_tok_rank_correct"])
        colors = [map_shape_to_color[sh] if sh !="FAR_FROM_ALL_SHAPES" else "w" for sh in dffix["assigned_task_shape"]]
        ax.scatter(dffix["time_global"], dffix["assigned_tok_rank_correct"], c=colors)

        self.plotmod_overlay_trial_events(ax, trialcode, only_on_edge="bottom")

        # legend
        from pythonlib.tools.plottools import legend_add_manual
        legend_add_manual(ax, map_shape_to_color.keys(), map_shape_to_color.values())


