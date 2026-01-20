# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import AnchoredText

from netCDF4 import num2date

from ..calibrate_func.evaluate_metrics import EvaluationMetric
from ..params_func.params_set import *
from .plot_utilities import *

## ------------------------ plot performance ------------------------
def plot_VIC_performance(cali_result, verify_result):
    """
    Plot the performance of VIC model calibration and verification.

    Parameters
    ----------
    cali_result : pandas.DataFrame
        Calibration results with observed and simulated streamflow.
    verify_result : pandas.DataFrame
        Verification results with observed and simulated streamflow.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing all subplots.
    axes : list of matplotlib.axes.Axes
        The list of axes for each subplot.
    """
    # fig set
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.98,
        hspace=0.15,
        wspace=0.3,
    )  # wspace=0.15, wspace=0.3
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])

    # plot timeseries
    ax1.plot(
        list(range(len(cali_result.index))),
        cali_result["obs_cali discharge(m3/s)"],
        label="obs",
        color="black",
        linestyle="-",
        linewidth=1,
    )
    ax1.plot(
        list(range(len(cali_result.index))),
        cali_result["sim_cali discharge(m3/s)"],
        label="sim",
        color="red",
        linestyle="",
        marker=".",
        markersize=1,
        linewidth=1,
    )
    ax1.set_xticks(
        list(range(len(cali_result)))[:: int(len(cali_result) / 5)],
        cali_result.index[:: int(len(cali_result) / 5)],
    )
    ax1.set_xlim(0, len(cali_result.index))
    ax1.set_ylabel("Streamflow (m$^3$/s)")

    ax3.plot(
        list(range(len(verify_result.index))),
        verify_result["obs_verify discharge(m3/s)"],
        label="obs",
        color="black",
        linestyle="-",
        linewidth=1,
    )
    ax3.plot(
        list(range(len(verify_result.index))),
        verify_result["sim_verify discharge(m3/s)"],
        label="sim",
        color="red",
        linestyle="",
        marker=".",
        markersize=1,
        linewidth=1,
    )
    ax3.set_xticks(
        list(range(len(verify_result)))[:: int(len(verify_result) / 5)],
        verify_result.index[:: int(len(verify_result) / 5)],
    )
    ax3.set_xlim(0, len(verify_result.index))
    ax3.set_ylabel("Streamflow (m$^3$/s)")
    ax3.set_xlabel("Date")

    # cal threshold for high flow and low flow
    total_flow = np.concatenate(
        [
            cali_result["obs_cali discharge(m3/s)"].values,
            verify_result["obs_verify discharge(m3/s)"].values,
        ]
    )
    lowflow_threshold = np.percentile(total_flow, 30)
    highflow_threshold = np.percentile(total_flow, 70)

    # plot calibration: total
    ax2.scatter(
        cali_result["obs_cali discharge(m3/s)"],
        cali_result["sim_cali discharge(m3/s)"],
        color="red",
        s=1,
    )
    p_cali = np.polyfit(
        cali_result["obs_cali discharge(m3/s)"],
        cali_result["sim_cali discharge(m3/s)"],
        deg=1,
        rcond=None,
        full=False,
        w=None,
        cov=False,
    )
    ax2.plot(
        np.arange(0, ax1.get_ylim()[1], 1),
        np.polyval(p_cali, np.arange(0, ax1.get_ylim()[1], 1)),
        color="black",
        linestyle="-",
        linewidth=1,
        label=f"total flow: y = {p_cali[0]:.2f}x {'+' if p_cali[1] >= 0 else '-'} {abs(p_cali[1]):.2f}",
    )

    # plot calibration: low flow
    cali_lowflow_index = cali_result["obs_cali discharge(m3/s)"] <= lowflow_threshold
    p_cali_lowflow = np.polyfit(
        cali_result["obs_cali discharge(m3/s)"][cali_lowflow_index],
        cali_result["sim_cali discharge(m3/s)"][cali_lowflow_index],
        deg=1,
        rcond=None,
        full=False,
        w=None,
        cov=False,
    )
    ax2.plot(
        np.arange(0, ax1.get_ylim()[1], 1),
        np.polyval(p_cali_lowflow, np.arange(0, ax1.get_ylim()[1], 1)),
        color="darkblue",
        linestyle="-",
        linewidth=1,
        label=f"low flow: y = {p_cali_lowflow[0]:.2f}x {'+' if p_cali_lowflow[1] >= 0 else '-'} {abs(p_cali_lowflow[1]):.2f}",
    )

    # plot calibration: high flow
    cali_highflow_index = cali_result["obs_cali discharge(m3/s)"] >= highflow_threshold
    p_cali_highflow = np.polyfit(
        cali_result["obs_cali discharge(m3/s)"][cali_highflow_index],
        cali_result["sim_cali discharge(m3/s)"][cali_highflow_index],
        deg=1,
        rcond=None,
        full=False,
        w=None,
        cov=False,
    )
    ax2.plot(
        np.arange(0, ax1.get_ylim()[1], 1),
        np.polyval(p_cali_highflow, np.arange(0, ax1.get_ylim()[1], 1)),
        color="darkgreen",
        linestyle="-",
        linewidth=1,
        label=f"high flow: y = {p_cali_highflow[0]:.2f}x {'+' if p_cali_highflow[1] >= 0 else '-'} {abs(p_cali_highflow[1]):.2f}",
    )

    # set ax2
    ax2.set_xlim(0, ax1.get_ylim()[1])
    ax2.set_ylim(0, ax1.get_ylim()[1])
    ax2.xaxis.set_major_locator(plt.LinearLocator(numticks=6))
    ax2.yaxis.set_major_locator(plt.LinearLocator(numticks=6))
    ax2.set_ylabel("Simulated streamflow (m$^3$/s)")

    # plot verification: total
    ax4.scatter(
        verify_result["obs_verify discharge(m3/s)"],
        verify_result["sim_verify discharge(m3/s)"],
        color="red",
        s=1,
    )
    p_verify = np.polyfit(
        verify_result["obs_verify discharge(m3/s)"],
        verify_result["sim_verify discharge(m3/s)"],
        deg=1,
        rcond=None,
        full=False,
        w=None,
        cov=False,
    )
    ax4.plot(
        np.arange(0, ax3.get_ylim()[1], 1),
        np.polyval(p_verify, np.arange(0, ax3.get_ylim()[1], 1)),
        color="black",
        linestyle="-",
        linewidth=1,
        label=f"total flow: y = {p_verify[0]:.2f}x {'+' if p_verify[1] >= 0 else '-'} {abs(p_verify[1]):.2f}",
    )

    # plot verification: low flow
    verify_lowflow_index = (
        verify_result["obs_verify discharge(m3/s)"] <= lowflow_threshold
    )
    p_verify_lowflow = np.polyfit(
        verify_result["obs_verify discharge(m3/s)"][verify_lowflow_index],
        verify_result["sim_verify discharge(m3/s)"][verify_lowflow_index],
        deg=1,
        rcond=None,
        full=False,
        w=None,
        cov=False,
    )
    ax4.plot(
        np.arange(0, ax3.get_ylim()[1], 1),
        np.polyval(p_verify_lowflow, np.arange(0, ax3.get_ylim()[1], 1)),
        color="darkblue",
        linestyle="-",
        linewidth=1,
        label=f"low flow: y = {p_verify_lowflow[0]:.2f}x {'+' if p_verify_lowflow[1] >= 0 else '-'} {abs(p_verify_lowflow[1]):.2f}",
    )

    # plot verification: high flow
    verify_highflow_index = (
        verify_result["obs_verify discharge(m3/s)"] >= highflow_threshold
    )
    p_verify_highflow = np.polyfit(
        verify_result["obs_verify discharge(m3/s)"][verify_highflow_index],
        verify_result["sim_verify discharge(m3/s)"][verify_highflow_index],
        deg=1,
        rcond=None,
        full=False,
        w=None,
        cov=False,
    )
    ax4.plot(
        np.arange(0, ax3.get_ylim()[1], 1),
        np.polyval(p_verify_highflow, np.arange(0, ax3.get_ylim()[1], 1)),
        color="darkgreen",
        linestyle="-",
        linewidth=1,
        label=f"high flow: y = {p_verify_highflow[0]:.2f}x {'+' if p_verify_highflow[1] >= 0 else '-'} {abs(p_verify_highflow[1]):.2f}",
    )

    # ax4 set
    ax4.set_xlim(0, ax3.get_ylim()[1])
    ax4.set_ylim(0, ax3.get_ylim()[1])
    ax4.xaxis.set_major_locator(plt.LinearLocator(numticks=6))
    ax4.yaxis.set_major_locator(plt.LinearLocator(numticks=6))
    ax4.set_xlabel("Observed streamflow (m$^3$/s)")
    ax4.set_ylabel("Simulated streamflow (m$^3$/s)")

    # legend set
    ax1.legend(loc="upper left", prop={"size": 12, "family": "Arial"})
    ax2.legend(loc="upper left", prop={"size": 12, "family": "Arial"})
    ax4.legend(loc="upper left", prop={"size": 12, "family": "Arial"})

    # calcluate metrics
    em_cali = EvaluationMetric(
        sim=cali_result["sim_cali discharge(m3/s)"],
        obs=cali_result["obs_cali discharge(m3/s)"],
    )
    em_verify = EvaluationMetric(
        sim=verify_result["sim_verify discharge(m3/s)"],
        obs=verify_result["obs_verify discharge(m3/s)"],
    )

    # add figure text: metrics
    at = AnchoredText(
        f"KGE: {em_cali.KGE():.2f}\nNSE: {em_cali.NSE():.2f}\nPBIAS: {em_cali.PBias():.2f}%",
        prop={"size": 12, "family": "Arial", "linespacing": 1.5},
        frameon=True,
        loc="upper right",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    at.patch.set_edgecolor("lightgray")
    ax1.add_artist(at)

    at = AnchoredText(
        f"KGE: {em_verify.KGE():.2f}\nNSE: {em_verify.NSE():.2f}\nPBIAS: {em_verify.PBias():.2f}%",
        prop={"size": 12, "family": "Arial", "linespacing": 1.5},
        frameon=True,
        loc="upper right",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    at.patch.set_edgecolor("lightgray")
    ax3.add_artist(at)

    axes = [ax1, ax2, ax3, ax4]

    return fig, axes


def taylor_diagram(
    obs,
    models,
    model_names,
    names_ha,
    names_va,
    model_colors=None,
    model_markers=None,
    title="Standard Taylor Diagram",
    fig=None,
    ax=None,
    add_text=True,
):
    """
    Create a Taylor diagram to compare multiple models against observations.

    Parameters
    ----------
    obs : ndarray
        A 1D array of observed data values.
    models : list of ndarray
        A list of 1D arrays, each representing model data to compare against the observations.
    model_names : list of str
        A list of model names, corresponding to each model in `models`.
    names_ha : list of str
        Horizontal alignment for each model name's position in the plot.
    names_va : list of str
        Vertical alignment for each model name's position in the plot.
    model_colors : list of str, optional
        A list of colors for each model's data points on the diagram. Default is None, which uses a color map.
    title : str, optional
        The title of the diagram. Default is "Standard Taylor Diagram".
    fig : matplotlib.figure.Figure, optional
        An existing figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        An existing axes object to plot on. If None, a new polar subplot is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the Taylor diagram.
    ax : matplotlib.axes.Axes
        The axes containing the Taylor diagram.

    Notes
    -----
    This function computes the standard deviation and correlation coefficient for each model
    relative to the observed data and plots the results on a polar plot. The diagram includes:
    - Observation point at the center (standard deviation = 1, correlation = 1).
    - Models plotted at angles based on the correlation coefficient and at radii based on the standard deviation.
    - Contour lines representing the RMSD (Root Mean Square Deviation).
    - Arcs for standard deviation levels and radial lines for correlation levels.

    examples:
    
    simulated_datasets = [simulated_dataset_12km, simulated_dataset_8km, simulated_dataset_6km]
    params_dataset_level0_sets = [params_dataset_level0_12km, params_dataset_level0_8km, params_dataset_level0_6km]
    params_dataset_level1_sets = [params_dataset_level1_12km, params_dataset_level1_8km, params_dataset_level1_6km]
    model_names = ["12km ", "8km ", "6km "]
    model_colors = ["red", "blue", "green"]
    cali_names_ha = ["left", "right", "left"]  # {'center', 'right', 'left'}
    cali_names_va = ["bottom", "top", "bottom"]  # {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
    verify_names_ha = ["left", "right", "left"]
    verify_names_va = ["bottom", "top", "bottom"]
    
    obs_cali = cali_result_12km["obs_cali discharge(m3/s)"].values
    obs_verify = verify_result_12km["obs_verify discharge(m3/s)"].values
    obs_total = np.concatenate([obs_cali, obs_verify])
    models_cali = [cali_result["sim_cali discharge(m3/s)"].values for cali_result in cali_results]
    models_verify = [verify_result["sim_verify discharge(m3/s)"].values for verify_result in verify_results]
    models_total = [np.concatenate([models_cali[i], models_verify[i]]) for i in range(len(models_cali))]
    
    fig_taylor = plt.figure(figsize=(12, 6))
    fig_taylor.subplots_adjust(left=0.08, right=0.92, bottom=0.01, top=0.9, wspace=0.3)
    ax1 = fig_taylor.add_subplot(121, projection='polar')
    ax2 = fig_taylor.add_subplot(122, projection='polar')
    
    fig_taylor, ax1 = taylor_diagram(obs_cali, models_cali, model_names, cali_names_ha, cali_names_va, model_colors=model_colors, title="(a) Calibration", fig=fig_taylor, ax=ax1)
    fig_taylor, ax2 = taylor_diagram(obs_verify, models_verify, model_names, verify_names_ha, verify_names_va, model_colors=model_colors, title="(b) Verification", fig=fig_taylor, ax=ax2)
    
    """
    # Normalize data: Set the standard deviation of observed data to 1
    obs_std = np.std(obs)
    obs_norm = obs / obs_std
    models_norm = [model / obs_std for model in models]

    # Calculate statistics: Standard deviation and correlation coefficient for each model
    model_stds = [np.std(model) for model in models_norm]
    model_corrs = [np.corrcoef(obs_norm, model)[0, 1] for model in models_norm]

    # set r
    r_max = 1.4
    r_interval = 0.2

    # Create a polar plot
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.95)
    if ax is None:
        ax = fig.add_subplot(111, projection="polar")

    # set grid
    ax.grid(False)

    # Set the starting direction and angle range of the polar plot
    ax.set_theta_zero_location("N")  # 0 degrees at the top
    ax.set_theta_direction(-1)  # Clockwise direction
    ax.set_thetamin(0)  # Minimum angle 0 degrees
    ax.set_thetamax(90)  # Maximum angle 90 degrees

    # Plot the observation point (location: std=1, correlation=1, angle=0 degrees)
    ax.scatter(np.pi / 2, 1, color="k", s=100, label="Observation", zorder=10)
    ax.text(
        np.pi / 2,
        1,
        "REF",
        ha="right",
        va="bottom",
        color="k",
        fontdict={"family": "Arial", "size": 12, "weight": "bold"},
    )

    # Plot model points (angle = arccos(correlation), radius = model standard deviation)
    if model_colors is None:
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    if model_markers is None:
        model_markers = ['o' for _ in range(len(models))]
        
    pad_theta = 0.01
    pad_r = 0.01
    for i, (corr, std) in enumerate(zip(model_corrs, model_stds)):
        theta = np.pi / 2 - np.arccos(corr)  # Convert correlation to radians (0 to π/2)
        ax.scatter(
            theta,
            std,
            color=model_colors[i],
            s=100,
            label=model_names[i],
            zorder=5,
            alpha=0.8,
            edgecolors="white",
            marker=model_markers[i],
        )
        
        if add_text:
            ax.text(
                theta + pad_theta,
                std + pad_r,
                model_names[i],
                ha=names_ha[i],
                va=names_va[i],
                color=model_colors[i],
                fontdict={"family": "Arial", "size": 12, "weight": "bold"},
                zorder=10,
            )
        else:
            plt.legend()

    # Draw standard deviation arcs
    for r in np.arange(0, r_max, r_interval):
        color = "k" if r == 1 else "gray"
        linestyle = "-." if r == 1 else "--"
        linewidth = 1 if r == 1 else 0.5
        alpha = 0.9 if r == 1 else 0.7
        ax.plot(
            np.linspace(0, np.pi / 2, 100),
            [r] * 100,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    # Draw correlation radial lines (angle: arccos([1.0, 0.9, ..., 0.0]))
    radial_lines = np.concatenate(
        [np.arange(0, 1.0, 0.1), np.array([0.95, 0.99, 1.00])]
    )
    for corr in radial_lines:
        theta = np.pi / 2 - np.arccos(corr)
        ax.plot(
            [theta, theta],
            [0, r_max],
            color="grey",
            linestyle="-",
            linewidth=0.8,
            alpha=0.3,
        )

    # Add RMSD contours
    theta_grid = np.linspace(0, np.pi, 100)
    r_grid = np.linspace(0, r_max, 100)
    Theta, R = np.meshgrid(theta_grid, r_grid)
    RMSD = np.sqrt(1 + R**2 - 2 * R * np.cos(Theta - np.pi / 2))
    contours = ax.contour(
        Theta,
        R,
        RMSD,
        levels=np.arange(0, r_max, r_interval),
        colors="blue",
        linestyles=":",
        linewidths=1,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    # Set polar axis labels and ticks
    ax.set_rlabel_position(90)  # Radius labels at 90 degrees
    xticks = np.pi / 2 - np.arccos(np.flip(radial_lines))
    ax.set_xticks(xticks)  # Angle ticks for correlation
    ax.set_xticklabels(
        [
            f"{x:.2f}" if (x > 0.9) and (x != 1) else f"{x:.1f}"
            for x in np.flip(radial_lines)
        ],
        fontproperties={"family": "Arial", "size": 12},
    )

    ax.set_yticks(
        np.arange(0, r_max, r_interval)
    )  # Radius ticks for standard deviation
    ax.set_yticklabels(
        [f"{y:.1f}" for y in np.arange(0, r_max, r_interval)],
        fontproperties={"family": "Arial", "size": 12},
    )

    ax.set_xlabel(
        "Standard Deviation", fontdict={"family": "Arial", "size": 12}
    )  # , labelpad=20
    ax.set_ylabel("Standard Deviation", fontdict={"family": "Arial", "size": 12})

    # set cc labels
    ax.text(
        np.pi / 4,
        r_max,
        "Correlation Coefficient",
        ha="left",
        va="bottom",
        fontsize=12,
        color="black",
        rotation=-45,
    )

    # # set rticks
    # r_ticks = np.arange(0, r_max, r_interval)
    # r_ticks_text = [f'{y:.1f}' for y in r_ticks] # r
    # r_ticks_text[r_ticks_text.index("1.0")] = "REF"

    # # ax.text(ticks_ceta, ticks_r, ticks_text)
    # for r, text in zip(r_ticks, r_ticks_text):
    #     ax.text(np.pi/2, r, text, ha='center', va='baseline', fontsize=12, color='black')

    # Set title and legend
    ax.set_title(title, pad=20)
    # ax.legend(loc='upper right')  # bbox_to_anchor=(1.15, 1),

    return fig, ax


def plot_multimodel_comparison_scatter(
    obs_total, models_total, model_names, model_colors=None
):
    """
    Plot a comparison scatter plot for multiple models against observed data,
    categorized into total, low flow, and high flow conditions.

    Parameters
    ----------
    obs_total : numpy.ndarray
        A 1D array of observed streamflow values.
    models_total : list of numpy.ndarray
        A list of 1D arrays, where each array contains simulated streamflow values for a model.
    model_names : list of str
        A list of names corresponding to the models in `models_total`.
    model_colors : list of str, optional
        A list of colors for each model's data points. If not provided, default colors are used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    axes : numpy.ndarray
        An array of axes corresponding to the subplots.
    """
    # threshold
    lowflow_threshold = np.percentile(obs_total, 30)
    highflow_threshold = np.percentile(obs_total, 70)
    lowflow_index = obs_total <= lowflow_threshold
    highflow_index = obs_total >= highflow_threshold

    if model_colors is None:
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    # plot
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        gridspec_kw={
            "left": 0.08,
            "right": 0.95,
            "bottom": 0.15,
            "top": 0.9,
            "wspace": 0.2,
        },
    )

    # set lim
    xylim_total = (
        min(np.min(obs_total), min([min(model) for model in models_total])),
        max(np.max(obs_total), max([max(model) for model in models_total])),
    )
    xylim_lowflow = (
        min(
            np.min(obs_total[lowflow_index]),
            min([min(model[lowflow_index]) for model in models_total]),
        ),
        max(
            np.max(obs_total[lowflow_index]),
            max([max(model[lowflow_index]) for model in models_total]),
        ),
    )
    xylim_highflow = (
        min(
            np.min(obs_total[highflow_index]),
            min([min(model[highflow_index]) for model in models_total]),
        ),
        max(
            np.max(obs_total[highflow_index]),
            max([max(model[highflow_index]) for model in models_total]),
        ),
    )

    axes[0].set_xlim(xylim_total)
    axes[0].set_ylim(xylim_total)
    axes[1].set_xlim(xylim_lowflow)
    axes[1].set_ylim(xylim_lowflow)
    axes[2].set_xlim(xylim_highflow)
    axes[2].set_ylim(xylim_highflow)

    axes[0].plot(
        np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1),
        np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1),
        "grey",
        alpha=0.5,
        linestyle="--",
        linewidth=1,
    )
    axes[1].plot(
        np.arange(axes[1].get_xlim()[0], axes[1].get_xlim()[1], 1),
        np.arange(axes[1].get_xlim()[0], axes[1].get_xlim()[1], 1),
        "grey",
        alpha=0.5,
        linestyle="--",
        linewidth=1,
    )
    axes[2].plot(
        np.arange(axes[2].get_xlim()[0], axes[2].get_xlim()[1], 1),
        np.arange(axes[2].get_xlim()[0], axes[2].get_xlim()[1], 1),
        "grey",
        alpha=0.5,
        linestyle="--",
        linewidth=1,
    )

    for i, (model, model_name, model_color) in enumerate(
        zip(models_total, model_names, model_colors)
    ):
        axes[0].scatter(
            obs_total,
            model,
            facecolors="none",
            edgecolor=model_color,
            s=10,
            linewidth=1,
            label=None,
            alpha=0.8,
        )
        axes[1].scatter(
            obs_total[lowflow_index],
            model[lowflow_index],
            facecolors="none",
            edgecolor=model_color,
            s=10,
            linewidth=1,
            label=None,
            alpha=0.8,
        )
        axes[2].scatter(
            obs_total[highflow_index],
            model[highflow_index],
            facecolors="none",
            edgecolor=model_color,
            s=10,
            linewidth=1,
            label=None,
            alpha=0.8,
        )

        p_total = np.polyfit(
            obs_total, model, deg=1, rcond=None, full=False, w=None, cov=False
        )
        axes[0].plot(
            np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1),
            np.polyval(
                p_total, np.arange(axes[0].get_xlim()[0], axes[0].get_xlim()[1], 1)
            ),
            color=model_color,
            linestyle="-",
            linewidth=1,
            label=f"{model_name}: y = {p_total[0]:.2f}x {'+' if p_total[1] >= 0 else '-'} {abs(p_total[1]):.2f}",
        )

        p_lowflow = np.polyfit(
            obs_total[lowflow_index],
            model[lowflow_index],
            deg=1,
            rcond=None,
            full=False,
            w=None,
            cov=False,
        )
        axes[1].plot(
            np.arange(axes[1].get_xlim()[0], axes[1].get_xlim()[1], 1),
            np.polyval(
                p_lowflow, np.arange(axes[1].get_xlim()[0], axes[1].get_xlim()[1], 1)
            ),
            color=model_color,
            linestyle="-",
            linewidth=1,
            label=f"{model_name}: y = {p_lowflow[0]:.2f}x {'+' if p_lowflow[1] >= 0 else '-'} {abs(p_lowflow[1]):.2f}",
        )

        p_highflow = np.polyfit(
            obs_total[highflow_index],
            model[highflow_index],
            deg=1,
            rcond=None,
            full=False,
            w=None,
            cov=False,
        )
        axes[2].plot(
            np.arange(axes[2].get_xlim()[0], axes[2].get_xlim()[1], 1),
            np.polyval(
                p_highflow, np.arange(axes[2].get_xlim()[0], axes[2].get_xlim()[1], 1)
            ),
            color=model_color,
            linestyle="-",
            linewidth=1,
            label=f"{model_name}: y = {p_highflow[0]:.2f}x {'+' if p_highflow[1] >= 0 else '-'} {abs(p_highflow[1]):.2f}",
        )

    axes[0].set_ylabel("Simulated streamflow (m$^3$/s)")
    [ax.set_xlabel("Observed streamflow (m$^3$/s)") for ax in axes]

    axes[0].set_title("Total flow")
    axes[1].set_title("Low flow")
    axes[2].set_title("High flow")

    axes[0].legend(loc="upper right", prop={"size": 10, "family": "Arial"})
    axes[1].legend(loc="upper right", prop={"size": 10, "family": "Arial"})
    axes[2].legend(loc="upper right", prop={"size": 10, "family": "Arial"})

    axes[0].annotate(
        "(a)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=14, fontweight="bold"
    )
    axes[1].annotate(
        "(b)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=14, fontweight="bold"
    )
    axes[2].annotate(
        "(c)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=14, fontweight="bold"
    )

    return fig, axes


def plot_multimodel_comparison_distributed_OUTPUT(
    cali_results,
    verify_results,
    simulated_datasets,
    MeteForcing_df,
    model_names,
    model_colors,
    event_period,
    rising_period,
    recession_period,
):
    """
    Plot a comparison of model simulations and observations for multiple models with distributed surface flow and baseflow.

    Parameters
    ----------
    cali_results : list of pandas.DataFrame
        Calibration results containing observed and simulated discharge for calibration period.
    verify_results : list of pandas.DataFrame
        Verification results containing observed and simulated discharge for verification period.
    simulated_datasets : list of netCDF4.Dataset
        Simulated datasets containing runoff and baseflow values.
    MeteForcing_df : pandas.DataFrame
        Meteorological forcing data, including precipitation.
    model_names : list of str
        List of model names for labeling the simulations.
    model_colors : list of str
        List of colors corresponding to each model for plotting.
    event_period : tuple of str
        Start and end dates for the event period.
    rising_period : tuple of str
        Start and end dates for the rising period of the hydrograph.
    recession_period : tuple of str
        Start and end dates for the recession period of the hydrograph.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plotted comparison.
    """
    # get data
    obs_cali = cali_results[0]["obs_cali discharge(m3/s)"].values
    obs_verify = verify_results[0]["obs_verify discharge(m3/s)"].values
    obs_total = np.concatenate([obs_cali, obs_verify])

    models_cali = [
        cali_result["sim_cali discharge(m3/s)"].values for cali_result in cali_results
    ]
    models_verify = [
        verify_result["sim_verify discharge(m3/s)"].values
        for verify_result in verify_results
    ]
    models_total = [
        np.concatenate([models_cali[i], models_verify[i]])
        for i in range(len(models_cali))
    ]

    date_total = np.concatenate([cali_results[0].index, verify_results[0].index])
    obs_total_df = pd.DataFrame(
        obs_total, index=date_total, columns=["obs_total discharge(m3/s)"]
    )
    models_total_df = [
        pd.DataFrame(
            models_total[i],
            index=date_total,
            columns=[f"sim_total discharge(m3/s)_{model_names[i].strip()}"],
        )
        for i in range(len(models_total))
    ]
    all_df = pd.concat([obs_total_df] + models_total_df, axis=1)
    all_df.index = pd.to_datetime(all_df.index)
    all_df_event = all_df.loc[event_period[0] : event_period[1], :]
    rising_df_event = all_df.loc[rising_period[0] : rising_period[1], :]
    recession_df_event = all_df.loc[recession_period[0] : recession_period[1], :]

    # time
    datasets_times = simulated_datasets[0].variables["time"]
    datasets_dates = num2date(
        datasets_times[:], units=datasets_times.units, calendar=datasets_times.calendar
    )
    datasets_datetime_index = pd.to_datetime(
        [date.strftime("%Y-%m-%d %H:%M:%S") for date in datasets_dates]
    )

    # fig set
    fig = plt.figure(figsize=(12, 8))

    outer_gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        left=0.08,
        right=0.93,
        bottom=0.08,
        top=0.98,
        height_ratios=[3, 4],
        hspace=0.25,
    )

    ax1 = fig.add_subplot(outer_gs[0])

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer_gs[1],
        hspace=0.05,
        wspace=0.15,
        height_ratios=[16, 1],
    )

    left_gs = gridspec.GridSpecFromSubplotSpec(
        3, 5, subplot_spec=inner_gs[0, 0], hspace=0.05, wspace=0.1
    )

    right_gs = gridspec.GridSpecFromSubplotSpec(
        3, 5, subplot_spec=inner_gs[0, 1], hspace=0.05, wspace=0.1
    )

    ax_left_cb = fig.add_subplot(inner_gs[1, 0])
    ax_right_cb = fig.add_subplot(inner_gs[1, 1])

    axes_12km_rising = [fig.add_subplot(left_gs[0, i]) for i in range(5)]
    axes_8km_rising = [fig.add_subplot(left_gs[1, i]) for i in range(5)]
    axes_6km_rising = [fig.add_subplot(left_gs[2, i]) for i in range(5)]

    axes_12km_recession = [fig.add_subplot(right_gs[0, i]) for i in range(5)]
    axes_8km_recession = [fig.add_subplot(right_gs[1, i]) for i in range(5)]
    axes_6km_recession = [fig.add_subplot(right_gs[2, i]) for i in range(5)]

    all_axes_rising = axes_12km_rising + axes_8km_rising + axes_6km_rising
    all_axes_recession = axes_12km_recession + axes_8km_recession + axes_6km_recession
    all_axes_rising_recession = all_axes_rising + all_axes_recession

    # plot events
    ax1.plot(
        list(range(len(all_df_event.index))),
        all_df_event["obs_total discharge(m3/s)"],
        label="obs",
        color="black",
        linestyle="-",
        linewidth=1,
        zorder=5,
    )
    ax1_twinx = ax1.twinx()
    ax1_twinx.invert_yaxis()
    ax1_twinx.bar(
        list(range(len(all_df_event.index))),
        MeteForcing_df["prcp mm"],
        label="prcp",
        color="dodgerblue",
        zorder=1,
        alpha=0.3,
        width=0.5,
    )

    for i in range(len(model_names)):
        ax1.plot(
            list(range(len(all_df_event.index))),
            all_df_event[f"sim_total discharge(m3/s)_{model_names[i].strip()}"],
            label=model_names[i],
            color=model_colors[i],
            linestyle="--",
            marker="o",
            markersize=5,
            markerfacecolor="none",
            linewidth=1,
            zorder=7,
        )

    ax1.set_xticks(
        list(range(len(all_df_event)))[:: int(len(all_df_event) / 10)],
        all_df_event.index[:: int(len(all_df_event) / 10)].strftime("%m/%d"),
    )
    ax1.set_xlim(0, len(all_df_event.index) - 1)
    ax1.set_ylabel("Streamflow (m$^3$/s)")
    ax1_twinx.set_ylabel("Precipitation (mm/d)")

    # start_ = date[start[i]]
    # end_ = date[end[i]]
    ax1_ylim = ax1.get_ylim()
    ax1.fill_betweenx(
        np.linspace(ax1_ylim[0], ax1_ylim[1], 100),
        all_df_event.index.get_loc(rising_df_event.index[0]),
        all_df_event.index.get_loc(rising_df_event.index[-1]),
        color="blue",
        alpha=0.2,
        label="rising",
        zorder=1,
    )
    ax1.fill_betweenx(
        np.linspace(ax1_ylim[0], ax1_ylim[1], 100),
        all_df_event.index.get_loc(recession_df_event.index[0]),
        all_df_event.index.get_loc(recession_df_event.index[-1]),
        color="red",
        alpha=0.2,
        label="recession",
        zorder=1,
    )
    ax1.set_ylim(ax1_ylim)

    # get cmap
    OUT_RUNOFF_array = simulated_datasets[0].variables["OUT_RUNOFF"][
        datasets_datetime_index.get_loc(
            rising_df_event.index[0]
        ) : datasets_datetime_index.get_loc(rising_df_event.index[-1])
        + 1,
        :,
        :,
    ]
    OUT_RUNOFF_array = np.ma.filled(OUT_RUNOFF_array, fill_value=0).flatten()
    OUT_RUNOFF_array = OUT_RUNOFF_array[OUT_RUNOFF_array != 0]
    OUT_RUNOFF_range = [
        np.floor(np.min(OUT_RUNOFF_array)),
        np.ceil(np.max(OUT_RUNOFF_array)),
    ]

    OUT_BASEFLOW_array = simulated_datasets[0].variables["OUT_BASEFLOW"][
        datasets_datetime_index.get_loc(
            recession_df_event.index[0]
        ) : datasets_datetime_index.get_loc(recession_df_event.index[-1])
        + 1,
        :,
        :,
    ]
    OUT_BASEFLOW_array = np.ma.filled(OUT_BASEFLOW_array, fill_value=0).flatten()
    OUT_BASEFLOW_array = OUT_BASEFLOW_array[OUT_BASEFLOW_array != 0]
    OUT_BASEFLOW_range = [
        np.floor(np.min(OUT_BASEFLOW_array)),
        np.ceil(np.max(OUT_BASEFLOW_array)),
    ]

    interval_num = 20
    interval_RUNOFF = (OUT_RUNOFF_range[1] - OUT_RUNOFF_range[0]) / interval_num
    bounds_RUNOFF = np.arange(
        OUT_RUNOFF_range[0], OUT_RUNOFF_range[1] + interval_RUNOFF, interval_RUNOFF
    )

    interval_BASEFLOW = (OUT_BASEFLOW_range[1] - OUT_BASEFLOW_range[0]) / interval_num
    bounds_BASEFLOW = np.arange(
        OUT_BASEFLOW_range[0],
        OUT_BASEFLOW_range[1] + interval_BASEFLOW,
        interval_BASEFLOW,
    )
    # bounds_RUNOFF = np.linspace(OUT_RUNOFF_range[0], OUT_RUNOFF_range[1], interval_num)
    # bounds_BASEFLOW = np.linspace(OUT_BASEFLOW_range[0], OUT_BASEFLOW_range[1], interval_num)

    cmap_RUNOFF = plt.get_cmap("viridis")
    norm_RUNOFF = mcolors.BoundaryNorm(bounds_RUNOFF, cmap_RUNOFF.N)

    cmap_BASEFLOW = plt.get_cmap("viridis")
    norm_BASEFLOW = mcolors.BoundaryNorm(bounds_BASEFLOW, cmap_BASEFLOW.N)

    # plot distributed surface flow: rising period
    for i in range(len(rising_df_event)):
        date_index = rising_df_event.index[i]
        index_num = datasets_datetime_index.get_loc(date_index)

        axes_12km_rising[i].imshow(
            simulated_datasets[0].variables["OUT_RUNOFF"][index_num, :, :],
            cmap=cmap_RUNOFF,
            norm=norm_RUNOFF,
        )
        axes_8km_rising[i].imshow(
            simulated_datasets[1].variables["OUT_RUNOFF"][index_num, :, :],
            cmap=cmap_RUNOFF,
            norm=norm_RUNOFF,
        )
        axes_6km_rising[i].imshow(
            simulated_datasets[2].variables["OUT_RUNOFF"][index_num, :, :],
            cmap=cmap_RUNOFF,
            norm=norm_RUNOFF,
        )

    # plot distributed baseflow: recession period
    for i in range(len(recession_df_event)):
        date_index = recession_df_event.index[i]
        index_num = datasets_datetime_index.get_loc(date_index)

        axes_12km_recession[i].imshow(
            simulated_datasets[0].variables["OUT_BASEFLOW"][index_num, :, :],
            cmap=cmap_BASEFLOW,
            norm=norm_BASEFLOW,
        )
        axes_8km_recession[i].imshow(
            simulated_datasets[1].variables["OUT_BASEFLOW"][index_num, :, :],
            cmap=cmap_BASEFLOW,
            norm=norm_BASEFLOW,
        )
        axes_6km_recession[i].imshow(
            simulated_datasets[2].variables["OUT_BASEFLOW"][index_num, :, :],
            cmap=cmap_BASEFLOW,
            norm=norm_BASEFLOW,
        )

    # set outline_patch as False
    [ax.spines["left"].set_visible(False) for ax in all_axes_rising_recession]
    [ax.spines["right"].set_visible(False) for ax in all_axes_rising_recession]
    [ax.spines["top"].set_visible(False) for ax in all_axes_rising_recession]
    [ax.spines["bottom"].set_visible(False) for ax in all_axes_rising_recession]

    [ax.set_xticks([]) for ax in all_axes_rising_recession]
    [ax.set_xticks([]) for ax in all_axes_rising_recession]
    [ax.set_xticks([]) for ax in all_axes_rising_recession]
    [ax.set_xticks([]) for ax in all_axes_rising_recession]

    [ax.set_yticks([]) for ax in all_axes_rising_recession]
    [ax.set_yticks([]) for ax in all_axes_rising_recession]
    [ax.set_yticks([]) for ax in all_axes_rising_recession]
    [ax.set_yticks([]) for ax in all_axes_rising_recession]

    # text
    [
        axes_12km_rising[i].set_title(
            format(rising_df_event.index[i], "%m%d"),
            pad=8,
            fontdict={"family": "Arial", "size": 12},
        )
        for i in range(len(axes_12km_rising))
    ]
    [
        axes_12km_recession[i].set_title(
            format(recession_df_event.index[i], "%m%d"),
            pad=8,
            fontdict={"family": "Arial", "size": 12},
        )
        for i in range(len(axes_12km_recession))
    ]

    axes_12km_rising[0].set_ylabel("12 km", labelpad=13)
    axes_8km_rising[0].set_ylabel("8 km", labelpad=13)
    axes_6km_rising[0].set_ylabel("6 km", labelpad=13)

    axes_12km_recession[0].set_ylabel("12 km", labelpad=13)
    axes_8km_recession[0].set_ylabel("8 km", labelpad=13)
    axes_6km_recession[0].set_ylabel("6 km", labelpad=13)

    fig.text(0.09, 0.95, "(a)", fontdict={"size": 14, "weight": "bold"})
    fig.text(0.07, 0.55, "(b)", fontdict={"size": 14, "weight": "bold"})
    fig.text(0.525, 0.55, "(c)", fontdict={"size": 14, "weight": "bold"})

    # legend and colorbar
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twinx.get_legend_handles_labels()
    plt.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        prop={"family": "Arial", "size": 12, "weight": "bold"},
    )

    sm_RUNOFF = ScalarMappable(norm=norm_RUNOFF, cmap=cmap_RUNOFF)
    cbar_RUNOFF = plt.colorbar(
        sm_RUNOFF, cax=ax_left_cb, orientation="horizontal", extend="both", pad=0.3
    )
    cbar_RUNOFF.set_label("SURFACE RUNOFF mm/d")

    sm_BASEFLOW = ScalarMappable(norm=norm_BASEFLOW, cmap=cmap_BASEFLOW)
    cbar_BASEFLOW = plt.colorbar(
        sm_BASEFLOW, cax=ax_right_cb, orientation="horizontal", extend="both", pad=0.3
    )
    cbar_BASEFLOW.set_label("BASEFLOW mm/d")

    return fig


def plot_params(params_dataset):
    """
    Plot four different parameter datasets in a 2x2 grid with colorbars.

    Parameters
    ----------
    params_dataset : Dataset
        A xarray Dataset containing the parameters to be plotted. The dataset should have variables:
        - "infilt"
        - "Ws"
        - "Ds"
        - "Dsmax"
        Additionally, the dataset should have "lon" and "lat" for proper axis labeling.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure containing the plots.
    axes : ndarray
        An array of axes objects, corresponding to each subplot.

    Notes
    -----
    - The color maps used for the plots are 'RdBu'.
    - The x and y ticks are adjusted for better readability and are based on the dataset's latitude and longitude.
    - Each subplot is annotated with a label ("(a)", "(b)", "(c)", "(d)").
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 8),
        gridspec_kw={
            "left": 0.05,
            "right": 0.98,
            "bottom": 0.05,
            "top": 0.95,
            "wspace": 0.15,
            "hspace": 0.16,
        },
    )
    im1 = axes[0, 0].imshow(
        params_dataset.variables["infilt"][:, :], cmap="RdBu"
    )  # vmin=0, vmax=0.4,
    im2 = axes[0, 1].imshow(params_dataset.variables["Ws"][:, :], cmap="RdBu")
    im3 = axes[1, 0].imshow(
        params_dataset.variables["Ds"][:, :], cmap="RdBu"
    )  # vmin=0, vmax=1,
    im4 = axes[1, 1].imshow(
        params_dataset.variables["Dsmax"][:, :], cmap="RdBu"
    )  # vmin=0, vmax=30,
    ims = [im1, im2, im3, im4]
    axes_flatten = axes.flatten()

    xticks = list(range(params_dataset.variables["infilt"].shape[1]))
    yticks = list(range(params_dataset.variables["infilt"].shape[0]))
    xticks_labels = [format_lon(lon, 0) for lon in params_dataset.variables["lon"][:]]
    yticks_labels = [format_lat(lat, 0) for lat in params_dataset.variables["lat"][:]]
    yticks_labels.reverse()

    [
        ax.set_xticks(
            xticks[:: int(len(xticks) / 4)],
            xticks_labels[:: int(len(xticks) / 4)],
            fontfamily="Arial",
            fontsize=10,
        )
        for ax in axes_flatten
    ]
    [
        ax.set_yticks(
            yticks[:: int(len(yticks) / 3.5)],
            yticks_labels[:: int(len(yticks) / 3.5)],
            fontfamily="Arial",
            fontsize=10,
        )
        for ax in axes_flatten
    ]
    [rotate_yticks(ax, yticks_rotation=90) for ax in axes.flatten()]
    cbs = [
        fig.colorbar(ims[i], ax=axes_flatten[i], extend="both", shrink=1)
        for i in range(len(axes_flatten))
    ]
    cbtitles = ["binfilt", "Ws", "Ds", "Dsmax"]
    cbs = [
        cbs[i].ax.set_title(
            label=cbtitles[i], fontdict={"family": "Arial", "size": 12}, pad=18
        )
        for i in range(len(cbs))
    ]

    bbox = dict(boxstyle="Square,pad=0.1", facecolor="white", edgecolor="none", alpha=1)
    axes_flatten[0].annotate(
        "(a)",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        color="k",
        bbox=bbox,
    )
    axes_flatten[1].annotate(
        "(b)",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        color="k",
        bbox=bbox,
    )
    axes_flatten[2].annotate(
        "(c)",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        color="k",
        bbox=bbox,
    )
    axes_flatten[3].annotate(
        "(d)",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        color="k",
        bbox=bbox,
    )

    return fig, axes

