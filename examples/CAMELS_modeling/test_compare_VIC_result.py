# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from easy_vic_build.tools.calibrate_func.evaluate_metrics import EvaluationMetric
from matplotlib import gridspec
plt.rcParams['font.family'] = 'Arial'




if __name__ == "__main__":
    home = "F:\\research\\Research\\easy_vic_build\\paper\\submit\\1.submit_1\\cases"
    VIC_12_result_fp = os.path.join(home, "397_12km", "VICResults", "fluxes.1998-01-01.nc")
    VIC_8_result_fp = os.path.join(home, "397_8km_transferability", "VICResults", "fluxes.1998-01-01.nc")
    VIC_6_result_fp = os.path.join(home, "397_6km_transferability", "VICResults", "fluxes.1998-01-01.nc")
    
    VIC_12_runoff = Dataset(VIC_12_result_fp).variables["OUT_RUNOFF"][730:, :, :]
    VIC_8_runoff = Dataset(VIC_8_result_fp).variables["OUT_RUNOFF"][730:, :, :]
    VIC_6_runoff = Dataset(VIC_6_result_fp).variables["OUT_RUNOFF"][730:, :, :]
    VIC_12_runoff = np.nanmean(VIC_12_runoff, axis=(1, 2))
    VIC_8_runoff = np.nanmean(VIC_8_runoff, axis=(1, 2))
    VIC_6_runoff = np.nanmean(VIC_6_runoff, axis=(1, 2))
    
    VIC_8_em = EvaluationMetric(VIC_8_runoff, VIC_12_runoff)
    VIC_6_em = EvaluationMetric(VIC_6_runoff, VIC_12_runoff)
    
    result = pd.DataFrame({
        "VIC_8km": [VIC_8_em.KGE(), VIC_8_em.NSE(), VIC_8_em.PBias()],
        "VIC_6km": [VIC_6_em.KGE(), VIC_6_em.NSE(), VIC_6_em.PBias()]
    }, index=["RMSE", "NSE", "PBIAS"])
    
    model_colors = ["red", "blue", "green"]
    model_names = ["VIC 12km", "VIC 8km", "VIC 6km"]
    
    # plot
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        left=0.08,
        right=0.98,
        bottom=0.15,
        top=0.98,
        hspace=0.15,
        wspace=0.3,
    )  # wspace=0.15, wspace=0.3
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    
    ax1.plot(
        list(range(len(VIC_12_runoff))),
        VIC_12_runoff,
        color="grey",
        linestyle="-",
        linewidth=1,
        label="VIC 12km",
    )
    
    ax1.plot(
        list(range(len(VIC_8_runoff))),
        VIC_8_runoff,
        color=model_colors[1],
        marker="o",
        markersize=2,
        linestyle="None",
        label="VIC 8km",
    )
    
    ax1.plot(
        list(range(len(VIC_6_runoff))),
        VIC_6_runoff,
        color=model_colors[2],
        marker="o",
        markersize=2,
        linestyle="None",
        label="VIC 6km",
    )
    
    
    ax2.scatter(
        VIC_12_runoff,
        VIC_8_runoff,
        facecolors="none",
        edgecolor=model_colors[1],
        s=10,
        linewidth=1,
        label=None,
        alpha=0.8,
    )
    
    ax2.scatter(
        VIC_12_runoff,
        VIC_6_runoff,
        facecolors="none",
        edgecolor=model_colors[2],
        s=10,
        linewidth=1,
        label=None,
        alpha=0.8,
    )
    
    p_total = np.polyfit(
        VIC_12_runoff, VIC_8_runoff, deg=1, rcond=None, full=False, w=None, cov=False
    )
    
    ax2.plot(
        np.arange(ax2.get_xlim()[0], ax2.get_xlim()[1], 1),
        np.polyval(
            p_total, np.arange(ax2.get_xlim()[0], ax2.get_xlim()[1], 1)
        ),
        color=model_colors[1],
        linestyle="-",
        linewidth=1,
        label=f"{model_names[1]}: y = {p_total[0]:.2f}x",
    )
    
    p_total = np.polyfit(
        VIC_12_runoff, VIC_6_runoff, deg=1, rcond=None, full=False, w=None, cov=False
    )
    
    ax2.plot(
        np.arange(ax2.get_xlim()[0], ax2.get_xlim()[1], 1),
        np.polyval(
            p_total, np.arange(ax2.get_xlim()[0], ax2.get_xlim()[1], 1)
        ),
        color=model_colors[2],
        linestyle="-",
        linewidth=1,
        label=f"{model_names[2]}: y = {p_total[0]:.2f}x",
    )
    
    # set ticks
    date = pd.date_range("20000101", "20101231", freq="D")
    date = date.strftime("%Y-%m-%d").tolist()
    ylim = (
        0,
        [max(model) for model in [VIC_12_runoff, VIC_8_runoff, VIC_6_runoff] if len(model) > 0][0] * 1.1,
    )
        
    ax1.set_xticks(
        list(range(len(VIC_12_runoff)))[:: int(len(VIC_12_runoff) / 5)],
        date[:: int(len(VIC_12_runoff) / 5)],
    )
    ax1.set_xlim(0, len(date))
    ax1.set_ylim(ylim)
    
    ax1.set_ylabel("Runoff (mm/day)")
    ax1.set_xlabel("Date")

    ax2.set_xlim(0, ax1.get_ylim()[1])
    ax2.set_ylim(0, ax1.get_ylim()[1])
    ax2.xaxis.set_major_locator(plt.LinearLocator(numticks=5))
    ax2.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
    ax2.set_xlabel("VIC 12 runoff (mm/day)")
    ax2.set_ylabel("VIC 8,6 runoff (mm/day)")
    
    ax1.legend(loc="upper right", prop={"size": 10, "family": "Arial"})
    ax2.legend(loc="upper right", prop={"size": 10, "family": "Arial"})

    ax1.annotate(
        "(a)", xy=(0.02, 0.92), xycoords="axes fraction", fontsize=14, fontweight="normal"
    )
    ax2.annotate(
        "(b)", xy=(0.02, 0.92), xycoords="axes fraction", fontsize=14, fontweight="normal"
    )
    
    fig.savefig(
        os.path.join(home, "test_compare_VIC_result_runoff.tiff"), dpi=300
    )
    
    result.to_csv(
        os.path.join(home, "test_compare_VIC_result_runoff.csv"),
        index=True,
        header=True
    )