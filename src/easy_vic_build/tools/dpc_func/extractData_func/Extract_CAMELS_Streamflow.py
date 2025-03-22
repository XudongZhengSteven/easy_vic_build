# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd


def readStreamflow(hru_id_list=None):
    # hru_id_list: [hru_id ...] to extract specific stations, or read all stations, note that hru_id is int type

    # general set
    usgs_streamflow_dir = "E:/data/hydrometeorology/CAMELS/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow"
    fns = []
    fpaths = []
    usgs_streamflows = []

    # loop read all fns
    for dir in os.listdir(usgs_streamflow_dir):
        fns.extend(
            [
                fn
                for fn in os.listdir(os.path.join(usgs_streamflow_dir, dir))
                if fn.endswith(".txt")
            ]
        )
        fpaths.extend(
            [
                os.path.join(usgs_streamflow_dir, dir, fn)
                for fn in os.listdir(os.path.join(usgs_streamflow_dir, dir))
                if fn.endswith(".txt")
            ]
        )

    # extract fn/fns based on hru_id
    if hru_id_list is not None:
        fns = [fn for fn in fns if int(fn[: fn.find("_")]) in hru_id_list]
        fpaths = [
            fpath
            for fpath in fpaths
            if int(fpath[fpath.rfind("\\") + 1 : fpath.rfind("streamflow") - 1])
            in hru_id_list
        ]

    for i in range(len(fns)):
        fpath = fpaths[i]
        usgs_streamflow_ = pd.read_csv(fpath, sep="\s+", header=None)
        usgs_streamflows.append(usgs_streamflow_)

    # fns -> id
    streamflow_ids = [int(fns[: fns.find("_")]) for fns in fns]

    return fns, fpaths, usgs_streamflows, streamflow_ids


# streamflow_id, usgs_streamflow, read_dates=None
def ExtractData(basin_shp, read_dates=None):
    # get data for hru in basin_shp
    fns_streamflow, fpaths_streamflow, usgs_streamflows, streamflow_ids = (
        readStreamflow(basin_shp.hru_id.values.tolist())
    )

    extract_lists = []
    for i in basin_shp.index:
        # extract hru_id
        basinShp_i = basin_shp.loc[i, :]
        hru_id = basinShp_i.hru_id
        extract_index = streamflow_ids.index(hru_id)
        usgs_streamflow_ = usgs_streamflows[extract_index]

        if read_dates is not None:
            # extract date
            date_period_range = pd.date_range(
                start=read_dates[0], end=read_dates[1], freq="D"
            )
            usgs_streamflow_date = list(
                map(
                    lambda i: datetime(*i),
                    zip(
                        usgs_streamflow_.loc[:, 1],
                        usgs_streamflow_.loc[:, 2],
                        usgs_streamflow_.loc[:, 3],
                    ),
                )
            )
            usgs_streamflow_date = np.array(usgs_streamflow_date)
            usgs_streamflow_date_str = np.array(
                [d.strftime("%Y%m%d") for d in usgs_streamflow_date]
            )

            try:
                startIndex = np.where(usgs_streamflow_date <= date_period_range[0])[0][
                    -1
                ]
            except:
                startIndex = 0
            try:
                endIndex = np.where(usgs_streamflow_date >= date_period_range[-1])[0][0]
            except:
                endIndex = len(usgs_streamflow_)

            usgs_streamflow_ = usgs_streamflow_.iloc[startIndex : endIndex + 1, :]
            usgs_streamflow_.loc[:, "date"] = usgs_streamflow_date_str[
                startIndex : endIndex + 1
            ]

        extract_lists.append(usgs_streamflow_)

    basin_shp["streamflow"] = extract_lists

    return basin_shp


def checkStreamflowMissing(usgs_streamflow, date_period=["19980101", "20101231"]):
    # check for each usgs_streamflow
    reason = ""
    date_period_range = pd.date_range(
        start=date_period[0], end=date_period[1], freq="D"
    )
    usgs_streamflow_date = list(
        map(
            lambda i: datetime(*i),
            zip(
                usgs_streamflow.loc[:, 1],
                usgs_streamflow.loc[:, 2],
                usgs_streamflow.loc[:, 3],
            ),
        )
    )
    usgs_streamflow_date = np.array(usgs_streamflow_date)

    try:
        startIndex = np.where(usgs_streamflow_date == date_period_range[0])[0][0]
        endIndex = np.where(usgs_streamflow_date == date_period_range[-1])[0][0]
        if "M" not in usgs_streamflow.iloc[startIndex : endIndex + 1, -1].values:
            judgement = True  # not remove
        else:
            judgement = False  # remove
            reason += f" M in {date_period[0]}-{date_period[1]} "
        if len(usgs_streamflow.iloc[startIndex : endIndex + 1, :]) < len(
            date_period_range
        ):
            judgement = False
            reason += f" len < {len(date_period_range)} "

    except:
        judgement = False
        reason += f" cannot find {date_period[0]} or {date_period[1]} in file "

    return judgement, reason


def removeStreamflowMissingfromlists(fns, fpaths, usgs_streamflows, date_period):
    """_summary_

    Returns:
        list of dicts: remove_files_Missing
            # unpack remove_files_Missing
            remove_reason_streamflow_Missing= [f["reason"] for f in remove_files_Missing]
            remove_fn_streamflow_Missing = [f["fn"] for f in remove_files_Missing]
            remove_fpath_streamflow_Missing = [f["fpath"] for f in remove_files_Missing]
            remove_usgs_streamflow_Missing = [f["usgs_streamflow"] for f in remove_files_Missing]
    """
    # copy
    fns = deepcopy(fns)
    fpaths = deepcopy(fpaths)
    usgs_streamflows = deepcopy(usgs_streamflows)

    # general set
    files_Missing = []

    # remove Streamflow with 'M' or less len
    i = 0
    while i < len(fns):
        fn = fns[i]
        fpath = fpaths[i]
        usgs_streamflow_ = usgs_streamflows[i]
        judgement, reason = checkStreamflowMissing(usgs_streamflow_, date_period)
        if judgement:  # not remove
            i += 1
        else:  # remove
            # remove file from fns and fpaths
            print(f"remove {fn}")
            files_Missing.append(
                {
                    "fn": fn,
                    "fpath": fpath,
                    "usgs_streamflow": usgs_streamflow_,
                    "reason": reason,
                }
            )
            fns.pop(i)
            fpaths.pop(i)
            usgs_streamflows.pop(i)

    # fns -> id
    streamflow_ids = [int(fns[: fns.find("_")]) for fns in fns]

    print(
        f"count: remove {len(files_Missing)} files, remaining {len(usgs_streamflows)} files"
    )

    return fns, fpaths, usgs_streamflows, streamflow_ids, files_Missing


def getremoveStreamflowMissing(date_period=["19980101", "20101231"]):
    # get all streamflow
    fns_streamflow, fpaths_streamflow, usgs_streamflows, streamflow_ids = (
        readStreamflow(hru_id_list=None)
    )

    # remove missing
    (
        fns_streamflow_removed_missing,
        fpaths_streamflow_removed_missing,
        usgs_streamflows_removed_missing,
        streamflow_ids_removed_missing,
        files_Missing,
    ) = removeStreamflowMissingfromlists(
        fns_streamflow, fpaths_streamflow, usgs_streamflows, date_period
    )  # 671 -> 652

    # save
    streamflows_dict_original = {
        "fns": fns_streamflow,
        "fpaths": fpaths_streamflow,
        "usgs_streamflows": usgs_streamflows,
        "streamflow_ids": streamflow_ids,
    }
    streamflows_dict_removed_missing = {
        "fns": fns_streamflow_removed_missing,
        "fpaths": fpaths_streamflow_removed_missing,
        "usgs_streamflows": usgs_streamflows_removed_missing,
        "streamflow_ids": streamflow_ids_removed_missing,
        "files_Missing": files_Missing,
    }

    return streamflows_dict_original, streamflows_dict_removed_missing
