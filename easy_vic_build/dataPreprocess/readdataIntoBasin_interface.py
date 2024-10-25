# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
from .ExtractData import *

# ------------------------ read data into basins ------------------------
def readCAMELSStreamflowIntoBasins(basin_shp, read_dates=None):
    # pd.date_range(start=read_dates[0], end=read_dates[1], freq="D")
    basin_shp = Extract_CAMELS_Streamflow.ExtractData(basin_shp, read_dates=read_dates)
    
    return basin_shp

def readCAMELSAttributeIntoBasins(basin_shp, k_list=None):
    basin_shp = Extract_CAMELS_Attribute.ExtractData(basin_shp, k_list=k_list)
    return basin_shp


# def checkStreamflowMissing(usgs_streamflow_, date_period=["19980101", "20101231"]):
#     reason = ''
#     date_period_range = pd.date_range(start=date_period[0], end=date_period[1], freq="D")
#     usgs_streamflow_date = list(map(lambda i: datetime(*i), zip(usgs_streamflow_.loc[:, 1], usgs_streamflow_.loc[:, 2], usgs_streamflow_.loc[:, 3])))
#     usgs_streamflow_date = np.array(usgs_streamflow_date)
    
#     try:
#         startIndex = np.where(usgs_streamflow_date == date_period_range[0])[0][0]
#         endIndex = np.where(usgs_streamflow_date == date_period_range[-1])[0][0]
#         if 'M' not in usgs_streamflow_.iloc[startIndex:endIndex + 1, -1].values:
#             judgement = True
#         else:
#             judgement = False
#             reason += f" M in {date_period[0]}-{date_period[1]} "
#         if len(usgs_streamflow_.iloc[startIndex:endIndex + 1, :]) < len(date_period_range):
#             judgement = False
#             reason += f" len < {len(date_period_range)} "

#     except:
#         judgement = False
#         reason += f" cannot find {date_period[0]} or {date_period[1]} in file "

#     return judgement, reason


# def removeStreamflowMissing(fns, fpaths, usgs_streamflow, date_period):
#     """_summary_

#     Returns:
#         list of dicts: remove_files_Missing
#             # unpack remove_files_Missing
#             remove_reason_streamflow_Missing= [f["reason"] for f in remove_files_Missing]
#             remove_fn_streamflow_Missing = [f["fn"] for f in remove_files_Missing]
#             remove_fpath_streamflow_Missing = [f["fpath"] for f in remove_files_Missing]
#             remove_usgs_streamflow_Missing = [f["usgs_streamflow"] for f in remove_files_Missing]
#     """
#     # copy
#     fns = copy(fns)
#     fpaths = copy(fpaths)
#     usgs_streamflow = copy(usgs_streamflow)

#     # general set
#     remove_files_Missing = []

#     # remove Streamflow with 'M' or less len
#     i = 0
#     while i < len(fns):
#         fn = fns[i]
#         fpath = fpaths[i]
#         usgs_streamflow_ = usgs_streamflow[i]
#         judgement, reason = checkStreamflowMissing(usgs_streamflow_, date_period)
#         if judgement:
#             i += 1
#         else:
#             # remove file from fns and fpaths
#             print(f"remove {fn}")
#             remove_files_Missing.append(
#                 {"fn": fn, "fpath": fpath, "usgs_streamflow": usgs_streamflow_, "reason": reason})
#             fns.pop(i)
#             fpaths.pop(i)
#             usgs_streamflow.pop(i)

#     # fns -> id
#     streamflow_id = [int(fns[:fns.find("_")]) for fns in fns]

#     print(f"count: remove {len(remove_files_Missing)} files, remaining {len(usgs_streamflow)} files")

#     return fns, fpaths, usgs_streamflow, streamflow_id, remove_files_Missing


# def readForcingDaymet(home):
#     forcingDaymet_dir = os.path.join(
#         home, "basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet")
#     fns = []
#     fpaths = []
#     forcingDaymet = []
#     forcingDaymetGaugeAttributes = []

#     for dir in os.listdir(forcingDaymet_dir):
#         fns.extend([fn for fn in os.listdir(os.path.join(forcingDaymet_dir, dir)) if fn.endswith(".txt")])
#         fpaths.extend([os.path.join(forcingDaymet_dir, dir, fn)
#                       for fn in os.listdir(os.path.join(forcingDaymet_dir, dir)) if fn.endswith(".txt")])

#     for i in range(len(fns)):
#         fn = fns[i]
#         fpath = fpaths[i]
#         forcingDaymet.append(pd.read_csv(fpath, sep="\s+", skiprows=3))
#         GaugeAttributes_ = pd.read_csv(fpath, header=None, nrows=3).values
#         forcingDaymetGaugeAttributes.append({"latitude": GaugeAttributes_[0][0],
#                                              "elevation": GaugeAttributes_[1][0],
#                                              "basinArea": GaugeAttributes_[2][0],
#                                              "gauge_id": int(fn[:8])})

#     return fns, fpaths, forcingDaymet, forcingDaymetGaugeAttributes


# def readForcingDaymetIntoBasins(forcingDaymet, forcingDaymetGaugeAttributes, basinShp, read_dates, read_keys):
#     """
#     params:
#         read_dates: pd.date_range("19800101", "20141231", freq="D"), should be set to avoid missing value
#         read_keys: ["prcp(mm/day)"]  # "prcp(mm/day)" "srad(W/m2)" "dayl(s)" "swe(mm)" "tmax(C)" "tmin(C)" "vp(Pa)"
#     """
#     extract_lists = [[] for i in range(len(read_keys))]
#     for i in tqdm(basinShp.index, desc="loop for reading forcing Daymet into basins", colour="green"):
#         basinShp_i = basinShp.loc[i, :]
#         hru_id = basinShp_i.hru_id
#         for j in range(len(read_keys)):
#             key = read_keys[j]
#             extract_list = extract_lists[j]
#             forcingDaymet_basin_set, _ = ExtractForcingDaymet(
#                 forcingDaymet, forcingDaymetGaugeAttributes, hru_id, read_dates)
#             extract_list.append(forcingDaymet_basin_set[key])

#     for j in range(len(read_keys)):
#         key = read_keys[j]
#         extract_list = extract_lists[j]
#         basinShp[key] = extract_list

#     return basinShp





