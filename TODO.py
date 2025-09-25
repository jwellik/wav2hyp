##################################
## PRIORITIES
# 1. P-S Identical picks


##################################
## PROGRAM

# Structure and Organization
# [x] config YML
# [x] Add config param nll_home (remove hard code from run_locator())
# [x] Organize as package
# [x] Remove extra folder from output paths

# Performance and convenience
# [x] Summary file/summary output (see note below)
# TODO Parallelize
# TODO Limits of HDF5 file? Is EQT output hitting that?

# Install
# TODO Install vdapseisutils and nllpy from github
# TODO VDAPSEISUTILS Install needs to be cleaned up
# - currently, there are so many imports, it messes up wav2hyp
# - - cartopy timezonefinder
# - Had to install conda install -c conda-forge pytables
# TODO Add note about using tmux to the README.md file

# Log files
# TODO Logging - send everything to log or to console
# TODO Log files should have meaningful name or be placed in named directory
# TODO Possible for config file to define paths like $BASEDIR/picks ?
# [x] Look at the timing logs. Something is up (see example at bottom)

# IO
# TODO Soft-coding of storage file names (e.g., eqt-volpick.h5 after method-model instead of hard-coded)

##################################
## WAVEFORM COLLECTION
# TODO Better print message (requires work in vdapseisutils?)
# #Downloading: UW.YEL..EHZ
# #  Trying client 1: VClient(Client: <obspy.clients.filesystem.sds.Client object at 0x7d44ae0671d0>)
# #  Success: Retrieved 80 traces
# TODO Better summary message about collected waveforms
# - TODO Summary text file abt waveforms is misleading
# ## --- Summary ---
# ## Total traces retrieved: 428
# ## Networks: UW


##################################
## PICKER
# [x] VInventory.get_waveforms(client, t1, t2)
# TODO Should I keep unassociated picks? Keep them in a different hd5file?
# TODO P & S sometimes getting picked with identical times (is this a picker problem, or later?)


##################################
## ASSOCIATOR
# [x] Print number of associated events (end of run + read_pyocto)
# TODO ? Limit number of picks for association with stricter threshold than saving picks?


##################################
## LOCATOR
# [x] Read in NonLinLoc Hyp file -> Catalog -> QuakeML
# TODO Handle NonLinLoc outputs (proper org. of output files and dirs)
# [x] obs file must be date specific so that parallel processing works!
# TODO Print number of located events (end of run + read_nlloc)
# TODO Change locator.run_vel2grid & locator.run_grid2time to overwrite methods (requires work in nllpy)
# TODO ? Limit number of locations based on quality of associated picks, quality of association?

## NLLPY
# TODO run_vel2grid(overwrite=False)  # search for missing stations if overwrite==False
# TODO run_grid2time(overwrite=False)  # search for missing stations if overwrite==False
# TODO run_nll(vel2grid=True, grid2time=True, nlloc=True) --> run_nll_all() + run_nlloc()
# - overwrite, skip


##################################
## PLOTS
# TODO Plots
# - [x] Helicorder
# - [x] Pensive (Spectrograms)
# - [x] Map
# - TODO Detections, Associations, Locations TimeSeries
# - [x] Histogram of RMS, azgap


"""
Notes:

Summary output file:

./results_local/spurr_qc_summary.txt
date,config,ncha,nsamp,pick_model,np,ns,npicks,ndetections,assoc_method,assignments,events,loc_method,locations,last_updated,t_pick,t_assoc,t_loc,t_total
2024/10/14,./examples_local/spurr.yaml,24,103680000,eqt-volpick,2651,8300,10951,3355,pyocto,210,21,nll,21,20250922T114418,169.28,1680.34,1893.93,3743.55

./results_local/spurr_qc_summary_awu1.txt
date,config,ncha,nsamp,pick_model,np,ns,npicks,ndetections,last_updated,t_pick

- Create while the program runs
- Create from h5 files retrospectively
- Simple plotting feature

"""