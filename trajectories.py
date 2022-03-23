import numpy as np
import numpy.typing as npt
import vibrations as vib
from typing import Union
import plotting as plot


##########################################################################################################
# Author: Kyle Acheson
# A module to perform normal mode analysis on a single or ensemble of trajectories                       #
# given a set of normal mode coordinates it can be used to determine the activity of each normal mode    #
# allows analysis over a set of requested time intervals or the whole trajectory run time                #
# also includes various other utilities for analysing trajectories                                       #
##########################################################################################################


def ensemble_analysis(norm_mode_trajs: npt.NDArray, traj_time_count: npt.NDArray, time_intervals:npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # performs analysis over the whole ensemble of trajectories
    # gives the average set of normal mode coordinates and the standard deviation of each mode
    # from the mean trajectory - also gives the total standard deviation of each mode
    # averaged across specified time intervals (by call to ensemble_tint_analysis)
    sz = np.shape(norm_mode_trajs)
    nvibs, traj, nts = sz[0], sz[1], sz[2]
    ntints = np.shape(time_intervals)[1]
    traj_avg_std = np.zeros((nvibs, nts), float)
    traj_avg_norm_modes = np.zeros((nvibs, nts), float)
    traj_sum = np.nansum(norm_mode_trajs, axis=1)
    traj_sum_sq = np.nansum(norm_mode_trajs**2, axis=1)
    exclude_traj_timesteps = np.where(traj_time_count == 0)[0].tolist()

    if len(exclude_traj_timesteps) > 0:
        print("No trajectories run to: %s timesteps - analysis truncated at previous timestep." % exclude_traj_timesteps)

    for i in range(nts):
        for j in range(nvibs):
            traj_avg_int = traj_sum[j, i] / traj_time_count[i]
            traj_avg_int_sq = traj_sum_sq[j, i] / traj_time_count[i]
            traj_avg_norm_modes[j, i] = traj_avg_int # norm modes averaged over all trajs
            std_traj = (traj_time_count[i] / (traj_time_count[i] - 1) * (traj_avg_int_sq - traj_avg_int ** 2)) ** 0.5
            traj_avg_std[j, i] = std_traj  # std of each nm over time on average traj

    avg_tint_std = ensemble_tint_analysis(traj_avg_norm_modes, time_intervals)
    return traj_avg_std, traj_avg_norm_modes, avg_tint_std


def ensemble_tint_analysis(traj_avg_norm_modes: npt.NDArray, time_intervals: npt.NDArray) -> Union[npt.NDArray, None]:
    # calculates standard deviation from the mean trajectory for each specified time interval
    # only possible if the specified time intervals are the same for every trajectory
    nvibs = np.shape(traj_avg_norm_modes)[0]
    ntints = np.shape(time_intervals)[1]
    avg_tint_std = np.zeros((nvibs, ntints), float)
    if len(np.unique(time_intervals, axis=2)) == ntints:
        for i in range(ntints):
            tint = time_intervals[:, i, 0]
            tstart, tend = int(tint[0]), int(tint[1])
            tdiff = tend - tstart
            avg_all_tint = np.nansum(traj_avg_norm_modes[:, tstart:tend], axis=1) / tdiff
            avg_sq_all_tint = np.nansum(traj_avg_norm_modes[:, tstart:tend]**2, axis=1) / tdiff
            avg_tint_std[:, i] = (tdiff/ (tdiff-1) * (avg_sq_all_tint - avg_all_tint ** 2)) ** 0.5
            # av_tint_std = av_tint_std * mult_array
            # std of averaged trajectories within each time interval
        return avg_tint_std

    else:
        print("""Can not perform std calculation over specified time intervals for mean trajectories as time
              intervals are not consistant over all trajectories.""")
        return None


def single_traj_nma(trajectories: npt.NDArray, ref_atoms: npt.NDArray,
                    norm_mode_mat: npt.NDArray, time_intervals: npt.NDArray, ntints: int, nvibs: int):
    # performs normal mode analysis on a single trajectory
    # does transformation to normal mode coordinates
    # also calculates the average traj in norm mode coords over a series of time intervals
    # calculates standard dev on the average over each time interval by call to calc_traj_std
    ntraj = np.shape(trajectories)[2]
    nts = np.shape(trajectories)[3]
    norm_mode_trajs = np.zeros((nvibs, ntraj, nts))
    interval_recs = np.zeros((ntints, ntraj))
    interval_stds = np.zeros((nvibs, ntints, ntraj), float)
    interval_avg = np.zeros((nvibs, ntints, ntraj), float)
    traj_std = np.zeros((nvibs, ntints, ntraj), float)
    traj_time_count = np.zeros((nts), int)

    #traj_std_b = np.zeros((nvibs, ntints, ntraj), float)
    #interval_avg_b = np.zeros((nvibs, ntints, ntraj), float)
    for traj in range(ntraj):
        trajectory = trajectories[:,:,traj,:]
        norm_mode_coords = vib.norm_mode_basis(trajectory, nts, nvibs, ref_atoms, norm_mode_mat)
        norm_mode_trajs[:, traj, :] = norm_mode_coords
        traj_time_intervals = time_intervals[:, :, traj]
        interval_rec, interval_std, interval_summed, interval_summed_sq, avg_tint = calc_traj_std(norm_mode_coords, traj_time_intervals, ntints, nvibs, traj)
        traj_std[:, :, traj] = interval_std
        interval_avg[:, :, traj] = avg_tint
        interval_recs[:, traj] = interval_rec
        interval_stds[:, :, traj] = interval_std
        #traj_std[:, :, traj], interval_avg[:, :, traj] = trg.tint_averaging(interval_summed, interval_summed_sq, interval_rec, ntints, nvibs)
        # for each traj - std and average of nm over time intervals
        plot.plot_single_traj(norm_mode_coords, avg_tint, interval_std, traj_time_intervals)
        traj_time_count += count_traj_timesteps(trajectory, nts)

    return norm_mode_trajs, traj_std, interval_avg, traj_time_count


def calc_traj_std(norm_mode_coords: npt.NDArray, time_intervals: npt.NDArray, ntints: int, nvibs: int, traj: int):
    # calculates the std and average of normal modes over specified time intervals for single trajectory

    # set up record array over specified time intervals - given for every traj as intervals may vary per traj if specified
    # all arrays are initialised with nan's in event time interval does not exist for trajectory in question
    interval_rec = np.full((ntints), np.nan) # time interval lengths for each trajectory
    interval_summed = np.full((nvibs, ntints), np.nan) # sum over time for each time interval and normal mode
    interval_summed_sq = np.full((nvibs, ntints), np.nan) # squared sum over time for each time interval and normal mode
    interval_std = np.full((nvibs, ntints), np.nan) # standard deviation over the duration of the specified intervals
    interval_avg = np.full((nvibs, ntints), np.nan)

    for j in range(ntints):
        tint = time_intervals[:, j]
        tstart, tend = int(tint[0]), int(tint[1])
        tdiff = tend - tstart
        norm_mode_coord = norm_mode_coords[:,tstart:tend]
        if np.isnan(norm_mode_coord).any():
            print("Trajectory %s terminates before time interval %s (tstart: %s, tend: %s)" % (traj, j, tstart, tend))
            tend = min(list(map(tuple, np.where(np.isnan(norm_mode_coord))))[1])
            tdiff = tend # rassign to timepoint traj crashes at
            print("Only including trajectory up till its termination point: %s." % (tend + tstart))
        interval_rec[j] = tdiff # length in time each interval is recorded over
        summed_tint = np.nansum(norm_mode_coord, axis=1) # sum over time interval for each mode
        interval_summed[:, j] = summed_tint
        summed_sq_tint = np.nansum(norm_mode_coord**2, axis=1) # squared sum of normal modes over time interval
        interval_summed_sq[:, j] = summed_sq_tint

        if tdiff != 0:
            avg_tint = summed_tint / tdiff
            interval_avg[:, j] = avg_tint
            avg_sq_tint = summed_sq_tint / tdiff
            interval_std[:, j] = (tdiff / (tdiff - 1) * (avg_sq_tint - avg_tint ** 2)) ** .5

    return interval_rec, interval_std, interval_summed, interval_summed_sq, interval_avg


def count_traj_timesteps(trajectory: npt.NDArray, nts: int):
    # adds +1 to timesteps that are present in a trajectory
    # returns vector of size nts with 1 and 0 corrosponding to
    # if the trajectory calculation exists at specified time step
    time_count = np.zeros((nts), int)
    for i in range(nts):
        if not np.isnan(trajectory[:, :, i]).any():
            time_count[i] += 1
    return time_count


#def tint_averaging(interval_summed: npt.NDArray, interval_summed_sq: npt.NDArray,
#                   interval_rec: npt.NDArray, ntints: int, nvibs: int) -> tuple[npt.NDArray, npt.NDArray]:
#    # this function is redundant - used in old implimentation
#    std_tint = np.zeros((nvibs, ntints,), float)
#    tint_avg = np.zeros((nvibs, ntints), float)
#    for j in range(ntints):
#        int_avg = interval_summed[:, j] / interval_rec[j]
#        int_avg_sq = interval_summed_sq[:, j] / interval_rec[j]
#        std = (interval_rec[j] / (interval_rec[j] - 1) * (int_avg_sq - int_avg ** 2)) ** 0.5
#        std_tint[:, j] = std;
#        tint_avg[:, j] = int_avg;
#        # mult std by mult array here
#    return std_tint, tint_avg

