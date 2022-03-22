import numpy as np
import numpy.typing as npt
import vibrations as vib
import trajectories as trj
import json
import argparse
import textwrap


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-i", dest="input_file", required=False, type=str,
                    help=textwrap.dedent('''\
                    Input file in json format to be read in\n'''))


def read_json(inputfile: str) -> dict:
    with open(inputfile, 'r') as f:
        json_dict = json.load(f)
    return json_dict

def format_2d(arr: npt.NDArray, delim: str) -> str:
    sz = arr.shape
    nd = len(sz)
    if nd > 2:
        raise Exception("Not a 2d array.")
    rows = sz[0]
    str_list = []
    for row in range(rows):
        str_list.append(delim.join([str(elem) for elem in arr[row]]) + "\n")
    return "".join(str_list)

def format_3d(arr: npt.NDArray, axis: int, delim: str) -> str:
    if axis != 2:
        raise Exception("Must be along second axis dimension.")
    nd = arr.shape[axis]
    str_list = []
    for dim in range(nd):
        arr_slice = arr[:, :, dim]
        slice_str = format_2d(arr_slice, delim)
        str_list.append(slice_str + "\n")
    return "".join(str_list)

def write_arr(filename: str, arr: npt.NDArray):
    sz = len(arr.shape)
    if sz == 3:
        arr_str = format_3d(arr, axis=2, delim='\t')
    elif sz == 2:
        arr_str = format_2d(arr, delim='\t')
    with open(filename, 'w+') as f:
        f.write(arr_str)


def main(INPUTS):
    freq_file = INPUTS['freq_path']
    traj_file_path = INPUTS['traj_path']
    read_from_xyz = INPUTS['read_from_xyz']
    code_type = INPUTS['code_name'].lower()
    mass_weight = INPUTS['mass_weight']
    atmnum = INPUTS['atmnum']
    dt = INPUTS['timestep']
    nts = INPUTS['numsteps']
    time_intervals = INPUTS['interval']

    if not read_from_xyz and len(time_intervals) == 0:
        raise Exception("If loading predifined trajectories an external .mat or .npy file - provide a set of time intervals for analysis")

    natom = len(atmnum)
    ref_atoms, freqs, vibs, ref_norm_modes = vib.read_molden_freqs(freq_file)
    nvibs = len(vibs)
    norm_mode_mat = vib.get_norm_mode_mat(ref_norm_modes, nvibs, mass_weight, atmnum)
    if read_from_xyz:
        trajectories, run_times, ntraj = trj.read_trajs_from_xyz(traj_file_path, code_type, natom)
        if len(time_intervals) == 0:
            time_intervals = run_times
        # for xyz files get run_times into time_intervals format using either run_times or user spec. time intervals
        # add check to ensure user spec. time intervals exists for a given traj
    else:
        trajectories, ntraj = trj.load_trajs(traj_file_path, natom, nts)
        # can be str for file or list - if user spec. list, proj over all trajs
        # if file - can be a set of diff intervals for each traj - or a single interval to proj over all trajs

    time_intervals, ntints = trj.get_time_intervals(time_intervals, ntraj)

    norm_mode_trajs, traj_std, interval_avg, traj_time_count = single_traj_nma(trajectories, ref_atoms, norm_mode_mat, time_intervals, ntints, nvibs)

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

    avg_tint_std = np.zeros((nvibs, ntints), float)
    if len(np.unique(time_intervals, axis=2)) == ntints:
        for i in range(ntints):
            tint = time_intervals[:, i, 0]
            tstart, tend = int(tint[0]), int(tint[1])
            tdiff = tend - tstart
            avg_all_tint = np.nansum(traj_avg_norm_modes[:, tstart:tend], axis=1) / tdiff
            avg_sq_all_tint = np.nansum(traj_avg_norm_modes[:, tstart:tend]**2, axis=1) / tdiff
            avg_tint_std[:, i] = (tdiff/ (tdiff-1) * (avg_sq_all_tint - avg_all_tint ** 2)) ** 0.5
            # std of averaged trajectories within each time interval
    else:
        print("""Can not perform std calculation over specified time intervals for mean trajectories as time
              intervals are not consistant over all trajectories.""")

    write_arr('test_std.txt', avg_tint_std)


        #av_std_array = av_std_array * mult_array


def single_traj_nma(trajectories: npt.NDArray, ref_atoms: npt.NDArray,
                    norm_mode_mat: npt.NDArray, time_intervals: npt.NDArray, ntints: int, nvibs: int):
    # performs normal mode analysis on a single trajectory
    # does transformation to normal mode coordinates
    # also calculates the average traj in norm mode coords over a series of time intervals
    # calculates standard dev on the average over each time interval
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
        #traj_std_b[:, :, traj], interval_avg_b[:, :, traj] = tint_averaging(interval_summed, interval_summed_sq, interval_rec, ntints, nvibs)
        # for each traj - std and average of nm over time intervals
        traj_time_count += count_traj_timesteps(trajectory, nts)

    return norm_mode_trajs, traj_std, interval_avg, traj_time_count


def count_traj_timesteps(trajectory: npt.NDArray, nts: int):
    # adds +1 to timesteps that are present in a trajectory
    time_count = np.zeros((nts), int)
    for i in range(nts):
        if not np.isnan(trajectory[:, :, i]).any():
            time_count[i] += 1
    return time_count


def tint_averaging(interval_summed: npt.NDArray, interval_summed_sq: npt.NDArray,
                   interval_rec: npt.NDArray, ntints: int, nvibs: int) -> tuple[npt.NDArray, npt.NDArray]:
    std_tint = np.zeros((nvibs, ntints,), float)
    tint_avg = np.zeros((nvibs, ntints), float)
    for j in range(ntints):
        int_avg = interval_summed[:, j] / interval_rec[j]
        int_avg_sq = interval_summed_sq[:, j] / interval_rec[j]
        std = (interval_rec[j] / (interval_rec[j] - 1) * (int_avg_sq - int_avg ** 2)) ** 0.5
        std_tint[:, j] = std;
        tint_avg[:, j] = int_avg;
        # mult std by mult array here
    return std_tint, tint_avg


def calc_traj_std(norm_mode_coords: npt.NDArray, time_intervals: npt.NDArray, ntints: int, nvibs: int, traj: int):

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



if __name__ == "__main__":
    DEBUG = True
    args = parser.parse_args()
    if args.input_file is not None:
        INPUTS = read_json(args.input_file)
    elif args.input_file is None and DEBUG:
        INPUTS = read_json("INPUTS.json")
    else:
        raise Exception("Please provide an input file as an argument.")

    main(INPUTS)
