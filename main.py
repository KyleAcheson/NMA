import numpy as np
import numpy.typing as npt
import vibrations as vib
import trajectories as trj
import readers as fread
import argparse
import textwrap



parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-i", dest="input_file", required=False, type=str,
                    help=textwrap.dedent('''\
                    Input file in json format to be read in\n'''))


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
    ref_atoms, freqs, vibs, ref_norm_modes = vib.molden_freqs(freq_file)
    nvibs = len(vibs)
    norm_mode_mat = vib.get_norm_mode_mat(ref_norm_modes, nvibs, mass_weight, atmnum)
    if read_from_xyz:
        trajectories, run_times, ntraj = fread.read_trajs_from_xyz(traj_file_path, code_type, natom)
        if len(time_intervals) == 0:
            time_intervals = run_times
        # for xyz files get run_times into time_intervals format using either run_times or user spec. time intervals
        # add check to ensure user spec. time intervals exists for a given traj
    else:
        trajectories, ntraj = fread.load_trajs(traj_file_path, natom, nts)
        # can be str for file or list - if user spec. list, proj over all trajs
        # if file - can be a set of diff intervals for each traj - or a single interval to proj over all trajs

    time_intervals, ntints = fread.get_time_intervals(time_intervals, ntraj)

    norm_mode_trajs, traj_std, interval_avg, traj_time_count = trj.single_traj_nma(trajectories, ref_atoms, norm_mode_mat, time_intervals, ntints, nvibs)
    traj_avg_std, avg_norm_modes, avg_tint_std = trj.ensemble_analysis(norm_mode_trajs, traj_time_count, time_intervals)

    if avg_tint_std is not None:
        pass





if __name__ == "__main__":
    DEBUG = True
    args = parser.parse_args()
    if args.input_file is not None:
        INPUTS = read_json(args.input_file)
    elif args.input_file is None and DEBUG:
        INPUTS = fread.read_json("INPUTS.json")
    else:
        raise Exception("Please provide an input file as an argument.")

    main(INPUTS)
