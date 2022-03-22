import numpy as np
import scipy.io as sio
import numpy.typing as npt
from typing import Union
import os
import natsort
from io import StringIO


def read_trajs_from_xyz(fpaths: str, code_type: str, natom: int) -> tuple[npt.NDArray, npt.NDArray, int]:
    # reads trajectories specified in every subdir listed in the parent dir fpaths
    # filters all possible subdirs down to suitable trajectory ones on the basis they contain the file traj_file and it is readable
    # by default it currently reads only SHARC files but can be easily extended
    # returns a np array of shape [natom, 3, ntraj, nts] where nts = maximum run time over all trajs read in
    # trajs that crash before nts have subsequent elements set to NaN
    # also returns a list of each trajs run time and the total num. of trajs
    if code_type == 'sharc':
        traj_file = 'output.xyz'
    else:
        print("Only SHARC trajectories currently available.")

    dirs = [(fpaths + '/' + d) for d in os.listdir(fpaths) if os.path.isdir(fpaths + '/' + d)]
    dirs = [d for d in dirs if os.path.isfile(d + '/' + traj_file)]
    dirs = natsort.natsorted(dirs)
    if len(dirs) < 1:
        raise Exception("No directories containing trajectories found. Please check cwd.")
    fpaths = [(f + '/' + traj_file) for f in dirs]
    print("Attempting to read in %s trajectories from xyz files" % len(fpaths))
    all_trajs, run_times, ntraj = load_trajs_xyz(fpaths, natom, 'sharc') # currently default is SHARC only
    trajs = format_traj_array(all_trajs, run_times, natom, ntraj)
    run_times = format_traj_run_times(run_times)
    return trajs, run_times, ntraj


def format_traj_array(traj_list: list, run_times: list, natom : int, ntraj : int) -> npt.NDArray:
    # formats nested list of trajectories into a np array of dimensions (natom, 3, ntraj, nts)
    # trajs might not all run over time frame, therefore nts is taken to be the longest
    # run time of all trajectories together. For trajectories that crash before the int nts,
    # the subsequent array indexes are filled with NaN

    max_run_time = max(run_times)
    trajs = np.full((natom, 3, ntraj, max_run_time), np.nan)
    for i in range(ntraj):
        run_time = run_times[i]
        trajs[:, :, i, 0:run_time] = np.swapaxes(np.swapaxes(np.array(traj_list[i]), 0, 2), 0, 1) # must do some reshaping
    return trajs


def load_trajs_xyz(fpaths: list, natom: int, traj_type: str = 'sharc') -> tuple[list, list, int]:
    # iterates over file paths for all trajectories in subdirs and returns
    # all suitable trajs in a nested list along with the run time for each traj simulation and the total traj number
    # currently only works for SHARC formatted trajectory files (see call to read_sharc() func)
    # can further extend to other quantum codes by adding a suitable function that returns a
    # list of trajs and their corrosponding run times
    run_times = []
    all_trajs = []
    ntraj = 0
    for idx, fpath in enumerate(fpaths):
        with open(fpath, 'r') as trj_file:

            if traj_type.lower() == 'sharc':
                traj_coords, traj_nts = read_sharc(trj_file, natom)
            else:
                pass # extend to other quantum md code readers if needed - write another function like read_sharc to call

            if traj_nts > 1:
                ntraj += 1
                run_times.append(traj_nts)
                all_trajs.append(traj_coords)
            else:
                print("%s contains less than one time step - excluding from analysis." % fpath)
    return all_trajs, run_times, ntraj


def read_sharc(trj_file: str, natom: int) -> tuple[list, int]:
    # reads SHARC style trajectory files in xyz files
    # returns a list of a single trajectories coordinates and its run time
    count = 0
    na = 0
    nts = 0
    geom_all = []
    geom_temp = []
    for idx, line in enumerate(trj_file):
        idx += 1
        count += 1
        if count > 2:
            atom_coord = np.genfromtxt(StringIO(line))[1:4].tolist()
            geom_temp.append(atom_coord)
            na += 1
        if na == natom:
            na = 0
            count = 0
            nts += 1
            geom_all.append(geom_temp)
            geom_temp = []

    return geom_all, nts


def read_mat(filename: str, var_names: list):
    # parses and reads in input arrays in .mat or .npy format
    file_type = filename.split('.')[-1]
    if file_type == 'npy':
        arr = np.load(filename)
    elif file_type == 'mat':
        arr = sio.loadmat(filename)
        keys = [k for k in arr if '__' not in k]
        if len(keys) > 1 and var_names is None:
            raise TypeError('.Mat file must have only one variable')
        elif len(keys) > 1 and var_names is not None:
            arr = get_mat_vars(arr, var_names, keys)
        else:
            key_val = keys[0]
            arr = arr[key_val]
    else:
        raise TypeError('Input file must be .npy or .mat')

    return arr


def get_mat_vars(arr: dict, var_names: list, keys: list) -> list:
    # extracts only desired variables from a .mat file
    # INPUTS:
    # Vars - variables to be extracted from .mat file
    # Arr - .mat file variables loaded into a dict
    # Keys - all variables listed in Arr
    # OUTPUTS:
    # arrs - list of np.arrays of each variable
    inds = [i for i, k in enumerate(keys) if k in var_names]
    arrs = []
    for i in range(len(inds)):
        arrs.append(arr[keys[inds[i]]])
    return arrs


def load_trajs(traj_file: str, natom:int, nts: int) -> tuple[npt.NDArray, int]:
    trajectories = read_mat(traj_file, [])
    trz = np.shape(trajectories)
    if trz[0] != natom and trz[1] != 3 and trz[3] != nts:
        raise Exception("Trajectories must be stored in format [natom, 3, ntraj, nts]")
    ntraj = trz[2]
    return trajectories, ntraj


def format_traj_run_times(run_times: list) -> npt.NDArray:
    # formats run_times read from xyz files for a series of trajs
    # into a time interval array
    ntraj = len(run_times)
    time_intervals = np.zeros((2, 1, ntraj))
    for traj in range(ntraj):
        end_time = run_times[traj]
        tint = np.array([0, end_time])
        time_intervals[:, :, traj] = np.zeros((2, 1)) + tint[:, None]
    return time_intervals



def load_time_intervals(tints_file: str, ntraj: int) -> tuple[npt.NDArray, int]:
    # load time_intervals from an existing file and format by projecting each interval
    # over all trajectories if required. Always returns tints in form [2, num_intervals, ntraj]
    # where the intervals for each traj can be the same, or different - depending on input array
    tints = read_mat(tints_file, [])
    tisz = np.shape(tints)
    ntints = tisz[1]
    if tisz[2] != ntraj and tisz[2] == 1:
        tints = np.zeros((2, ntints, ntraj)) + tints[:, :, None] # project the ntints time intervals over all trajs
    elif tisz[2] != ntraj and tisz[2] != 1: # error if incorrect format
        raise Exception("Time interval array must be in format [2, num_intervals, x] where x = 1 or ntraj.")
    else:
        pass # if already in format [2, num_intervals, ntraj] do nothing
    return tints, ntints



def format_time_intervals(tints: list, ntraj: int) -> tuple[npt.NDArray, int]:
    # if time intervals given in a user specified list and NOT read in from external file
    # project every time interval for all trajs - each traj will have the same time intervals
    ntints = len(tints)
    tints_arr = np.zeros((2, ntints, ntraj))
    for i in range(ntints):
        tint = np.array(tints[i])
        tints_arr[:, i, :] = np.zeros((2,ntraj)) + tint[:,None]
    return tints_arr, ntints


def get_time_intervals(time_intervals: Union[str, list], ntraj: int) -> tuple[npt.NDArray, int]:
    # if a string is specified - read time intervals from external file
    # else calculate them over all trajs for the user specified list
    if type(time_intervals) == str: # load from external file
        time_intervals, ntints = load_time_intervals(time_intervals, ntraj)
    else:
        time_intervals, ntints = format_time_intervals(time_intervals, ntraj)
    return time_intervals, ntints


if __name__ == "__main__":
    cwd = '/Users/kyleacheson/PycharmProjects/SHARC_NMA'
    atoms = ["C", "S","S"]
    read_trajs_from_xyz(cwd, 'output.xyz', atoms)