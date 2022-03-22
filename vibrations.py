import numpy as np
import numpy.typing as npt

#au2ang = .529177
au2ang = 1


def read_molden_freqs(file_path: str) -> tuple[list, list, list, npt.NDArray]:

    atoms, freqs, vibs = [], [], []

    mfile = open(file_path, 'r')
    Atoms = False
    FREQ = False
    FRNORMCOORD = False

    actvib = -1
    for line in mfile:
        # what section are we in
        if '[' in line or '--' in  line:
            Atoms = False
            FREQ = False
            FRNORMCOORD = False

        if '[Atoms]' in line:
            Atoms = True
        elif '[FREQ]' in line:
            FREQ = True
        elif '[FR-NORM-COORD]' in line:
            FRNORMCOORD = True
        # extract the information in that section
        elif Atoms:
            words = line.split()
            atom_vec = words[3:6]
            atoms += [[eval(coords) for coords in atom_vec]]
        elif FREQ:
            freqs += [eval(line)]
        elif FRNORMCOORD:
            if 'vibration' in line or 'Vibration' in line:
                vib_list = []
                actvib += 1
                if actvib > -1:
                    vibs += [vib_list]
            else:
                vib_list += [[eval(coor) for coor in line.split()]]

    natoms = len(atoms)
    nfreqs = len(freqs)
    flattened_vibs = vibs
    while len(flattened_vibs) != (natoms * nfreqs * 3):
        flattened_vibs = [item for items in flattened_vibs for item in items]
    norm_modes = np.reshape(flattened_vibs, (nfreqs, natoms*3))
    atoms = [item for items in atoms for item in items]
    atoms = np.array(atoms)

    return atoms, freqs, vibs, norm_modes


def get_norm_mode_mat(norm_modes, nvibs: int, mass_weight: bool, atmnum: list) -> npt.NDArray:

    natom = len(atmnum)
    norm_mode_mat = np.linalg.pinv(norm_modes)

    if mass_weight:
        mass_mat = mass_matrix(atmnum, natom, nvibs)
        norm_mode_mat = np.dot(mass_mat, norm_mode_mat)

    return norm_mode_mat


def mass_matrix(atmnum: list, natom: int, nfreqs: int) -> npt.NDArray:
    mw_atom_vec = np.array([a**0.5 for a in atmnum for i in range(3)])
    mass_mat = np.eye(nfreqs, natom*3) * mw_atom_vec
    return mass_mat


def norm_mode_basis(trajectory, nts, nvib, ref_struct, norm_mode_mat):
    norm_mode_coords = np.zeros((nvib,nts))
    for ts in range(nts):
        geom = trajectory[:,:,ts].flatten()
        if np.isnan(geom).any():
            norm_mode_coords[:, ts] = np.squeeze(np.full((nvib, 1), np.nan))
            break
        else:
            displacement_vec = geom - ref_struct
            norm_mode_coords[:,ts] = np.dot(displacement_vec, norm_mode_mat)
    return norm_mode_coords


def get_multiplication_array(nvib: int):
    multiplication_array = numpy.zeros(nvib, float) # for final summary files, output is multiplied with this list
    for i in range(nvib):
        if i+1 in neg_list:
            multiplication_array[i] = -1
        else:
            multiplication_array[i] = 1
    return multiplication_array