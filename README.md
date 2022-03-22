# NMA

Python module for performing analysis of quantum dynamics trajectories in a normal mode basis

## Features:

- Transform a set of input trajectories into a normal basis given a input frequencey calculation in molden format
- Reads trajectories from SHARC xyz files directly, or from a user input .npy or .mat file
- Specify a set of time intervals to break each trajectory down into to average over
- Can also input an array in .npy or .mat format for a set of time intervals
- Input time intervals can be the same for every trajectory, or a different set of intervals (useful for dissociation)
- For a given set of time intervals the time average of each normal mode coordinate is calculated along with its standard deviation
- Total average and standard deviation for the given intervals is calculated from all trajectories and for each one individually
- Calculates the average and standard deviation of each coordinate over time over all available time steps

- Given the deviation of the normal mode displacements over time - one can reduce the dimensionality of the system using this information

### TODO:
- Add routines to perform an SVD of the normal mode coordinates as a function of time into a time eigenbasis
- Use singular values to measure the significance of each mode in each trajectory
- Interface to clustering modules to cluster trajectories based on this
