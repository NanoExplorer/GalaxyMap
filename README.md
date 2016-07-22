This is a project for doing statistical analysis on galaxy positions and velocities. (Mostly velocities!)

To run any python script in the Mayavi folder, you need to use python 2 and have Mayavi installed. (Canopy should work!)

To run any python script in the Matplot folder, you need to use python 3 and have matplotlib, numpy, and scipy installed. There may be others too. Python 3 is used because of its superior garbage collector, which is required for doing long-distance correlations on thousands of galaxies. 

The scripts in the matplot directory have many functions, including processing data from the Millennium Simulation, calculating the spatial and velocity correlations over the processed data, and surveying Millennium simulation data to make it look like real-life data. If you have trouble using any of it, feel free to post an issue! That'll give me an excuse to update all of the documentation.

A normal workflow is to create surveys from the Millennium Simulation data, then run the velocity correlations over that to find the cosmic variance. Then we perturb the surveys and rerun correlations for statistical error.

I try to keep the code optimized so that it runs in a reasonable amount of time even though it's generating and processing hundreds of gigabytes of data.
