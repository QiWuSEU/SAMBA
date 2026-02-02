## Overview

SAMBA is an optimization algorithm for the miniaturized patch antenna area with constraints, implemented by calling the CST interface. The algorithm is also applicable to other miniaturization problems; however, modifications are required for the calculation of cheap objectives such as area and the setup of constraints. For cheap objectives, they are typically defined as those whose values can be determined through analytical formulas.

## Optimization Framework and Utilities

### CST Interface 
After building the model in CST, the files ```CSTScriptGenerator```, ```function_call_CST_parallel``` and ```function_call_CST``` are all related to calling the CST interface. In the ```function_call_CST_parallel``` file, the number of parallel simulations can be set based on the configuration of your computer.

In my example, since the constraints for antenna optimization are reflection coefficient and gain, these files allow precise extraction of such data. For constraints in other optimization problems, adjustments may be required.
    
### Data Reading
The files ```function_read_results``` and ```read_results```implement the functionality to extract the corresponding data from the exported ```snp``` and ```txt``` files. Additionally, the area calculation and constraint satisfaction scoring are also implemented in ```read_results```.

### Dataset Sampling
The files ```LHSsample_file``` and ```sampling_main```are used to sample and construct the training set samples required for Gaussian Process Regression (GPR). The ```LHSsample_file``` performs Latin Hypercube Sampling based on the variable ranges, while ```sampling_main``` serves as the main function for the sampling process.

### Miniaturization Optimization
 ```optimize_main``` is the main function for implementing miniaturization optimization and ```objective_function``` is the objective function setup required for the multi-objective optimization algorithm.

## Examples
The antenna1 and antenna2 files in the ```Example``` folder are optimization examples used by the algorithm. The referenced papers can be found at the following URLs:
https://ieeexplore.ieee.org/abstract/document/9373969 and https://ieeexplore.ieee.org/abstract/document/9877920

    @ARTICLE{9373969,
      author={Shao, Zijian and Zhang, Yueping},
      journal={IEEE Antennas and Wireless Propagation Letters}, 
      title={A Single-Layer Miniaturized Patch Antenna Based on Coupled Microstrips}, 
      year={2021},
      volume={20},
      number={5},
      pages={823-827},
      keywords={Patch antennas;Antennas;Microstrip;Antenna radiation patterns;Microstrip antennas;Loading;Magnetic resonance;Coupled lines;microstrip antennas;miniaturized antennas;slow wave},

    @ARTICLE{9877920,
      author={Shao, Zijian and Hou, Yaowei and Fang, Yulin and Zhang, Yueping},
      journal={IEEE Transactions on Magnetics}, 
      title={Miniaturization and Cross-Polarization Reduction of Quarter-Wave Microstrip Antennas Based on Magnetic Currents Reconstruction}, 
      year={2022},
      volume={58},
      number={11},
      pages={1-6},
      keywords={Microstrip antennas;Microstrip;Antenna measurements;Magnetic resonance;Electric fields;Antennas;Frequency measurement;Cross-polarization reduction;magnetic currents reconstruction;microstrip antennas;miniaturization},
