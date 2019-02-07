# Playne-Equivalence Algorithm
--------------------

An illustrative CUDA implementation of the Playne-Equivalence Connected-Component Labelling Algorithm described in:

D. P. Playne and K. Hawick,<br/>
"A New Algorithm for Parallel Connected-Component Labelling on GPUs,"<br/>
in IEEE Transactions on Parallel and Distributed Systems,<br/>
vol. 29, no. 6, pp. 1217-1230, 1 June 2018.<br/>
* URL: https://ieeexplore.ieee.org/document/8274991

Code has been updated for CUDA 10.0


Usage
--------------------
The examples are written as independent programs for the Label-Equivalence algorithm, the Playne-Equivalence algorithm with both the Direct and Block methods for 2D and 3D with clamped boundary conditions.

Usage:
./\<method\> \<gpu-device\> \<input-files...\> 

A simple makefile is provided to compile the examples.


Disclaimer
--------------------
The source code is provided "as is" for the purpose of illustration only and is not intended to be production-ready code. Please read the license for full details.


License
--------------------
The source code is provided under The MIT license (see LICENSE.txt)
