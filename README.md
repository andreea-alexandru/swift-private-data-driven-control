# swift-private-data-driven-control
Encrypted computations for client-server private data-driven control.

Proof of concept for the solutions proposed in the paper "Data-driven control on encrypted data" by Andreea B. Alexandru, Anastasios Tsiamis and George J. Pappas, available at https://arxiv.org/pdf/2008.12671.pdf. This is an improvement in both runtime and memory compared to the code in the private-data-driven-control repository.

I implemented the solutions in the PALISADE library https://gitlab.com/palisade/palisade-development, version 1.9.1. Please download and install the library in order to run this code.

Files:
  - optimized-data-driven-control.cpp: contains the encrypted solutions for the offline and online feedback control
  - Plant.h: contains the functions related to the simulation of the plant
  - helperControl.h: contains functions related to the encrypted computations
  - plantData folder: contains examples of files for two systems, used in optimized-data-driven-control.cpp. Make sure that the path of this folder is correctly updated at the beginning of data-driven-control.cpp.
  - matrix.h, matrix.cpp: these files already exist in PALISADE. I added some extra functionalities for the Matrix class. These changes are signaled by comments like "Andreea Alexandru added this".
  - pubkeylp.h, cryptocontext.h, ckks.h, ckks-impl.cpp: these files already exist in PALISADE. They include some modifications of EvalFastRotation, EvalFastRotationHybrid and KeySwitchHybrid. These changes will be included in a future version of PALISADE-development. 
