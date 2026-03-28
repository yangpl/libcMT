#!/bin/bash

python3 make_model_3d.py
python3 make_acquisition.py
python3 make_freqs.py


#mpirun -np 2 ../bin/libcMT ffreqs=freqs.h5 fmodel=model.h5 frec=receivers.h5

../bin/libcMT freqs=0.01,0.1,1,10,100 fmodel=model.h5 frec=receivers.h5
