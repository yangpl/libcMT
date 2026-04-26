#!/bin/bash
rm -f *.log

python3 make_models_3d.py 
python3 make_acquisition.py
python3 plot_models_3d.py --save-prefix model_3d

#mode=0, modelling in true model model_true.h5 to obtain observed data, saved as mt_data.h5
#mode=1, inversion starting from initial model model_init.h5, create inversion.log, model_recovered.h5
#mode=2, output inversion gradient only
../bin/libcMT \
    mode=1 \
    freqs=1 \
    fmodel=model_init.h5 \
    frec=receivers.h5 \
    fdata=mt_data.h5

#python3 plot_gradient_3d.py gradient_iter0000.h5 --component grad_mh
#python3 plot_gradient_3d.py gradient_iter0000.h5 --component grad_mv

#python3 plot_iterate.py iterate.txt inversion_convergence.png
#python3 extract_final_model.py model_init.h5 inversion.log model_recovered.h5

