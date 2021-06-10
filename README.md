# event-based-dataset-h5
## Dataset
*N-MNIST*

*N-CARS*

*CIFAR10-DVS*
## Description

This project using other people's work to create a h5 file from raw data of Event-based dataset.

*load_events.py*
                
                load_bin_events():support .bin file(N-MNIST).Using 【aertb】 https://github.com/rfma23/aertb                
                load_aedat_events():support .aedat file(CIFAR10-DVS).Using 【events_tfds】 https://github.com/jackd/events-tfds
                
*load_dat_prophesee.py*

                load_dat_events():support .dat file(N-CARS).Using【prophesee-automotive-dataset-toolbox】https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
