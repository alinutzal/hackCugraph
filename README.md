# Basic Standalone libcudf C++ application

This C++ example demonstrates a basic libcudf use case and provides a minimal
example of building your own application based on libcudf using CMake.

The example source code loads a csv file that contains stock prices from 4
companies spanning across 5 days, computes the average of the closing price
for each company and writes the result in csv format.

## Compile and execute

1. Run `rapidsai` developer container
```bash
shifter --image=rapidsai/rapidsai-dev:21.06-cuda11.0-devel-ubuntu20.04-py3.8 bash
```

2. Clone the repository and run `cmake`. 
```bash
mkdir build && cd build && cmake ..
```
If everything works well, compile the code with `make -j4`.

3. Test the code in a GPU.
```bash
module load cgpu
srun -C gpu -N 1 -G 1 -c 10 -t 4:00:00 -A m1759 --pty /bin/bash -l
```

Export the dataset directory `export RAPIDS_DATASET_ROOT_DIR=/global/cfs/cdirs/atlas/xju/software/hackCugraph/datasets`


If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
