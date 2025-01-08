#!/bin/bash -l

#PBS -A SEEr-Polaris
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle:grand
#PBS -N train_dis_um_UM_1
#PBS -m bea
#PBS -M rstrin4@uic.edu
#PBS -V
#PBS -l select=1:ncpus=4:ngpus=4
#PBS -r y


export LD_LIBRARY_PATH=/soft/libraries/libtorch/libtorch-2.4.0+cu124/lib:$HOME/TorchFort-def/lib:$LD_LIBRARY_PATH


export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_RDMA_ENABLED_CUDA=1
module load craype-accel-nvidia80
module load cray-hdf5/1.12.2.9

# Number of GPUs per rank --> 1:1
ngpus=1
# Number of ranks per node
nranks=4
# Medium batch size is the only one picked for multinode tests



# Log dir path
log_path=$HOME/TorchFort/train_dis_log

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi

NNODES=$(wc -l < $PBS_NODEFILE)
NTOTRANKS=$(( NNODES * nranks ))

cd $HOME/TorchFort-def/bin/examples/fortran/simulation


for size in 16 32 64; do

for i in {1..5}; do
    mpirun --envall --np "${NTOTRANKS}" --ppn "${nranks}" \
    --hostfile "$PBS_NODEFILE" --cpu-bind list:0,8,16,24 \
    ./train_distributed_um --size $size --batch $size > "${log_path}/UM_gpus_${NTOTRANKS}_size_${size}_iter_${i}.txt"
done

done
