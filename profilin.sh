nsys profile --stats=true --trace=cuda,nvtx,openacc \
--output torchfort-noum --force-overwrite true \
mpirun -np 2 ./train_distributed --ntrain_steps 1000 --nval_steps 1000

nsys profile --stats=true --trace=cuda,nvtx,openacc \
--cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true \
--output torchfort-um --force-overwrite true \
mpirun -np 2 ./train_distributed_um --ntrain_steps 1000 --nval_steps 1000

nsys profile --stats=true --trace=cuda,nvtx,openacc \
--cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true \
--output torchfort-umt --force-overwrite true \
mpirun -np 2 ./train_distributed_um --ntrain_steps 1000 --nval_steps 1000 --tuning

nsys profile --stats=true --trace=cuda,nvtx,openacc \
--cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true \
--output torchfort-umt-preadv --force-overwrite true \
mpirun -np 2 ./train_distributed_um --ntrain_steps 1000 --nval_steps 1000 --tuning

nsys profile --stats=true --trace=cuda,nvtx,openacc \
--cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true \
--output torchfort-um-nohostdata --force-overwrite true \
mpirun -np 2 ./train_distributed_um --ntrain_steps 1000 --nval_steps 1000
