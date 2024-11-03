### Steps

- ~Create a new file called `train_distributed_um` which will be the program to implement UM and accepts tuning as a param~
- Verify that UM works by removing those !acc pragma
- Since MPI_AlltoAllv is being used try to understand whether a NCCL communication can be fully implemented instead
- Add cudeevents to measure time
