# TorchFort Integration Guide for WRF (Weather Research and Forecasting Model)

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Integration Overview](#integration-overview)
- [Step-by-Step Integration](#step-by-step-integration)
- [Use Cases](#use-cases)
- [Performance Considerations](#performance-considerations)
- [Example Implementation](#example-implementation)
- [Troubleshooting](#troubleshooting)

## Introduction

This guide demonstrates how to integrate TorchFort into the Weather Research and Forecasting (WRF) model to enable online deep learning capabilities. WRF is a mesoscale numerical weather prediction system designed for both atmospheric research and operational forecasting applications. By integrating TorchFort, you can:

- Train ML models on-the-fly using WRF simulation data
- Replace expensive physical parameterizations with fast ML surrogates
- Learn data-driven closures for unresolved processes
- Implement ML-based bias correction and post-processing

## Prerequisites

### Software Requirements

1. **WRF Model** (v4.0 or later recommended)
   - Source code available at: https://github.com/wrf-model/WRF
   - Follow standard WRF installation procedures

2. **TorchFort Library**
   - Built and installed following the [TorchFort installation guide](installation.rst)
   - Ensure GPU support is enabled if using NVIDIA GPUs

3. **Compilers**
   - Fortran compiler (gfortran, ifort, or nvfortran)
   - C/C++ compiler compatible with TorchFort
   - MPI library (required for distributed WRF runs)

4. **CUDA Toolkit** (optional, for GPU acceleration)
   - CUDA 11.0 or later
   - cuDNN library

### Knowledge Requirements

- Familiarity with WRF model structure and physics packages
- Basic understanding of Fortran programming
- Knowledge of deep learning concepts and PyTorch
- Understanding of YAML configuration files

## Integration Overview

### Where to Integrate TorchFort in WRF

TorchFort can be integrated at various points in the WRF workflow:

1. **Physics Parameterization Layer**: Replace or augment existing physics schemes (e.g., cumulus, PBL, microphysics)
2. **Dynamics Core**: Learn sub-grid scale processes or apply ML-based corrections
3. **Data Assimilation**: Enhance observations or model states using ML
4. **Post-Processing**: Apply ML-based bias correction or downscaling

### Integration Architecture

```
┌─────────────────────────────────────┐
│      WRF Main Program (Fortran)     │
│                                     │
│  ┌───────────────────────────────┐ │
│  │   Physics Package (e.g., PBL) │ │
│  │                               │ │
│  │  ┌─────────────────────────┐  │ │
│  │  │  TorchFort Interface    │  │ │
│  │  │  - torchfort_train()    │  │ │
│  │  │  - torchfort_inference()│  │ │
│  │  └─────────────────────────┘  │ │
│  │                               │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │   PyTorch Models via          │ │
│  │   TorchScript (.pt files)     │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
```

## Step-by-Step Integration

### Step 1: Modify WRF Build System

#### 1.1 Update configure.wrf

Add TorchFort library paths and linker flags to your WRF build configuration:

```makefile
# Add to configure.wrf after running ./configure

# TorchFort library path
TORCHFORT_ROOT = /path/to/torchfort/installation
TORCHFORT_INC = -I$(TORCHFORT_ROOT)/include
TORCHFORT_LIB = -L$(TORCHFORT_ROOT)/lib -ltorchfort

# Update LIB_EXTERNAL
LIB_EXTERNAL = ... $(TORCHFORT_LIB) -lstdc++

# Update INCLUDE_MODULES
INCLUDE_MODULES = ... $(TORCHFORT_INC)
```

#### 1.2 Modify Makefile Dependencies

Ensure that modules calling TorchFort are properly linked:

```makefile
# In phys/Makefile or dyn_em/Makefile
module_bl_mynn.o: $(TORCHFORT_ROOT)/include/torchfort.h
```

### Step 2: Add TorchFort Module Interface

Create a Fortran module to interface with TorchFort. This encapsulates TorchFort calls and makes integration cleaner.

Create file: `phys/module_torchfort_interface.F`

```fortran
MODULE module_torchfort_interface
   USE iso_c_binding
   IMPLICIT NONE

   ! TorchFort interface declarations
   INTERFACE
      INTEGER(c_int) FUNCTION torchfort_create_model(model_name, config_file, device) &
                     BIND(C, name="torchfort_create_model")
         USE iso_c_binding
         CHARACTER(c_char) :: model_name(*), config_file(*)
         INTEGER(c_int), VALUE :: device
      END FUNCTION torchfort_create_model

      INTEGER(c_int) FUNCTION torchfort_train(model_name, input, label, loss, stream) &
                     BIND(C, name="torchfort_train")
         USE iso_c_binding
         CHARACTER(c_char) :: model_name(*)
         REAL(c_float) :: input(*), label(*)
         REAL(c_float) :: loss
         INTEGER(c_int), VALUE :: stream
      END FUNCTION torchfort_train

      INTEGER(c_int) FUNCTION torchfort_inference(model_name, input, output, stream) &
                     BIND(C, name="torchfort_inference")
         USE iso_c_binding
         CHARACTER(c_char) :: model_name(*)
         REAL(c_float) :: input(*), output(*)
         INTEGER(c_int), VALUE :: stream
      END FUNCTION torchfort_inference

      INTEGER(c_int) FUNCTION torchfort_save_checkpoint(model_name, checkpoint_dir) &
                     BIND(C, name="torchfort_save_checkpoint")
         USE iso_c_binding
         CHARACTER(c_char) :: model_name(*), checkpoint_dir(*)
      END FUNCTION torchfort_save_checkpoint

      INTEGER(c_int) FUNCTION torchfort_load_checkpoint(model_name, checkpoint_dir, &
                                                        step_train, step_inference) &
                     BIND(C, name="torchfort_load_checkpoint")
         USE iso_c_binding
         CHARACTER(c_char) :: model_name(*), checkpoint_dir(*)
         INTEGER(c_int64_t) :: step_train, step_inference
      END FUNCTION torchfort_load_checkpoint
   END INTERFACE

   LOGICAL :: torchfort_initialized = .FALSE.
   CHARACTER(LEN=256) :: torchfort_config_file
   INTEGER :: torchfort_device_id = 0

CONTAINS

   SUBROUTINE torchfort_wrf_init(config_file, device_id, ierr)
      CHARACTER(LEN=*), INTENT(IN) :: config_file
      INTEGER, INTENT(IN) :: device_id
      INTEGER, INTENT(OUT) :: ierr

      torchfort_config_file = config_file
      torchfort_device_id = device_id
      torchfort_initialized = .TRUE.
      ierr = 0
   END SUBROUTINE torchfort_wrf_init

   SUBROUTINE torchfort_wrf_finalize()
      torchfort_initialized = .FALSE.
   END SUBROUTINE torchfort_wrf_finalize

END MODULE module_torchfort_interface
```

### Step 3: Integrate into Physics Schemes

Here's an example of integrating TorchFort into a planetary boundary layer (PBL) scheme:

#### 3.1 Modify Physics Scheme Module

Edit the physics scheme file (e.g., `phys/module_bl_mynn.F`):

```fortran
SUBROUTINE mynn_bl_driver( ... )
   USE module_torchfort_interface

   ! Existing WRF variables
   REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(INOUT) :: &
        qc, qi, th, qv

   ! TorchFort variables
   REAL, ALLOCATABLE :: ml_input(:), ml_output(:)
   REAL :: loss_value
   INTEGER :: istat, i, j, k, idx
   LOGICAL :: use_ml_pbl

   ! Check if ML-based PBL is enabled (via namelist)
   use_ml_pbl = .FALSE.  ! Set via namelist.input

   IF (use_ml_pbl .AND. torchfort_initialized) THEN
      ! Prepare input features for ML model
      ! Example: vertical profiles of temperature, moisture, wind
      ALLOCATE(ml_input(kme-kms+1))
      ALLOCATE(ml_output(kme-kms+1))

      DO j = jts, jte
         DO i = its, ite
            ! Pack input data (e.g., temperature profile)
            DO k = kts, kte
               idx = k - kts + 1
               ml_input(idx) = th(i,k,j)
            ENDDO

            ! Run inference to get ML prediction
            istat = torchfort_inference("pbl_model"//C_NULL_CHAR, &
                                       ml_input, ml_output, 0)

            IF (istat == 0) THEN
               ! Apply ML correction or replacement
               DO k = kts, kte
                  idx = k - kts + 1
                  ! Example: add ML-predicted tendency
                  th(i,k,j) = th(i,k,j) + ml_output(idx)
               ENDDO
            ENDIF
         ENDDO
      ENDDO

      DEALLOCATE(ml_input, ml_output)
   ELSE
      ! Use traditional physics parameterization
      CALL traditional_pbl_scheme( ... )
   ENDIF

END SUBROUTINE mynn_bl_driver
```

#### 3.2 Add Training Loop (Optional)

For online training, add a training phase:

```fortran
SUBROUTINE mynn_bl_driver_train( ... )
   USE module_torchfort_interface

   REAL, ALLOCATABLE :: ml_input(:), ml_label(:)
   REAL :: loss_value
   INTEGER :: istat, training_interval

   training_interval = 100  ! Train every 100 timesteps

   IF (MOD(current_timestep, training_interval) == 0) THEN
      ! Prepare training data
      ALLOCATE(ml_input(input_size))
      ALLOCATE(ml_label(output_size))

      ! Populate input features from WRF state
      CALL prepare_training_data(ml_input, ml_label)

      ! Train the model
      istat = torchfort_train("pbl_model"//C_NULL_CHAR, &
                             ml_input, ml_label, loss_value, 0)

      IF (istat == 0) THEN
         PRINT *, 'Training loss:', loss_value
      ENDIF

      DEALLOCATE(ml_input, ml_label)
   ENDIF

END SUBROUTINE mynn_bl_driver_train
```

### Step 4: Configure WRF Namelist

Add TorchFort-specific options to `namelist.input`:

```fortran
&physics
 mp_physics                          = 8,
 ra_lw_physics                       = 4,
 ra_sw_physics                       = 4,
 bl_pbl_physics                      = 5,  ! Your physics scheme
 use_ml_pbl                          = .true.,  ! Enable ML PBL
 torchfort_config                    = 'torchfort_pbl.yaml',
 torchfort_device                    = 0,  ! GPU device ID
 /
```

### Step 5: Create TorchFort Configuration File

Create `torchfort_pbl.yaml`:

```yaml
model:
  type: torchscript
  torchscript_file: pbl_model.pt

optimizer:
  type: Adam
  parameters:
    lr: 0.001
    betas: [0.9, 0.999]
    weight_decay: 0.0001

loss:
  type: mse

lr_scheduler:
  type: ReduceLROnPlateau
  parameters:
    mode: min
    factor: 0.5
    patience: 10
    verbose: true

training:
  batch_size: 16
  enable_amp: true  # Automatic Mixed Precision
```

### Step 6: Prepare PyTorch Model

Create your model in PyTorch and export to TorchScript:

```python
# generate_pbl_model.py
import torch
import torch.nn as nn

class PBLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PBLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# Create and export model
model = PBLModel(input_size=50, hidden_size=128, output_size=50)
scripted_model = torch.jit.script(model)
scripted_model.save('pbl_model.pt')
print("Model saved to pbl_model.pt")
```

Run: `python generate_pbl_model.py`

### Step 7: Build and Run WRF

```bash
# Build WRF with TorchFort
cd WRF
./clean -a
./configure  # Select your configuration
# Edit configure.wrf to add TorchFort flags (Step 1)
./compile em_real >& compile.log

# Run WRF
cd run
# Ensure pbl_model.pt and torchfort_pbl.yaml are in run directory
mpirun -np 4 ./wrf.exe
```

## Use Cases

### 1. ML-Enhanced Microphysics

Replace or augment microphysics schemes with ML models trained on high-resolution simulations:

```fortran
! In module_mp_thompson.F
istat = torchfort_inference("microphysics_model"//C_NULL_CHAR, &
                           mp_input, mp_output, 0)
! Apply ML-predicted precipitation rates
```

### 2. Turbulence Parameterization

Learn sub-grid scale turbulence closures:

```fortran
! In module_bl_mynn.F
! Train on LES data to predict turbulent fluxes
istat = torchfort_train("turbulence_model"//C_NULL_CHAR, &
                       turb_input, turb_label, loss, 0)
```

### 3. Radiation Scheme Acceleration

Replace expensive radiation calculations with fast ML surrogates:

```fortran
! In module_ra_rrtmg.F
! Use ML to predict radiative heating rates
istat = torchfort_inference("radiation_model"//C_NULL_CHAR, &
                           rad_input, rad_output, 0)
```

### 4. Cumulus Parameterization

Learn convective tendencies from cloud-resolving models:

```fortran
! In module_cu_gf.F
! Predict convective heating and moistening
istat = torchfort_inference("cumulus_model"//C_NULL_CHAR, &
                           cu_input, cu_output, 0)
```

## Performance Considerations

### GPU Acceleration

1. **Asynchronous Operations**: Use CUDA streams to overlap computation
   ```fortran
   ! Create CUDA stream
   INTEGER(c_int) :: cuda_stream = 1
   istat = torchfort_inference("model"//C_NULL_CHAR, input, output, cuda_stream)
   ```

2. **Batch Processing**: Process multiple grid points simultaneously
   ```fortran
   ! Pack multiple columns into batch dimension
   REAL :: batch_input(batch_size, vertical_levels)
   istat = torchfort_inference("model"//C_NULL_CHAR, batch_input, batch_output, 0)
   ```

### Memory Management

- Allocate/deallocate arrays outside main time loop when possible
- Use WRF's existing memory management for large arrays
- Consider using persistent TorchFort models to avoid repeated loading

### MPI Considerations

For multi-process WRF runs:

```fortran
! Each MPI rank can have its own model instance
CHARACTER(LEN=256) :: model_name
WRITE(model_name, '(A,I4.4)') 'pbl_model_rank_', mpi_rank
istat = torchfort_create_model(TRIM(model_name)//C_NULL_CHAR, &
                               config_file, device_id)
```

## Example Implementation

A complete working example integrating TorchFort into WRF's MYNN PBL scheme:

### Directory Structure
```
WRF/
├── phys/
│   ├── module_torchfort_interface.F  (new)
│   ├── module_bl_mynn.F  (modified)
│   └── Makefile  (modified)
├── run/
│   ├── namelist.input  (modified)
│   ├── torchfort_pbl.yaml  (new)
│   └── pbl_model.pt  (new)
└── configure.wrf  (modified)
```

### Minimal Working Example

See the `examples/fortran/simulation` directory in TorchFort repository for a complete example that demonstrates the integration pattern applicable to WRF.

## Troubleshooting

### Common Issues

1. **Linking Errors**
   ```
   Error: Undefined reference to 'torchfort_create_model'
   ```
   **Solution**: Ensure `$(TORCHFORT_LIB)` is in `LIB_EXTERNAL` and includes `-lstdc++`

2. **Runtime Segmentation Faults**
   ```
   Segmentation fault in torchfort_train
   ```
   **Solution**: Check array dimensions match model expectations; verify CUDA device availability

3. **YAML Configuration Errors**
   ```
   Error loading config file
   ```
   **Solution**: Validate YAML syntax; ensure all required fields are present

4. **Model Not Found**
   ```
   Error: Cannot load TorchScript model
   ```
   **Solution**: Verify `.pt` file path is correct and accessible from WRF run directory

### Debug Tips

1. **Enable Verbose Logging**: Set environment variable
   ```bash
   export TORCHFORT_LOG_LEVEL=DEBUG
   ```

2. **Check Model Loading**: Test model outside WRF
   ```python
   import torch
   model = torch.jit.load('pbl_model.pt')
   print(model)
   ```

3. **Validate Input/Output Shapes**: Print array dimensions
   ```fortran
   PRINT *, 'Input shape:', SHAPE(ml_input)
   PRINT *, 'Output shape:', SHAPE(ml_output)
   ```

## Additional Resources

- **TorchFort Documentation**: https://nvidia.github.io/TorchFort/
- **WRF User's Guide**: https://www2.mmm.ucar.edu/wrf/users/
- **TorchFort Examples**: See `examples/fortran/` directory
- **WRF Physics Documentation**: https://www2.mmm.ucar.edu/wrf/users/physics/

## Support

For questions and issues:
- TorchFort GitHub Issues: https://github.com/NVIDIA/TorchFort/issues
- WRF Forum: https://forum.mmm.ucar.edu/

## Citation

If you use TorchFort with WRF in your research, please cite:
```
@software{torchfort,
  title = {TorchFort: Deep Learning Interface for HPC},
  author = {NVIDIA Corporation},
  url = {https://github.com/NVIDIA/TorchFort},
  year = {2024}
}
```
