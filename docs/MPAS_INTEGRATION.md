# TorchFort Integration Guide for MPAS (Model for Prediction Across Scales)

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

This guide demonstrates how to integrate TorchFort into the Model for Prediction Across Scales (MPAS) to enable deep learning capabilities within Earth system simulations. MPAS is a collaborative project for developing atmosphere, ocean, and other earth-system simulation components using unstructured Voronoi meshes. By integrating TorchFort, you can:

- Train ML models using MPAS simulation data in real-time
- Develop data-driven parameterizations for unresolved processes
- Create ML-based closures for sub-grid scale physics
- Implement mesh-aware neural network models for irregular grids
- Accelerate expensive physics computations with ML surrogates

## Prerequisites

### Software Requirements

1. **MPAS Model** (v7.0 or later recommended)
   - Source code: https://github.com/MPAS-Dev/MPAS-Model
   - Can integrate with MPAS-Atmosphere, MPAS-Ocean, or MPAS-Seaice cores

2. **TorchFort Library**
   - Built and installed following the [TorchFort installation guide](installation.rst)
   - GPU support recommended for performance

3. **Compilers and Libraries**
   - Fortran compiler (gfortran, ifort, or nvfortran)
   - C/C++ compiler compatible with TorchFort
   - MPI library (OpenMPI, MPICH, or Intel MPI)
   - NetCDF library (for MPAS I/O)
   - PIO library (Parallel I/O)

4. **CUDA Toolkit** (optional, for GPU acceleration)
   - CUDA 11.0 or later
   - cuDNN library

### Knowledge Requirements

- Familiarity with MPAS framework structure and mesh topology
- Understanding of Fortran 90/95/2003
- Knowledge of unstructured mesh numerical methods
- Basic deep learning concepts and PyTorch
- Experience with YAML configuration files

## Integration Overview

### MPAS Architecture and Integration Points

MPAS uses a unique architecture with a shared framework and independent model cores:

```
┌────────────────────────────────────────────────┐
│          MPAS Framework (Shared)               │
│  ┌──────────────────────────────────────────┐ │
│  │  Framework Infrastructure                │ │
│  │  - Memory Management                     │ │
│  │  - I/O (PIO)                            │ │
│  │  - Communication                         │ │
│  │  - Timekeeping                          │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  TorchFort Integration Layer (NEW)       │ │
│  │  - Model Management                      │ │
│  │  - Data Marshalling                      │ │
│  │  - Mesh-Aware Operations                 │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│ MPAS-Atmosphere  │   │   MPAS-Ocean     │
│                  │   │                  │
│ ┌──────────────┐ │   │ ┌──────────────┐ │
│ │  Physics     │ │   │ │ Mixing       │ │
│ │  Schemes     │ │   │ │ Schemes      │ │
│ │              │ │   │ │              │ │
│ │ ┌──────────┐ │ │   │ │ ┌──────────┐ │ │
│ │ │TorchFort │ │ │   │ │ │TorchFort │ │ │
│ │ │  Calls   │ │ │   │ │ │  Calls   │ │ │
│ │ └──────────┘ │ │   │ │ └──────────┘ │ │
│ └──────────────┘ │   │ └──────────────┘ │
└──────────────────┘   └──────────────────┘
```

### Key Integration Points

1. **Physics Parameterizations**: Convection, radiation, PBL schemes (MPAS-Atmosphere)
2. **Mixing Schemes**: Vertical mixing, eddy parameterizations (MPAS-Ocean)
3. **Dynamics Core**: Sub-grid scale corrections
4. **Data Assimilation**: ML-enhanced analysis increments

## Step-by-Step Integration

### Step 1: Modify MPAS Build System

#### 1.1 Update Makefile

Edit the main `Makefile` in your MPAS core directory (e.g., `MPAS-Model/src/core_atmosphere/`):

```makefile
# Add TorchFort paths
TORCHFORT_ROOT = /path/to/torchfort/installation
TORCHFORT_INC = -I$(TORCHFORT_ROOT)/include
TORCHFORT_LIB = -L$(TORCHFORT_ROOT)/lib -ltorchfort

# Add to FCINCLUDES
FCINCLUDES = -I../framework -I../operators -I../external/esmf_time_f90 \
             $(TORCHFORT_INC)

# Add to LIBS
LIBS = $(PIO_LIB) $(NETCDF_LIB) $(PNETCDF_LIB) $(TORCHFORT_LIB) -lstdc++

# Add C++ linker flag
LDFLAGS += -lstdc++
```

#### 1.2 Modify Build Script

Update the build script to ensure proper linking:

```bash
# In Makefile or build script
ifeq "$(TORCHFORT)" "true"
    CPPFLAGS += -DUSE_TORCHFORT
    FFLAGS += -DUSE_TORCHFORT
endif
```

### Step 2: Create TorchFort Interface Module

Create a new module for TorchFort integration in the MPAS framework:

File: `src/framework/mpas_torchfort_interface.F`

```fortran
MODULE mpas_torchfort_interface
   USE iso_c_binding
   USE mpas_kind_types
   USE mpas_pool_routines
   USE mpas_dmpar

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: mpas_torchfort_init, &
             mpas_torchfort_finalize, &
             mpas_torchfort_create_model, &
             mpas_torchfort_train_on_mesh, &
             mpas_torchfort_inference_on_mesh

   ! TorchFort C interface
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

   ! Module state
   LOGICAL :: torchfort_enabled = .FALSE.
   INTEGER :: torchfort_device = 0
   CHARACTER(LEN=StrKIND) :: torchfort_config_path

CONTAINS

   !-----------------------------------------------------------------------
   SUBROUTINE mpas_torchfort_init(domain, ierr)
      TYPE (domain_type), INTENT(IN) :: domain
      INTEGER, INTENT(OUT) :: ierr

      TYPE (block_type), POINTER :: block
      CHARACTER(LEN=StrKIND) :: config_torchfort_enabled
      CHARACTER(LEN=StrKIND) :: config_torchfort_device
      CHARACTER(LEN=StrKIND) :: config_torchfort_config

      ierr = 0

      ! Read configuration from namelist
      CALL mpas_pool_get_config(domain % configs, 'config_torchfort_enabled', &
                                config_torchfort_enabled)
      CALL mpas_pool_get_config(domain % configs, 'config_torchfort_device', &
                                config_torchfort_device)
      CALL mpas_pool_get_config(domain % configs, 'config_torchfort_config', &
                                config_torchfort_config)

      IF (TRIM(config_torchfort_enabled) == 'true') THEN
         torchfort_enabled = .TRUE.
         READ(config_torchfort_device, *) torchfort_device
         torchfort_config_path = TRIM(config_torchfort_config)

         CALL mpas_log_write('TorchFort integration enabled')
         CALL mpas_log_write('  Device: $i', intArgs=[torchfort_device])
         CALL mpas_log_write('  Config: ' // TRIM(torchfort_config_path))
      ELSE
         torchfort_enabled = .FALSE.
         CALL mpas_log_write('TorchFort integration disabled')
      ENDIF

   END SUBROUTINE mpas_torchfort_init

   !-----------------------------------------------------------------------
   SUBROUTINE mpas_torchfort_finalize()
      torchfort_enabled = .FALSE.
   END SUBROUTINE mpas_torchfort_finalize

   !-----------------------------------------------------------------------
   SUBROUTINE mpas_torchfort_create_model(model_name, config_file, device, ierr)
      CHARACTER(LEN=*), INTENT(IN) :: model_name
      CHARACTER(LEN=*), INTENT(IN) :: config_file
      INTEGER, INTENT(IN) :: device
      INTEGER, INTENT(OUT) :: ierr

      INTEGER(c_int) :: status

      status = torchfort_create_model(TRIM(model_name)//C_NULL_CHAR, &
                                      TRIM(config_file)//C_NULL_CHAR, &
                                      INT(device, c_int))
      ierr = INT(status)

      IF (ierr == 0) THEN
         CALL mpas_log_write('TorchFort model created: ' // TRIM(model_name))
      ELSE
         CALL mpas_log_write('ERROR: Failed to create TorchFort model: ' // &
                            TRIM(model_name), messageType=MPAS_LOG_ERR)
      ENDIF

   END SUBROUTINE mpas_torchfort_create_model

   !-----------------------------------------------------------------------
   SUBROUTINE mpas_torchfort_train_on_mesh(model_name, input_fields, label_fields, &
                                          nCells, nVertLevels, loss_val, ierr)
      CHARACTER(LEN=*), INTENT(IN) :: model_name
      REAL(KIND=RKIND), DIMENSION(:,:), INTENT(IN) :: input_fields
      REAL(KIND=RKIND), DIMENSION(:,:), INTENT(IN) :: label_fields
      INTEGER, INTENT(IN) :: nCells, nVertLevels
      REAL(KIND=RKIND), INTENT(OUT) :: loss_val
      INTEGER, INTENT(OUT) :: ierr

      REAL(c_float), ALLOCATABLE :: input_flat(:), label_flat(:)
      REAL(c_float) :: loss_c
      INTEGER :: i, k, idx
      INTEGER(c_int) :: status

      ! Flatten mesh data to 1D array
      ALLOCATE(input_flat(nCells * nVertLevels))
      ALLOCATE(label_flat(nCells * nVertLevels))

      idx = 1
      DO k = 1, nVertLevels
         DO i = 1, nCells
            input_flat(idx) = REAL(input_fields(k, i), c_float)
            label_flat(idx) = REAL(label_fields(k, i), c_float)
            idx = idx + 1
         ENDDO
      ENDDO

      ! Call TorchFort training
      status = torchfort_train(TRIM(model_name)//C_NULL_CHAR, &
                              input_flat, label_flat, loss_c, 0)

      loss_val = REAL(loss_c, RKIND)
      ierr = INT(status)

      DEALLOCATE(input_flat, label_flat)

   END SUBROUTINE mpas_torchfort_train_on_mesh

   !-----------------------------------------------------------------------
   SUBROUTINE mpas_torchfort_inference_on_mesh(model_name, input_fields, output_fields, &
                                              nCells, nVertLevels, ierr)
      CHARACTER(LEN=*), INTENT(IN) :: model_name
      REAL(KIND=RKIND), DIMENSION(:,:), INTENT(IN) :: input_fields
      REAL(KIND=RKIND), DIMENSION(:,:), INTENT(OUT) :: output_fields
      INTEGER, INTENT(IN) :: nCells, nVertLevels
      INTEGER, INTENT(OUT) :: ierr

      REAL(c_float), ALLOCATABLE :: input_flat(:), output_flat(:)
      INTEGER :: i, k, idx
      INTEGER(c_int) :: status

      ! Flatten mesh data to 1D array
      ALLOCATE(input_flat(nCells * nVertLevels))
      ALLOCATE(output_flat(nCells * nVertLevels))

      idx = 1
      DO k = 1, nVertLevels
         DO i = 1, nCells
            input_flat(idx) = REAL(input_fields(k, i), c_float)
            idx = idx + 1
         ENDDO
      ENDDO

      ! Call TorchFort inference
      status = torchfort_inference(TRIM(model_name)//C_NULL_CHAR, &
                                   input_flat, output_flat, 0)

      ! Unflatten output
      idx = 1
      DO k = 1, nVertLevels
         DO i = 1, nCells
            output_fields(k, i) = REAL(output_flat(idx), RKIND)
            idx = idx + 1
         ENDDO
      ENDDO

      ierr = INT(status)

      DEALLOCATE(input_flat, output_flat)

   END SUBROUTINE mpas_torchfort_inference_on_mesh

END MODULE mpas_torchfort_interface
```

### Step 3: Integrate into MPAS Physics

Example integration into MPAS-Atmosphere convection scheme:

File: `src/core_atmosphere/physics/mpas_atmphys_driver_convection.F`

```fortran
MODULE mpas_atmphys_driver_convection
   USE mpas_torchfort_interface

   CONTAINS

   SUBROUTINE driver_convection(configs, mesh, state, diag, tend, &
                                 its, ite, kts, kte)

      ! Existing MPAS variables
      TYPE (mpas_pool_type), INTENT(IN) :: configs, mesh, state
      TYPE (mpas_pool_type), INTENT(INOUT) :: diag, tend

      ! Local variables for ML
      REAL(KIND=RKIND), DIMENSION(:,:), ALLOCATABLE :: ml_input, ml_output
      REAL(KIND=RKIND) :: loss_value
      INTEGER :: ierr, iCell, k
      LOGICAL :: use_ml_convection
      CHARACTER(LEN=StrKIND) :: config_conv_scheme

      ! Arrays from MPAS mesh
      INTEGER, DIMENSION(:), POINTER :: nCells
      REAL(KIND=RKIND), DIMENSION(:,:), POINTER :: theta, qv, w
      REAL(KIND=RKIND), DIMENSION(:,:), POINTER :: rthcuten, rqvcuten

      ! Get configuration
      CALL mpas_pool_get_config(configs, 'config_convection_scheme', config_conv_scheme)
      use_ml_convection = (TRIM(config_conv_scheme) == 'ml_convection')

      IF (use_ml_convection .AND. torchfort_enabled) THEN

         ! Get pointers to mesh and state arrays
         CALL mpas_pool_get_dimension(mesh, 'nCells', nCells)
         CALL mpas_pool_get_array(state, 'theta', theta)
         CALL mpas_pool_get_array(state, 'qv', qv)
         CALL mpas_pool_get_array(state, 'w', w)
         CALL mpas_pool_get_array(tend, 'rthcuten', rthcuten)
         CALL mpas_pool_get_array(tend, 'rqvcuten', rqvcuten)

         ! Prepare input: stack vertical profiles of theta, qv, w
         ALLOCATE(ml_input(3, nCells))
         ALLOCATE(ml_output(2, nCells))

         ! Example: use column-integrated values or specific levels
         DO iCell = 1, nCells
            ml_input(1, iCell) = SUM(theta(:, iCell)) / SIZE(theta, 1)
            ml_input(2, iCell) = SUM(qv(:, iCell)) / SIZE(qv, 1)
            ml_input(3, iCell) = SUM(w(:, iCell)) / SIZE(w, 1)
         ENDDO

         ! Run ML inference
         CALL mpas_torchfort_inference_on_mesh('convection_model', &
                                               ml_input, ml_output, &
                                               nCells, 2, ierr)

         IF (ierr == 0) THEN
            ! Apply ML-predicted convective tendencies
            DO iCell = 1, nCells
               rthcuten(:, iCell) = rthcuten(:, iCell) + ml_output(1, iCell)
               rqvcuten(:, iCell) = rqvcuten(:, iCell) + ml_output(2, iCell)
            ENDDO

            CALL mpas_log_write('Applied ML convection scheme')
         ELSE
            CALL mpas_log_write('ERROR: ML convection inference failed', &
                               messageType=MPAS_LOG_ERR)
         ENDIF

         DEALLOCATE(ml_input, ml_output)

      ELSE
         ! Use traditional convection scheme
         CALL convection_scheme_traditional( ... )
      ENDIF

   END SUBROUTINE driver_convection

END MODULE mpas_atmphys_driver_convection
```

### Step 4: Configure MPAS Namelist

Add TorchFort options to `namelist.atmosphere` or `namelist.ocean`:

```fortran
&torchfort
    config_torchfort_enabled = true
    config_torchfort_device = 0
    config_torchfort_config = 'torchfort_convection.yaml'
/

&physics
    config_convection_scheme = 'ml_convection'
/
```

### Step 5: Add Registry Entries

Add new configuration options to MPAS Registry file (e.g., `Registry.xml`):

```xml
<nml_record name="torchfort">
    <nml_option name="config_torchfort_enabled" type="logical"
                default_value=".false."
                description="Enable TorchFort deep learning integration"/>

    <nml_option name="config_torchfort_device" type="integer"
                default_value="0"
                description="GPU device ID for TorchFort (-1 for CPU)"/>

    <nml_option name="config_torchfort_config" type="character"
                default_value="torchfort_config.yaml"
                description="Path to TorchFort YAML configuration file"/>
</nml_record>
```

### Step 6: Create TorchFort Configuration

Create `torchfort_convection.yaml`:

```yaml
model:
  type: torchscript
  torchscript_file: convection_model.pt

optimizer:
  type: Adam
  parameters:
    lr: 0.0001
    betas: [0.9, 0.999]
    weight_decay: 0.00001

loss:
  type: mse

lr_scheduler:
  type: CosineAnnealingLR
  parameters:
    T_max: 1000
    eta_min: 0.00001

training:
  batch_size: 32
  enable_amp: true
```

### Step 7: Create PyTorch Model for MPAS Mesh

```python
# generate_convection_model.py
import torch
import torch.nn as nn

class ConvectionModel(nn.Module):
    """
    Model for predicting convective tendencies on MPAS mesh.
    Input: Column-mean theta, qv, w (3 features)
    Output: Convective heating and moistening tendencies (2 outputs)
    """
    def __init__(self):
        super(ConvectionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# Create and export
model = ConvectionModel()
scripted_model = torch.jit.script(model)
scripted_model.save('convection_model.pt')
print("Model saved to convection_model.pt")
```

### Step 8: Build and Run MPAS

```bash
# Build MPAS with TorchFort
cd MPAS-Model
make clean CORE=atmosphere
export TORCHFORT=true
make gfortran CORE=atmosphere AUTOCLEAN=true

# Run MPAS
cd testcases/your_test_case
# Ensure convection_model.pt and torchfort_convection.yaml are present
mpirun -np 8 ../../atmosphere_model

# Monitor output
tail -f log.atmosphere.0000.out
```

## Use Cases

### 1. Ocean Mixing Parameterization (MPAS-Ocean)

Learn vertical mixing coefficients from high-resolution ocean simulations:

```fortran
! In mpas_ocn_vmix_coefs_rich.F
CALL mpas_torchfort_inference_on_mesh('ocean_mixing_model', &
                                      input_profiles, mixing_coeffs, &
                                      nCells, nVertLevels, ierr)
```

**Training Data**: High-resolution LES ocean simulations
**Inputs**: Vertical profiles of temperature, salinity, velocity shear
**Outputs**: Turbulent diffusivity and viscosity profiles

### 2. Atmospheric Radiation Surrogate (MPAS-Atmosphere)

Replace expensive radiation calculations with ML:

```fortran
! In mpas_atmphys_driver_radiation_sw.F
CALL mpas_torchfort_inference_on_mesh('radiation_sw_model', &
                                      atmos_state, heating_rates, &
                                      nCells, nVertLevels, ierr)
```

**Training Data**: Offline RRTMG calculations
**Inputs**: Temperature, water vapor, cloud properties, solar zenith angle
**Outputs**: Shortwave heating rates

### 3. Sea Ice Dynamics (MPAS-Seaice)

Learn sub-grid scale sea ice rheology:

```fortran
! In mpas_seaice_dynamics.F
CALL mpas_torchfort_inference_on_mesh('seaice_rheology_model', &
                                      ice_state, stress_tensor, &
                                      nCells, 1, ierr)
```

### 4. Mesh-Aware Graph Neural Networks

For truly mesh-aware models, use Graph Neural Networks (GNNs) that respect MPAS's unstructured mesh topology:

```python
# generate_gnn_model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class MPASGraphModel(nn.Module):
    """
    Graph Neural Network for MPAS unstructured mesh.
    Requires pre-processing to build edge connectivity from MPAS mesh.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MPASGraphModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x

# Export to TorchScript (requires example inputs)
model = MPASGraphModel(in_channels=10, hidden_channels=64, out_channels=5)
# Note: GNN models require edge_index - implement custom export
```

## Performance Considerations

### Mesh-Specific Optimizations

1. **Spatial Locality**: Group neighboring cells for batch processing
   ```fortran
   ! Process cells in spatial batches
   DO iBatch = 1, nBatches
      start_cell = (iBatch - 1) * batch_size + 1
      end_cell = MIN(iBatch * batch_size, nCells)
      CALL mpas_torchfort_inference_on_mesh( ... )
   ENDDO
   ```

2. **Vertical Column Processing**: Process all vertical levels simultaneously
   ```fortran
   ! Full 3D column as input
   ALLOCATE(column_input(nVertLevels, nFeatures))
   ALLOCATE(column_output(nVertLevels, nOutputs))
   ```

### GPU Acceleration for MPAS

```fortran
! Use GPU-resident data with CUDA-aware MPI
#ifdef USE_CUDA
   !$acc data copyin(input_fields) copyout(output_fields)
   !$acc host_data use_device(input_fields, output_fields)
   CALL mpas_torchfort_inference_on_mesh('model', input_fields, &
                                         output_fields, nCells, &
                                         nVertLevels, ierr)
   !$acc end host_data
   !$acc end data
#endif
```

### Parallel Decomposition

MPAS domain decomposition considerations:

```fortran
! Each MPI rank handles its own subdomain
! TorchFort operations are local to each rank
DO WHILE (associated(block))
   CALL mpas_torchfort_inference_on_mesh('model', &
                                         block % input, &
                                         block % output, &
                                         block % nCells, &
                                         nVertLevels, ierr)
   block => block % next
ENDDO

! Halo exchanges may be needed for spatially-aware models
CALL mpas_dmpar_exch_halo_field(output_field)
```

### Memory Considerations

MPAS simulations can be memory-intensive:

```fortran
! Reuse allocated arrays across timesteps
IF (.NOT. ALLOCATED(ml_workspace)) THEN
   ALLOCATE(ml_workspace(nCells, nVertLevels))
ENDIF

! Release during finalization only
IF (finalize_step) THEN
   DEALLOCATE(ml_workspace)
ENDIF
```

## Example Implementation

### Complete Working Example: ML-Enhanced PBL in MPAS-Atmosphere

Directory structure:
```
MPAS-Model/
├── src/
│   ├── framework/
│   │   ├── mpas_torchfort_interface.F (new)
│   │   └── Makefile (modified)
│   └── core_atmosphere/
│       ├── physics/
│       │   └── mpas_atmphys_driver_pbl.F (modified)
│       └── Makefile (modified)
├── testcases/
│   └── your_case/
│       ├── namelist.atmosphere (modified)
│       ├── torchfort_pbl.yaml (new)
│       └── pbl_model.pt (new)
└── Makefile (modified)
```

### Minimal Example Code

See TorchFort `examples/fortran/simulation` for patterns applicable to MPAS integration.

## Troubleshooting

### Common Issues

1. **NetCDF/PIO Conflicts**
   ```
   Error: Symbol conflicts between TorchFort and NetCDF
   ```
   **Solution**: Ensure consistent library versions; link TorchFort after NetCDF

2. **MPI Initialization Order**
   ```
   Error: MPI already initialized
   ```
   **Solution**: Call `mpas_torchfort_init()` after `MPI_Init()` in MPAS

3. **Memory Errors on Large Meshes**
   ```
   Error: Out of memory allocating TorchFort tensors
   ```
   **Solution**: Process mesh in batches; use GPU memory efficiently

4. **Mesh Connectivity Issues**
   ```
   Error: Invalid cell indices
   ```
   **Solution**: Ensure proper indexing between MPAS (1-based) and ML models (0-based)

### Debugging Tips

1. **Enable MPAS Logging**
   ```fortran
   CALL mpas_log_write('TorchFort input min/max: $r $r', &
                       realArgs=[MINVAL(input), MAXVAL(input)])
   ```

2. **Validate Mesh Data**
   ```fortran
   ! Check for NaN or Inf values
   IF (ANY(IEEE_IS_NAN(input_fields))) THEN
      CALL mpas_log_write('ERROR: NaN in input fields', &
                         messageType=MPAS_LOG_ERR)
   ENDIF
   ```

3. **Test Model Outside MPAS**
   ```python
   import torch
   model = torch.jit.load('pbl_model.pt')
   test_input = torch.randn(10, 50)  # 10 cells, 50 levels
   output = model(test_input)
   print(f"Output shape: {output.shape}")
   ```

### Performance Profiling

```bash
# Profile MPAS with TorchFort
export TORCHFORT_LOG_LEVEL=INFO
export CUDA_LAUNCH_BLOCKING=1
mpirun -np 4 nsys profile -o mpas_profile ./atmosphere_model

# Analyze with NVIDIA Nsight Systems
nsys-ui mpas_profile.qdrep
```

## Advanced Topics

### 1. Conservative Remapping

Ensure ML predictions conserve physical quantities:

```fortran
! Apply mass-weighted correction
total_before = SUM(field(:) * cell_area(:))
CALL mpas_torchfort_inference_on_mesh( ... )
total_after = SUM(field(:) * cell_area(:))
correction = total_before / total_after
field(:) = field(:) * correction
```

### 2. Multi-Resolution Meshes

Handle MPAS variable-resolution meshes:

```fortran
! Include cell spacing as input feature
DO iCell = 1, nCells
   ml_input(1, iCell) = dcEdge(iCell)  ! Local mesh spacing
   ml_input(2:, iCell) = physics_state(:, iCell)
ENDDO
```

### 3. Online Learning During Simulation

Implement continuous learning:

```fortran
IF (MOD(itimestep, training_interval) == 0) THEN
   CALL mpas_torchfort_train_on_mesh('model', input, label, loss, ierr)
   IF (MOD(itimestep, checkpoint_interval) == 0) THEN
      CALL torchfort_save_checkpoint('model'//C_NULL_CHAR, 'checkpoint_dir'//C_NULL_CHAR)
   ENDIF
ENDIF
```

## Additional Resources

- **TorchFort Documentation**: https://nvidia.github.io/TorchFort/
- **MPAS Documentation**: https://mpas-dev.github.io/
- **MPAS-Atmosphere User's Guide**: https://www2.mmm.ucar.edu/projects/mpas/
- **MPAS-Ocean User's Guide**: https://mpas-dev.github.io/ocean/ocean.html
- **TorchFort Examples**: See `examples/` directory in TorchFort repository

## Support and Community

- **TorchFort Issues**: https://github.com/NVIDIA/TorchFort/issues
- **MPAS Forum**: https://forum.mmm.ucar.edu/forums/mpas.110/
- **MPAS GitHub**: https://github.com/MPAS-Dev/MPAS-Model/issues

## Citation

If you use TorchFort with MPAS in your research, please cite both:

```bibtex
@software{torchfort,
  title = {TorchFort: Deep Learning Interface for HPC},
  author = {NVIDIA Corporation},
  url = {https://github.com/NVIDIA/TorchFort},
  year = {2024}
}

@article{mpas,
  title = {A multi-scale nonhydrostatic atmospheric model using
           centroidal Voronoi tesselations and C-grid staggering},
  author = {Skamarock, W. C. and others},
  journal = {Monthly Weather Review},
  volume = {140},
  pages = {3090--3105},
  year = {2012}
}
```

## Contributing

We welcome contributions and use cases! If you've successfully integrated TorchFort with MPAS:
- Share your experience in the discussions
- Submit pull requests with improvements
- Report issues and suggest features

---

**Last Updated**: November 2025
**TorchFort Version**: Compatible with v0.3+
**MPAS Version**: Tested with v7.0+
