.. _torchfort_api_f-ref:

#####################
TorchFort Fortran API
#####################

These are all the types and functions available in the TorchFort Fortran API.

********
General
********

Types
=====

.. _torchfort_datatype_t-f-ref:

torchfort_datatype
------------------
See documentation for equivalent C enumerator, :ref:`torchfort_datatype_t-ref`.

------

.. _torchfort_result_t-f-ref:

torchfort_result
----------------
See documentation for equivalent C enumerator, :ref:`torchfort_result_t-ref`.

------

.. _torchfort_tensor_list_t-f-ref:

torchfort_tensor_list
---------------------
See documentation for equivalent C typedef, :ref:`torchfort_tensor_list_t-ref`.

------

Global Context Settings
========================

These are global routines which affect the behavior of the libtorch backend. It is therefore recommended to call these functions before any other TorchFort calls are made. 

.. _torchfort_set_cudnn_benchmark-f-ref:

torchfort_set_cudnn_benchmark
-----------------------------
.. f:function :: torchfort_set_cudnn_benchmark(flag)

  Enables or disables cuDNN benchmark mode. See the `PyTorch documentation <https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark>`_ for more details.
  
  :p logical flag[in]: A flag to enable (:code:`.true.`) or disable (:code:`.false.`) cuDNN kernel benchmarking.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

Tensor List Management
======================

.. _torchfort_tensor_list_create-f-ref:

torchfort_tensor_list_create
----------------------------
.. f:function :: torchfort_tensor_list_create(tensor_list)

   Creates a TorchFort tensor list.

  :p torchfort_tensor_list tensor_list[out]: An unintialized TorchFort tensor list.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_tensor_list_destroy-f-ref:

torchfort_tensor_list_destroy
-----------------------------
.. f:function :: torchfort_tensor_list_destroy(tensor_list)

   Destroys a TorchFort tensor list.

  :p torchfort_tensor_list tensor_list[in]: A TorchFort tensor list.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_tensor_list_add_tensor-f-ref:

torchfort_tensor_list_add_tensor
--------------------------------
.. f:function :: torchfort_tensor_list_add_tensor(tensor_list, data_arr)

  Adds a tensor to a TorchFort tensor list. Tensor data is added by reference, so changes to externally provided memory will modify tensors contained in the list.

  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`, :code:`integer(int32)`, :code:`integer(int64)`

  :p torchfort_tensor_list tensor_list[in]: A TorchFort tensor list.
  :p T(*) data_arr [in]: An array containing the tensor data.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_general_f-ref:

*******************
Supervised Learning
*******************

Model Creation
===================================

.. _torchfort_create_model-f-ref:

torchfort_create_model
----------------------

.. f:function:: torchfort_create_model(name, config_fname, device)

  Creates a model from a provided configuration file.

  :p character(:) handle [in]: A name to assign to the created model instance to use as a key for other TorchFort routines.
  :p character(:) config_fname [in]: The filesystem path to the user-defined model configuration file to use.
  :p integer device [in]: Which device to place and run the model on. For ``TORCHFORT_DEVICE_CPU`` (-1), model will be placed on CPU. For values >= 0, model will be placed on GPU with index corresponding to value.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_create_distributed_model-f-ref:

torchfort_create_distributed_model
----------------------------------

.. f:function:: torchfort_create_distributed_model(name, config_fname, mpi_comm, device)

  Creates a distributed data-parallel model from a provided configuration file.

  :p character(:) handle [in]: A name to assign to the created model instance to use as a key for other TorchFort routines.
  :p character(:) config_fname [in]: The filesystem path to the user-defined configuration file to use.
  :p integer mpi_comm [in]: MPI communicator to use to initialize NCCL communication library for data-parallel communication.
  :p integer device [in]: Which device to place and run the model on. For ``TORCHFORT_DEVICE_CPU`` (-1), model will be placed on CPU. For values >= 0, model will be placed on GPU with index corresponding to value.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

Model Training/Inference
===================================

.. _torchfort_train-f-ref:

torchfort_train
---------------

.. f:function:: torchfort_train(mname, input, label, loss_val, stream)

  Runs a training iteration of a model instance using provided input and label data.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) mname [in]: The key of the model instance.
  :p T(*) input [in]: An array containing the input data. The last array dimension should be the batch dimension, the other dimensions are the feature dimensions.
  :p T(*) label [in]: An array containing the label data. The last array dimension should be the batch dimension. :code:`label` does not need to be of the same shape as :code:`input` but the batch dimension should match. Additionally, :code:`label` should be of the same rank as `input`.
  :p T loss_val [out]: A variable that will hold the loss value computed during the training iteration.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_train_multiarg-f-ref:

torchfort_train_multiarg
------------------------

.. f:function:: torchfort_train(mname, inputs, labels, loss_val, extra_loss_args, stream)

  Runs a training iteration of a model instance using provided input and label tensor lists.

  :p character(:) mname [in]: The key of the model instance.
  :p torchfort_tensor_list inputs [in]: A tensor list of input tensors.
  :p torchfort_tensor_list labels [in]: A tensor list of label tensors.
  :p real(real32) loss_val [out]: A single precision scalar that will hold the loss value computed during the training iteration.
  :p torchfort_tensor_list extra_loss_args [in,optional]: A tensor list of additional tensors to pass into loss computation. 
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_inference-f-ref:

torchfort_inference
-------------------

.. f:function:: torchfort_inference(mname, input, output, stream)

   Runs inference on a model using provided input data.
   
   For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
   
   :p character(:) mname [in]: The key of the model instance.
   :p T(*) input [in]: An array containing the input data. The last array dimension should be the batch dimension, the other dimensions are the feature dimensions.
   :p T(*) output [out]: An array which will hold the output of the model. The last array dimension should be the batch dimension. :code:`output` does not need to be of the same shape as :code:`input` but the batch dimension should match. Additionally, :code:`output` should be of the same rank as `input`. 
   :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
   :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
   
------

.. _torchfort_inference_multiarg-f-ref:

torchfort_inference_multiarg
----------------------------

.. f:function:: torchfort_inference_multiarg(mname, inputs, outputs, stream)

   Runs inference on a model using provided input data tensor list.

   :p character(:) mname [in]: The key of the model instance.
   :p torchfort_tensor_list inputs [in]: A tensor list of input tensors.
   :p torchfort_tensor_list outputs [inout]: A tensor list of output tensors.
   :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
   :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

Model Management
================

.. _torchfort_save_model-f-ref:

torchfort_save_model
--------------------

.. f:function:: torchfort_save_model(mname, fname)

  Saves a model to file.
  
  :p character(:) mname [in]: The name of model instance to save, as defined during model creation.
  :p character(:) fname [in]: The filename to save the model weights to.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_load_model-f-ref:

torchfort_load_model
--------------------

.. f:function:: torchfort_load_model(mname, fname)

  Loads a model from a file.
  
  :p character(:) mname [in]: The name of model instance to load the model weights to, as defined during model creation.
  :p character(:) fname [in]: The filename to load the model weights from.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_save_checkpoint-f-ref:

torchfort_save_checkpoint
-------------------------

.. f:function:: torchfort_save_checkpoint(mname, checkpoint_dir)

  Saves a training checkpoint to a directory. In contrast to :code:`torchfort_save_model`, this function saves additional state to restart training, like the optimizer states and learning rate schedule progress.
  
  :p character(:) mname [in]: The name of model instance to save, as defined during model creation.
  :p character(:) checkpoint_dir [in]: A writeable filesystem path to a directory to save the checkpoint data to.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_load_checkpoint-f-ref:

torchfort_load_checkpoint
-------------------------

.. f:function:: torchfort_load_checkpoint(mname, checkpoint_dir)

  Loads a training checkpoint from a directory. In contrast to :code:`torchfort_load_model`, this function loads additional state to restart training, like the optimizer states and learning rate schedule progress.
  
  :p character(:) mname [in]: The name of model instance to load checkpoint data into, as defined during model creation.
  :p character(:) checkpoint_dir [in]: A readable filesystem path to a directory to load the checkpoint data from.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

Weights and Biases Logging
==========================

.. _torchfort_wandb_log_int-f-ref:

torchfort_wandb_log_int
-----------------------

.. f:function:: torchfort_wandb_log_int(mname, metric_name, step, val)
   
   Write an integer value to a Weights and Bias log. Use the :code:`_float` and :code:`_double` variants to write :code:`real32` and :code:`real64` values respectively. 
   
   :p character(:) mname [in]: The name of model instance to associate this metric value with, as defined during model creation.
   :p character(:) metric_name [in]: Metric label.
   :p integer step [in]: Training/inference step to associate with metric value.
   :p integer val [in]: Metric value to log.
   :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_wandb_log_float-f-ref:

torchfort_wandb_log_float
-------------------------

.. f:function:: torchfort_wandb_log_float(mname, metric_name, step, val)

------

.. _torchfort_wandb_log_double-f-ref:

torchfort_wandb_log_double
--------------------------

.. f:function:: torchfort_wandb_log_double(mname, metric_name, step, val)

------

.. _torchfort_rl_f-ref:

**********************
Reinforcement Learning
**********************

Similar to other reinforcement learning frameworks such as `Spinning Up <https://spinningup.openai.com/en/latest/>`_ 
from OpenAI or `Stable Baselines <https://stable-baselines3.readthedocs.io/en/master/>`_, 
we distinguish between on-policy and off-policy algorithms since those two types require different APIs.

------

.. _torchfort_rl_off_policy_f-ref:

Off-Policy Algorithms
=====================

System Creation
-----------------------------------

Basic routines to create and register a reinforcement learning system in the internal registry. A (synchronous) data parallel distributed option is available.

.. _torchfort_rl_off_policy_create_system-f-ref:

torchfort_rl_off_policy_create_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_create_system(name, config_fname, model_device, rb_device)

  Creates an off-policy reinforcement learning training system instance from a provided configuration file.

  :p character(:) name [in]: A name to assign to the created training system instance to use as a key for other TorchFort routines.
  :p character(:) config_fname [in]: The filesystem path to the user-defined configuration file to use.
  :p integer model_device [in]: Which device to place and run the model on. For ``TORCHFORT_DEVICE_CPU`` (-1), model will be placed on CPU. For values >= 0, model will be placed on GPU with index corresponding to value.
  :p integer rb_device [in]: Which device to place and run the replay buffer on. For ``TORCHFORT_DEVICE_CPU`` (-1), replay buffer will be placed on CPU. For values >= 0, it will be placed on GPU with index corresponding to value.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_rl_off_policy_create_distributed_system-f-ref:

torchfort_rl_off_policy_create_distributed_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_create_distributed_system(name, config_fname, mpi_comm, model_device, rb_device)

  Creates a (synchronous) data-parallel off-policy reinforcement learning system instance from a provided configuration file.

  :p character(:) name [in]: A name to assign to the created training system instance to use as a key for other TorchFort routines.
  :p character(:) config_fname [in]: The filesystem path to the user-defined configuration file to use.
  :p integer mpi_comm [in]: MPI communicator to use to initialize NCCL communication library for data-parallel communication.
  :p integer model_device [in]: Which device to place and run the model on. For ``TORCHFORT_DEVICE_CPU`` (-1), model will be placed on CPU. For values >= 0, model will be placed on GPU with index corresponding to value.
  :p integer rb_device [in]: Which device to place the replay buffer on. For ``TORCHFORT_DEVICE_CPU`` (-1), replay buffer will be placed on CPU. For values >= 0, it will be placed on GPU with index corresponding to value.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

Training/Evaluation
-----------------------------------------

These routines are be used for training the reinforcement learning system or for steering the environment. 

.. _torchfort_rl_off_policy_train_step-f-ref:

torchfort_rl_off_policy_train_step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_train_step(name, p_loss_val, q_loss_val, stream)

  Runs a training iteration of an off-policy refinforcement learning instance and returns loss values for policy and value functions.
  This routine samples a batch of specified size from the replay buffer according to the buffers sampling procedure
  and performs a train step using this sample. The details of the training procedure are abstracted away from the user and depend on the 
  chosen system algorithm.
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T p_loss_val [out]: A single or double precision variable which will hold the policy loss value computed during the training iteration.
  :p T q_loss_val [out]: A single or double precision variable which will hold the critic loss value computed during the training iteration, averaged over all available critics (depends on the chosen algorithm).
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_rl_off_policy_predict_explore-f-ref:

torchfort_rl_off_policy_predict_explore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_predict_explore(name, state, act, stream)

  Suggests an action based on the current state of the system and adds noise as specified by the coprresponding reinforcement learning system. 
  Depending on the reinforcement learning algorithm used, the prediction is performed by the main network (not the target network). In contrast to :code:`torchfort_rl_off_policy_predict`, this routine adds noise and thus is called explorative. The kind of noise is specified during system creation.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T state [in]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the state space.
  :p T act [out]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the action space.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_rl_off_policy_predict-f-ref:

torchfort_rl_off_policy_predict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_predict(name, state, act, stream)

  Suggests an action based on the current state of the system. 
  Depending on the algorithm used, the prediction is performed by the target network. 
  In contrast to :code:`torchfort_rl_off_policy_predict_explore`, this routine does not add noise, which means it is exploitative.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T state [in]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the state space.
  :p T act [out]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the action space.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_rl_off_policy_evaluate-f-ref:

torchfort_rl_off_policy_evaluate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_evaluate(name, state, act, reward, stream)

  Predicts the future reward based on the current state and selected action.
  Depending on the learning algorithm, the routine queries the target critic networks for this. 
  The routine averages the predictions over all critics.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T state [in]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the state space.
  :p T act [in]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the action space.
  :p T reward [out]: One-dimensional array of size (:code:`batch_size`) which will hold the predicted reward values.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
 
------

System Management
-----------------

.. _torchfort_rl_off_policy_update_replay_buffer-f-ref:

torchfort_rl_off_policy_update_replay_buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_update_replay_buffer(name, state_old, act_old, state_new, reward, terminal, stream)
  
  Adds a new :math:`(s, a, s', r, d)` tuple to the replay buffer. Here :math:`s` (:code:`state_old`) is the state for which action :math:`a` (:code:`action_old`) was taken, leading to :math:`s'` (:code:`state_new`) and receiving reward :math:`r` (:code:`reward`). The terminal state flag :math:`d` (:code:`final_state`) specifies whether :math:`s'` is the final state in the episode. For a local multi-env environment (n_envs>=1), the last dim on the passed tensors has to be equal to n_envs, an reward and terminal both have to be 1D tensors of size n_env as well. For single env (n_env=1), the env dimension can be omitted and in that case reward has to be a scalar and terminal a boolean flag.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T state_old [in]: Multi-dimensional array of size of the state space.
  :p T act_old [in]: Multi-dimensional array of size of the action space.
  :p T state_new [in]: Multi-dimensional array of size of the state space.
  :p T reward [in]: Reward value.
  :p logical final_state [in]: Terminal flag.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------
 
.. _torchfort_rl_off_policy_is_ready-f-ref:
 
torchfort_rl_off_policy_is_ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. f:function:: torchfort_rl_off_policy_is_ready(name, ready)
 
  Queries a reinforcement learning system for rediness to start training.
  A user should call this method before starting training to make sure the reinforcement learning system is ready.
  This ensures that the replay buffer is filled sufficiently with exploration data as specified during system creation. 
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p logical ready [out]: Logical indicating if the system is ready for training.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_rl_off_policy_save_checkpoint-f-ref:
 
torchfort_rl_off_policy_save_checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. f:function:: torchfort_rl_off_policy_save_checkpoint(name, checkpoint_dir)

  Saves a reinforcement learning training checkpoint to a directory. 
  This method saves all models (policies, critics, target models if available) together with their corresponding optimizer and LR scheduler.
  states. It also saves the state of the replay buffer, to allow for smooth restarts of reinforcement learning training processes.
  This function should be used in conjunction with :code:`torchfort_rl_off_policy_load_checkpoint`.
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p character(:) checkpoint_dir [in]: A filesystem path to a directory to save the checkpoint data to.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_rl_off_policy_load_checkpoint-f-ref:
 
torchfort_rl_off_policy_load_checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. f:function:: torchfort_rl_off_policy_load_checkpoint(name, checkpoint_dir)

  Restores a reinforcement learning system from a checkpoint. 
  This method restores all models (policies, critics, target models if available) together with their corresponding optimizer and LR scheduler
  states. It also fully restores the state of the replay buffer, but not the current RNG seed.
  This function should be used in conjunction with :code:`torchfort_rl_off_policy_save_checkpoint`.
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p character(:) checkpoint_dir [in]: A filesystem path to a directory which contains the checkpoint data to load.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

Weights and Biases Logging
--------------------------

.. _torchfort_rl_off_policy_wandb_log_int-f-ref:

torchfort_rl_off_policy_wandb_log_int
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_wandb_log_int(mname, metric_name, step, val)
   
   Write an integer value to a Weights and Bias log. Use the :code:`_float` and :code:`_double` variants to write :code:`real32` and :code:`real64` values respectively. 
   
   :p character(:) mname [in]: The name of model instance to associate this metric value with, as defined during model creation.
   :p character(:) metric_name [in]: Metric label.
   :p integer step [in]: Training/inference step to associate with metric value.
   :p integer val [in]: Metric value to log.
   :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_rl_off_policy_wandb_log_float-f-ref:

torchfort_rl_off_policy_wandb_log_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_wandb_log_float(mname, metric_name, step, val)

------

.. _torchfort_rl_off_policy_wandb_log_double-f-ref:

torchfort_rl_off_policy_wandb_log_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_off_policy_wandb_log_double(mname, metric_name, step, val)

------

.. _torchfort_rl_on_policy_f-ref:

On-Policy Algorithms
=====================

System Creation
-----------------------------------

Basic routines to create and register a reinforcement learning system in the internal registry. 
A (synchronous) data parallel distributed option is available.

.. _torchfort_rl_on_policy_create_system-f-ref:

torchfort_rl_on_policy_create_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_create_system(name, config_fname, model_device, rb_device)

  Creates an on-policy reinforcement learning training system instance from a provided configuration file.

  :p character(:) name [in]: A name to assign to the created training system instance to use as a key for other TorchFort routines.
  :p character(:) config_fname [in]: The filesystem path to the user-defined configuration file to use.
  :p integer model_device [in]: Which device to place and run the model on. For ``TORCHFORT_DEVICE_CPU`` (-1), model will be placed on CPU. For values >= 0, model will be placed on GPU with index corresponding to value.
  :p integer rb_device [in]: Which device to place the rollout buffer on. For ``TORCHFORT_DEVICE_CPU`` (-1), rollout buffer will be placed on CPU. For values >= 0, it will be placed on GPU with index corresponding to value.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_rl_on_policy_create_distributed_system-f-ref:

torchfort_rl_on_policy_create_distributed_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_create_distributed_system(name, config_fname, mpi_comm, model_device, rb_device)

  Creates a (synchronous) data-parallel on-policy reinforcement learning system instance from a provided configuration file.

  :p character(:) name [in]: A name to assign to the created training system instance to use as a key for other TorchFort routines.
  :p character(:) config_fname [in]: The filesystem path to the user-defined configuration file to use.
  :p integer mpi_comm [in]: MPI communicator to use to initialize NCCL communication library for data-parallel communication.
  :p integer model_device [in]: Which device to place and run the model on. For ``TORCHFORT_DEVICE_CPU`` (-1), model will be placed on CPU. For values >= 0, model will be placed on GPU with index corresponding to value.
  :p integer rb_device [in]: Which device to place the rollout buffer on. For ``TORCHFORT_DEVICE_CPU`` (-1), rollout buffer will be placed on CPU. For values >= 0, it will be placed on GPU with index corresponding to value.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

Training/Evaluation
-----------------------------------------

These routines are be used for training the reinforcement learning system or for steering the environment. 

.. _torchfort_rl_on_policy_train_step-f-ref:

torchfort_rl_on_policy_train_step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_train_step(name, p_loss_val, q_loss_val, stream)

  Runs a training iteration of an on-policy refinforcement learning instance and returns loss values for policy and value functions.
  This routine samples a batch of specified size from the rollout buffer according to the buffers sampling procedure
  and performs a train step using this sample. The details of the training procedure are abstracted away from the user and depend on the 
  chosen system algorithm. Note that the rollout buffer needs to be finalized or otherwise the train step will be skipped.
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T p_loss_val [out]: A single or double precision variable which will hold the policy loss value computed during the training iteration.
  :p T q_loss_val [out]: A single or double precision variable which will hold the critic loss value computed during the training iteration, averaged over all available critics (depends on the chosen algorithm).
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_rl_on_policy_predict_explore-f-ref:

torchfort_rl_on_policy_predict_explore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_predict_explore(name, state, act, stream)

  Suggests an action based on the current state of the system and adds noise as specified by the coprresponding reinforcement learning system. 
  Depending on the reinforcement learning algorithm used, the prediction is performed by the main network (not the target network). In contrast to :code:`torchfort_rl_off_policy_predict`, this routine adds noise and thus is called explorative. The kind of noise is specified during system creation.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T state [in]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the state space.
  :p T act [out]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the action space.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_rl_on_policy_predict-f-ref:

torchfort_rl_on_policy_predict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_predict(name, state, act, stream)

  Suggests an action based on the current state of the system. 
  Depending on the algorithm used, the prediction is performed by the target network. 
  In contrast to :code:`torchfort_rl_on_policy_predict_explore`, this routine does not add noise, which means it is exploitative.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T state [in]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the state space.
  :p T act [out]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the action space.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_rl_on_policy_evaluate-f-ref:

torchfort_rl_on_policy_evaluate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_evaluate(name, state, act, reward, stream)

  Predicts the future reward based on the current state and selected action.
  Depending on the learning algorithm, the routine queries the target critic networks for this. 
  The routine averages the predictions over all critics.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T state [in]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the state space.
  :p T act [in]: Multi-dimensional array of size (..., :code:`batch_size`), depending on the dimensionality of the action space.
  :p T reward [out]: One-dimensional array of size (:code:`batch_size`) which will hold the predicted reward values.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
 
------

System Management
-----------------

.. _torchfort_rl_on_policy_update_rollout_buffer-f-ref:

torchfort_rl_on_policy_update_rollout_buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_update_rollout_buffer(name, state_old, act_old, state_new, reward, terminal, stream)
  
  Adds a new :math:`(s, a, r, d)` tuple to the rollout buffer. Here :math:`s` (:code:`state`) is the state for which action :math:`a` (:code:`action`) was taken, leading to reward :math:`r` (:code:`reward`). The terminal state flag :math:`d` (:code:`terminal`) specifies whether the state is the final state in the episode.
  Note that value estimates :math:`q` as well was log-probabilities are also stored but the user does not need to pass those manually, , those values are computed internally from the current policy and stored with the other values. For a local multi-env environment (n_envs>=1), the last dim on the passed tensors has to be equal to n_envs, an reward and terminal both have to be 1D tensors of size n_env as well. For single	env (n_env=1), the env dimension can be	omitted and in that case reward has to be a scalar and terminal a boolean flag.
  
  For this operation, :code:`T` can be one of :code:`real(real32)`, :code:`real(real64)`
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p T state [in]: Multi-dimensional array of size of the state space.
  :p T act [in]: Multi-dimensional array of size of the action space.
  :p T reward [in]: Reward value.
  :p logical terminal [in]: Terminal flag.
  :p integer(int64) stream[in,optional]: CUDA stream to enqueue the operation. This argument is ignored if the model is on the CPU.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_rl_on_policy_reset_rollout_buffer-f-ref:
 
torchfort_rl_on_policy_reset_rollout_buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. f:function:: torchfort_rl_on_policy_reset_rollout_buffer(name)
 
  This function call clears the rollout buffer and resets all variables.
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------
 
.. _torchfort_rl_on_policy_is_ready-f-ref:
 
torchfort_rl_on_policy_is_ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. f:function:: torchfort_rl_on_policy_is_ready(name, ready)
 
  Queries a reinforcement learning system for rediness to start training.
  A user should call this method before starting training to make sure the reinforcement learning system is ready.
  This ensures that the rollout buffer is filled sufficiently with exploration data as specified during system creation. 
  It also checks if the rollout buffer was properly finalized, e.g. all advantages were computed.
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p logical ready [out]: Logical indicating if the system is ready for training.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_rl_on_policy_save_checkpoint-f-ref:
 
torchfort_rl_on_policy_save_checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. f:function:: torchfort_rl_on_policy_save_checkpoint(name, checkpoint_dir)

  Saves a reinforcement learning training checkpoint to a directory. 
  This method saves all models (policies, critics, target models if available) together with their corresponding optimizer and LR scheduler.
  states. It also saves the state of the rollout buffer, to allow for smooth restarts of reinforcement learning training processes.
  This function should be used in conjunction with :code:`torchfort_rl_on_policy_load_checkpoint`.
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p character(:) checkpoint_dir [in]: A filesystem path to a directory to save the checkpoint data to.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.
  
------

.. _torchfort_rl_on_policy_load_checkpoint-f-ref:
 
torchfort_rl_on_policy_load_checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. f:function:: torchfort_rl_on_policy_load_checkpoint(name, checkpoint_dir)

  Restores a reinforcement learning system from a checkpoint. 
  This method restores all models (policies, critics, target models if available) together with their corresponding optimizer and LR scheduler
  states. It also fully restores the state of the rollout buffer, but not the current RNG seed.
  This function should be used in conjunction with :code:`torchfort_rl_on_policy_save_checkpoint`.
  
  :p character(:) name [in]: The name of system instance to use, as defined during system creation.
  :p character(:) checkpoint_dir [in]: A filesystem path to a directory which contains the checkpoint data to load.
  :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

Weights and Biases Logging
--------------------------

.. _torchfort_rl_on_policy_wandb_log_int-f-ref:

torchfort_rl_on_policy_wandb_log_int
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_wandb_log_int(mname, metric_name, step, val)
   
   Write an integer value to a Weights and Bias log. Use the :code:`_float` and :code:`_double` variants to write :code:`real32` and :code:`real64` values respectively. 
   
   :p character(:) mname [in]: The name of model instance to associate this metric value with, as defined during model creation.
   :p character(:) metric_name [in]: Metric label.
   :p integer step [in]: Training/inference step to associate with metric value.
   :p integer val [in]: Metric value to log.
   :r torchfort_result res: :code:`TORCHFORT_RESULT_SUCCESS` on success or error code on failure.

------

.. _torchfort_rl_on_policy_wandb_log_float-f-ref:

torchfort_rl_on_policy_wandb_log_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_wandb_log_float(mname, metric_name, step, val)

------

.. _torchfort_rl_on_policy_wandb_log_double-f-ref:

torchfort_rl_on_policy_wandb_log_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: torchfort_rl_on_policy_wandb_log_double(mname, metric_name, step, val)


