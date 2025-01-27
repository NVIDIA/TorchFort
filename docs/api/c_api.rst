.. _torchfort_api_c-ref:

###############
TorchFort C API
###############

These are all the types and functions available in the TorchFort C API.

*******
General
*******

Types
=====

.. _torchfort_datatype_t-ref:

torchfort_datatype_t
--------------------
.. doxygenenum :: torchfort_datatype_t

------

.. _torchfort_result_t-ref:

torchfort_result_t
------------------
.. doxygenenum :: torchfort_result_t

------

.. _torchfort_tensor_list_t-ref:

torchfort_tensor_list_t
-----------------------
.. doxygentypedef :: torchfort_tensor_list_t

------

Global Context Settings
=======================

These are global routines which affect the behavior of the libtorch backend. It is therefore recommended to call these functions before any other TorchFort calls are made. 

.. _torchfort_set_cudnn_benchmark-ref:

torchfort_set_cudnn_benchmark
-----------------------------
.. doxygenfunction:: torchfort_set_cudnn_benchmark


Tensor List Management
======================

.. _torchfort_tensor_list_create-ref:

torchfort_tensor_list_create
----------------------------
.. doxygenfunction:: torchfort_tensor_list_create

------

.. _torchfort_tensor_list_destroy-ref:

torchfort_tensor_list_destroy
-----------------------------
.. doxygenfunction:: torchfort_tensor_list_destroy

------

.. _torchfort_tensor_list_add_tensor-ref:

torchfort_tensor_list_add_tensor
--------------------------------
.. doxygenfunction:: torchfort_tensor_list_add_tensor

------

.. _torchfort_general_c-ref:

*******************
Supervised Learning
*******************

Model Creation
==============

.. _torchfort_create_model-ref:

torchfort_create_model
----------------------
.. doxygenfunction:: torchfort_create_model

------

.. _torchfort_create_distributed-model-ref:

torchfort_create_distributed_model
----------------------------------
.. doxygenfunction:: torchfort_create_distributed_model

------

Model Training/Inference
========================

.. _torchfort_train-ref:

torchfort_train
---------------
.. doxygenfunction:: torchfort_train

------

.. _torchfort_train_multiarg-ref:

torchfort_train_multiarg
------------------------
.. doxygenfunction:: torchfort_train_multiarg

------

.. _torchfort_inference-ref:

torchfort_inference
-------------------
.. doxygenfunction:: torchfort_inference

------

.. _torchfort_inference_multiarg-ref:

torchfort_inference_multiarg
----------------------------
.. doxygenfunction:: torchfort_inference_multiarg

------

Model Management
================

.. _torchfort_save_model-ref:

torchfort_save_model
--------------------
.. doxygenfunction:: torchfort_save_model

------

.. _torchfort_load_model-ref:

torchfort_load_model
--------------------
.. doxygenfunction:: torchfort_load_model

------

.. _torchfort_save_checkpoint-ref:

torchfort_save_checkpoint
-------------------------
.. doxygenfunction:: torchfort_save_checkpoint

------

.. _torchfort_load_checkpoint-ref:

torchfort_load_checkpoint
-------------------------
.. doxygenfunction:: torchfort_load_checkpoint

------

Weights and Biases Logging
==========================

.. _torchfort_wandb_log_int-ref:

torchfort_wandb_log_int
-----------------------
.. doxygenfunction:: torchfort_wandb_log_int

------

.. _torchfort_wandb_log_float-ref:

torchfort_wandb_log_float
-------------------------
.. doxygenfunction:: torchfort_wandb_log_float

------

.. _torchfort_wandb_log_double-ref:

torchfort_wandb_log_double
--------------------------
.. doxygenfunction:: torchfort_wandb_log_double

------

.. _torchfort_rl_c-ref:

**********************
Reinforcement Learning
**********************

Similar to other reinforcement learning frameworks such as `Spinning Up <https://spinningup.openai.com/en/latest/>`_ 
from OpenAI or `Stable Baselines <https://stable-baselines3.readthedocs.io/en/master/>`_, 
we distinguish between on-policy and off-policy algorithms since those two types require different APIs.

------

.. _torchfort_rl_off_policy_c-ref:

Off-Policy Algorithms
=====================

System Creation
-----------------------------------

Basic routines to create and register a reinforcement learning system in the internal registry. A (synchronous) data parallel distributed option is available.

.. _torchfort_rl_off_policy_create_system-ref:
		     
torchfort_rl_off_policy_create_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_create_system

------

.. _torchfort_rl_off_policy_create_distributed_system-ref:

torchfort_rl_off_policy_create_distributed_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_create_distributed_system

------

Training/Evaluation
-------------------

These routines are used for training the reinforcement learning system or for steering the environment. 

.. _torchfort_rl_off_policy_train_step-ref:

torchfort_rl_off_policy_train_step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_train_step

------

.. _torchfort_rl_off_policy_predict_explore-ref:

torchfort_rl_off_policy_predict_explore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_predict_explore

------

.. _torchfort_rl_off_policy_predict-ref:

torchfort_rl_off_policy_predict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_predict

------

.. _torchfort_rl_off_policy_evaluate-ref:

torchfort_rl_off_policy_evaluate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_evaluate

------

System Management
-----------------

The purpose of these routines is to manage the reinforcement learning systems internal data. It allows the user to add tuples to the replay buffer and query the system for readiness. Additionally, save and restore functionality is also provided.

.. _torchfort_rl_off_policy_update_replay_buffer-ref:

torchfort_rl_off_policy_update_replay_buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_update_replay_buffer

------

.. _torchfort_rl_off_policy_update_replay_buffer_multi-ref:

torchfort_rl_off_policy_update_replay_buffer_multi
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_update_replay_buffer_multi

------

.. _torchfort_rl_off_policy_is_ready-ref:

torchfort_rl_off_policy_is_ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_is_ready

------

.. _torchfort_rl_off_policy_save_checkpoint-ref:

torchfort_rl_off_policy_save_checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_save_checkpoint

------

.. _torchfort_rl_off_policy_load_checkpoint-ref:

torchfort_rl_off_policy_load_checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_load_checkpoint

------


Weights and Biases Logging
--------------------------

The reinforcement learning system performs logging for all involved networks automatically during training. The following routines are provided for additional logging of system relevant quantities, such as e.g. the accumulated reward.

.. _torchfort_rl_off_policy_wandb_log_int-ref:

torchfort_rl_off_policy_wandb_log_int
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_wandb_log_int

------

.. _torchfort_rl_off_policy_wandb_log_float-ref:

torchfort_rl_off_policy_wandb_log_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_wandb_log_float

------

.. _torchfort_rl_off_policy_wandb_log_double-ref:

torchfort_rl_off_policy_wandb_log_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_off_policy_wandb_log_double

------

.. _torchfort_rl_on_policy_c-ref:

On-Policy Algorithms
=====================

System Creation
-----------------------------------

Basic routines to create and register a reinforcement learning system in the internal registry. 
A (synchronous) data parallel distributed option is available.

.. _torchfort_rl_on_policy_create_system-ref:
		     
torchfort_rl_on_policy_create_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_create_system

------

.. _torchfort_rl_on_policy_create_distributed_system-ref:

torchfort_rl_on_policy_create_distributed_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_create_distributed_system

------

Training/Evaluation
-----------------------------------------

These routines are used for training the reinforcement learning system or for steering the environment. 

.. _torchfort_rl_on_policy_train_step-ref:

torchfort_rl_on_policy_train_step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_train_step

------

.. _torchfort_rl_on_policy_predict_explore-ref:

torchfort_rl_on_policy_predict_explore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_predict_explore

------

.. _torchfort_rl_on_policy_predict-ref:

torchfort_rl_on_policy_predict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_predict

------

.. _torchfort_rl_on_policy_evaluate-ref:

torchfort_rl_on_policy_evaluate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_evaluate

------

System Management
-----------------

The purpose of these routines is to manage the reinforcement learning systems internal data. 
It allows the user to add tuples to the replay buffer and query the system for readiness. 
Additionally, save and restore functionality is also provided.

.. _torchfort_rl_on_policy_update_rollout_buffer-ref:

torchfort_rl_on_policy_update_rollout_buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_update_rollout_buffer

------

.. _torchfort_rl_on_policy_update_rollout_buffer_multi-ref:

torchfort_rl_on_policy_update_rollout_buffer_multi
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_update_rollout_buffer_multi

------

.. _torchfort_rl_on_policy_is_ready-ref:

torchfort_rl_on_policy_is_ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_is_ready

------

.. _torchfort_rl_on_policy_save_checkpoint-ref:

torchfort_rl_on_policy_save_checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_save_checkpoint

------

.. _torchfort_rl_on_policy_load_checkpoint-ref:

torchfort_rl_on_policy_load_checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_load_checkpoint

------


Weights and Biases Logging
--------------------------

The reinforcement learning system performs logging for all involved networks automatically during training. 
The following routines are provided for additional logging of system relevant quantities, such as e.g. 
the accumulated reward.

.. _torchfort_rl_on_policy_wandb_log_int-ref:

torchfort_rl_on_policy_wandb_log_int
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_wandb_log_int

------

.. _torchfort_rl_on_policy_wandb_log_float-ref:

torchfort_rl_on_policy_wandb_log_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_wandb_log_float

------

.. _torchfort_rl_on_policy_wandb_log_double-ref:

torchfort_rl_on_policy_wandb_log_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchfort_rl_on_policy_wandb_log_double

