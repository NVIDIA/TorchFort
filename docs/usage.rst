###########
Usage Guide
###########

TorchFort provides API functions to support two common Deep Learning paradigms: supervised learning and reinforcement learning. Due to the differences in structure between these approaches, the API
is organized largely into separate sets of functions designed to facilitate each approach. 

.. _supervised_learning-ref:

Supervised Learning
===================

Supervised learning approaches are by far the most common within Deep Learning. TorchFort provides functions to create/load/save models and run supervised training and inference on models, all directly within a C/C++/Fortran program.
For a complete list of functions available, see :ref:`torchfort_general_c-ref` for C/C++ or :ref:`torchfort_general_f-ref` for Fortran bindings.


Here, we will provide a high-level guide to using TorchFort for a supervised learning problem. For a more detailed usage example, we refer to the ``simulation`` example provided in this repository in the ``examples/fortran`` directory.

Creating a model
----------------
First, create a model instance.

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_create_model(model_name, configfile, device)

  .. code-tab:: c++

    istat = torchfort_create_model(model_name, configfile, device);

This function instantiates a model, associating a string ``model_name`` with a model instance created using a YAML configuration file located at path ``configfile``. See :ref:`torchfort_config-ref` for details
on the YAML configuration file format and options.
The last argument ``device`` specifies where to place and run the model on. Values 0 or greater correspond to GPU device indices (e.g., a value of ``0`` will place the model on GPU device 0). The constant ``TORCHFORT_DEVICE_GPU (-1)`` can be used to place the model on the CPU.

Run a Training Step
-------------------
Run a training step on input and label data generated from your program.

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_train(model_name, input, label, loss, stream)

  .. code-tab:: c++

    istat = torchfort_train(model_name,
                            input, input_dim, input_shape,
                            label, label_dim, label_shape,
                            loss, dtype, stream);

This function passes the ``input`` and ``label`` data to the model instance associated with ``model_name`` and runs a training iteration, updating the model parameters as necessary. The training loss is returned by reference to the ``loss`` variable.
In the Fortran API, ``input`` and ``label`` are multi-dimensional Fortran arrays, with dimension/shape/datatype information automatically inferred in the interface.
The ``stream`` argument specifies a CUDA stream to enqueue the training operations into. This argument is ignored if the model was placed on the CPU.

Run Inference to generate predicted output
------------------------------------------
After training the model on one or more samples, we can run an inference to generate a predicted output.

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_inference(model_name, input, output, stream)

  .. code-tab:: c++

    istat = torchfort_inference(model_name,
                                input, output_dim, output_shape,
                                output, output_dim, output_shape,
                                dtype, stream);

This function passes the ``input`` data to the model instance associated with ``model_name`` and runs an inference, returning the predicted output from the model to ``output``.
In the Fortran API, ``input`` and ``output`` are multi-dimensional Fortran arrays, with dimension/shape/datatype information automatically inferred in the interface.
The ``stream`` argument specifies a CUDA stream to enqueue the inference operations into. This argument is ignored if the model was placed on the CPU.


Checkpoint/Restart
------------------
The complete training state (current model parameters, optimizer state, learning rate scheduler progress) can be written to a checkpoint directory at any point during training.

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_save_checkpoint(model_name, directory_name)

  .. code-tab:: c++

    istat = torchfort_save_checkpoint(model_name, directory_name);

This function will write checkpoint data for the model instance associated with ``model_name`` to the directory provided by ``directory_name``. The directory will contain several subdirectories and files
containing required information for restart.

To load a checkpoint into a created model instance, run the following:

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_load_checkpoint(model_name, directory_name, step_train, step_inference)

  .. code-tab:: c++

    istat = torchfort_load_checkpoint(model_name, directory_name, step_train, step_inference);

This function will load the checkpoint data from the directory ``directory_name`` into the model instance associated with ``model_name``. ``step_train`` and ``step_inference`` are the checkpointed training step and inference step
respectively, returned by reference.

For inference, only the model and not a full checkpoint needs to be loaded. For this, run the following function instead:

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_load_model(model_name, model_file)

  .. code-tab:: c++

    istat = torchfort_load_checkpoint(model_name, model_file);

This function will load the model data from the file ``model_file`` into the model instance associated with ``model_name``. The model file
can be one generated using the ``torchfort_save_model`` function or one found within a saved checkpoint directory, found at ``<checkpoint directory name>/model.pt``.

Multi-argument API
------------------
While the functions described in the previous sections cover the most models which work on single input/label/output tensors, TorchFort supports more complex models that require multiple input/label/output tensors
as well as custom loss functions with additional tensor arguments (e.g., an indexing tensor for masking). To accomplish this, we have alternative training and inference functions that accept TorchFort tensor lists as arguments, which contain
one or more tensors.

To run a training step using the multi-argument training function, run the following:

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_train_multiarg(inputs, labels, loss_val, extra_loss_args, stream)

  .. code-tab:: c++

    istat = torchfort_train_multiarg(inputs, labels, loss_val, extra_loss_args, stream);

where ``inputs``, ``labels``, and ``extra_loss_args`` are Torchfort tensor lists.

You must create and populate tensor lists to use with this function. For example, to create an ``inputs`` tensor list containing two tensors, run the following:

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_tensor_list_create(inputs)
    istat = torchfort_tensor_list_add_tensor(inputs, input1)
    istat = torchfort_tensor_list_add_tensor(inputs, input2)

  .. code-tab:: c++

    istat = torchfort_tensor_list_create(inputs);
    istat = torchfort_tensor_list_add_tensor(inputs, input1, input1_dim, input1_shape, dtype)
    istat = torchfort_tensor_list_add_tensor(inputs, input2, input2_dim, input2_shape, dtype)

In the Fortran API, ``input1`` and ``input2`` are multi-dimensional Fortran arrays, with dimension/shape/datatype information automatically inferred in the interface. The ``torchfort_tensor_list_add_tensor`` function adds tensor data by reference
so changes to the externally provided memory buffer will modify the tensors stored in the list. This is convenient as it enables creating these tensor lists once and reusing them as your program updates the underlying values. 

Internally, the training backend unpacks the tensor lists and provides them to the model and loss functions as follows:

.. code-block:: c++

  predictions = model.forward(inputs[0], inputs[1], ..., inputs[n]);
  loss_val = loss.forward(predictions[0], predictions[1], ..., predictions[n],
                          labels[0], labels[1], ..., labels[n],
                          extra_loss_args[0], extra_loss_args[1], ..., extra_loss_args[n]);

As you can see, this multi-argument training function enables more complexity in model and loss function definitions.

Similarly to training, to run an inference using the multi-argument inference function, run the following:

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_inference_multiarg(inputs, outputs, stream)

  .. code-tab:: c++

    istat = torchfort_inference_multiarg(inputs, outputs, stream)

where ``inputs`` and ``outputs`` are Torchfort tensor lists. The tensor lists are unpacked and provided to the model similarly to training, with the ``outputs`` tensor list containing the predicted output from the model, in order.


For a more detailed usage example of the multi-argument API, we refer to the ``graph`` example provided in this repository in the ``examples/fortran`` directory.

.. _reinforcement_learning-ref:

Reinforcement Learning
======================

Most modern reinforcement learning (RL) algorithms utilize different neural networks for policy and value functions and often require keeping track of multiple copies for each of the models (e.g., the `DDPG <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>`_ or `TD3 <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ algorithms). Furthermore, the training algorithm causes those networks to interact in a non-trivial way. Additionally, off-policy methods require keeping track of historic data stored in a replay buffers, keeping track of actions and system states and rewards. Many reinforcement learning algorithms are deterministic in nature and thus require manual injection of randomness into the training process by employing parameter or action space noise. 

The practitioner who seeks to employ these methods is often not interested in implementing all these features by hand, since it would significantly increase the complexity of the wrapped simulation application. In fact, most of these features can be reused among a broad range of applications. Therefore, instead of providing access to all the individual parts, TorchFort encapsulates all these details into a structure we call an ``rl_system`` and abstracts all the bookeeping away from the user. The user only has to configure the system and then ensure that data is added to the replay buffer and training steps or action predictions are performed whenever it is necessary from the simulation code. Routines for reinforcement learning routines are prefixed with ``torchfort_rl_off_policy``. 
Currently, TorchFort only provides off-policy methods as those have been proven to be most versatile and powerful for a broad range of tasks. On-policy methods may be added in the future, hence we distinguish between these two cases in the TorchFort API.

We will provide a high-level guide for users who would like to add reinforcement learning functionality to their code. We assume that the user is familiar with the basic concepts of deep and reinforcement learning and understands the possibilities and limitations of these methods. This guide is far from exhaustive and for a complete list of reinforcement learning functions see :ref:`torchfort_rl_c-ref` for C/C++ or :ref:`torchfort_rl_f-ref` for Fortran respectively. We also suggest reviewing the ``example`` folder where we have implemented the cartpole RL problem in C++ using TorchFort.

Creating an RL system
---------------------

To start, a TorchFort rl system has to be initialized with the call:

.. tabs::

  .. code-tab:: fortran
  
    istat = torchfort_rl_off_policy_create_system(system_name, configfile, model_device, rb_device)
    
  .. code-tab:: c++

    istat = torchfort_rl_off_policy_create_system(system_name, configfile, model_device, rb_device);

where ``system_name`` is a name which used by TorchFort to identify the system created using  YAML configuration file ``configfile``. See :ref:`torchfort_config-ref` for details on the YAML configuration file format and options.
The last two arguments ``model_device`` and ``rb_device`` specify where to place the model and replay buffer on respectively. Values 0 or greater correspond to GPU device indices (e.g., a value of ``0`` will place the model or replay buffer on GPU device 0). The constant ``TORCHFORT_DEVICE_GPU (-1)`` can be used to place the model or replay buffer on the CPU.

Replay Buffer Management
------------------------

The user application (usually called *environment* in the RL context) will generate states and rewards based on actions suggested by the policy function or some other mechanism. For off-policy methods, this information needs to be passed to the replay buffer from which the training process will sample. This is performed via:
    
.. tabs::

  .. code-tab:: fortran
  
    istat = torchfort_rl_off_policy_update_replay_buffer(system_name, state_old, action, state_new, reward, terminal, stream)
    
  .. code-tab:: c++
  
    istat = torchfort_rl_off_policy_update_replay_buffer(system_name, 
                                                         state_old, state_new, state_dim, state_shape,
                                                         action, action_dim, action_shape, 
                                                         reward, terminal, dtype, stream);

``state_old`` is an array describing the old environment state to which ``action`` is applied, resulting in a new environment state ``state_new`` and a corresponding scalar ``reward``. The variable ``terminal`` is a flag which specifies whether the end of an episode is reached. In the Fortran API, the states and actions are multi-dimensional Fortran arrays with dimension/shape/datatype automatically inferred in the interface. In the C++ API, all arrays are ``void`` pointers and the state and action dimensions and shapes have to be passed explicitly. The ``stream`` argument specifies a CUDA stream to enqueue the update operations into. This argument is ignored if the replay buffer was placed on the CPU.

.. note::
    The update replay buffer functions expect a single tuple containing single samples and hence no batch dimension should be included.

.. warning::
    It is important to mention that this function should be called in causal order on the state tuples, i.e., the data inserted into the replay buffer should contain subsequent steps of the environment. In case of n-step rollouts, the sampling logic assumes that the list of tuples are ordered causally and different trajectories are separated by a terminal flag set to true for the last step in trajectory. Any non-causal ordering would likely yield sub-optimal training performance.

Before training can start, the replay buffer needs to contain a minimum number of state-action tuples. The readiness can be queried with:

.. tabs::

  .. code-tab:: fortran
  
    istat = torchfort_rl_off_policy_is_ready(system_name, ready)
    
  .. code-tab:: c++
  
    istat = torchfort_rl_off_policy_is_ready(system_name, ready);


Generating Action Predictions
-----------------------------

TorchFort provides the following two functions to generate action predictions from the policy network infrastructure:

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_rl_off_policy_predict(system_name, state, action, stream)
    
    istat = torchfort_rl_off_policy_predict_explore(system_name, state, action, stream)
    
  .. code-tab:: c++
  
    istat = torchfort_rl_off_policy_predict(system_name, 
                                            state, state_dim, state_shape,
                                            action, action_dim, action_shape,
                                            dtype, stream);
                                            
    istat = torchfort_rl_off_policy_predict_explore(system_name, 
                                                    state, state_dim, state_shape,
                                                    action, action_dim, action_shape,
                                                    dtype, stream);
    
Both functions predict an ``action`` based on a ``state``. The first variant generates a deterministic prediction from the target network (for algorithms which use target networks, i.e., a shadow network which gets updated less often than the active networks the regular weight updates are applied to). The second variant generates a prediction using the active network and also adds noise as specified in the configuration file.
The ``stream`` argument specifies a CUDA stream to enqueue the prediction operations into. This argument is ignored if the model was placed on the CPU.

.. note::
    The prediction methods are inference methods and thus expect a batch of data. Therefore, the state and action arrays need to include a batch dimension.

Training Step
-------------

Once the system is ready, a training step (forward, backward, optimizer step, learning rate decay) can be performed by calling:

.. tabs::

  .. code-tab:: fortran
  
    istat = torchfort_rl_off_policy_train_step(system_name, p_loss, q_loss, stream)
    
  .. code-tab:: c++
  
    istat = torchfort_rl_off_policy_train_step(system_name, p_loss, q_loss, stream);
    
This function will retrieve a single batch from the replay buffer and perform a training step, populating the variables ``p_loss``, ``q_loss`` by reference. 
The ``stream`` argument specifies a CUDA stream to enqueue the training operations into. This argument is ignored if the model was placed on the CPU.

.. note::
    If the RL algorithm uses a policy lag bigger than zero, for some steps only the value function networks are updated. In this case, ``p_loss`` is not computed and will be equal to zero.

Checkpoint/Restart
------------------

At any time during or after training, a checkpoint of the full system can be stored using:

.. tabs::

  .. code-tab:: fortran
  
    istat = torchfort_rl_off_policy_save_checkpoint(system_name, directory_name)
    
  .. code-tab:: c++
  
    istat = torchfort_rl_off_policy_save_checkpoint(system_name, directory_name);
    
This will save everything from the RL system with name ``system_name`` into the directory with name ``directory_name``. This checkpoint includes all models, i.e. all value and policy functions, active and target, multiple critics, etc. It will also save all optimizer and learning rate scheduler states. Additionally, the function will also save the replay buffer data and additional information such as episode number. This is required for being able to restore the full state of the RL system via:

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_rl_off_policy_load_checkpoint(system_name, directory_name)
    
  .. code-tab:: c++
  
    istat = torchfort_rl_off_policy_load_checkpoint(system_name, directory_name)
    
This function is only required if RL training from the checkpoint should be resumed. In an inference setting, where only the policy should be run, the method:

.. tabs::

  .. code-tab:: fortran

    istat = torchfort_load_model(model_name, policy_model_file);
    
  .. code-tab:: c++
  
    istat = torchfort_load_model(model_name, policy_model_file);
    
can be used instead. The model instance should be created beforehand using the methods described in the :ref:`supervised_learning-ref` section. 

