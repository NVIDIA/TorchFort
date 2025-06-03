.. _torchfort_config-ref:

#############################
TorchFort Configuration Files
#############################

The TorchFort library relies on a user-defined YAML configuration file to define several aspects of the training procedure,
with specific blocks to control:

  - general properties
  - model properties
  - optimizer properties
  - loss function properties
  - learning rate schedule properties

The following sections define each configuration block and available options.

Common
======

The following sections list configuration file blocks common to supervised learning and reinforcement learning configuration files.

General Properties
~~~~~~~~~~~~~~~~~~
The block in the configuration file defining general properties takes the following structure:

.. code-block:: yaml

  general:
    <option> = <value>

The following table lists the available options:

+-----------------------+-----------+------------------------------------------------------------------------------------------------+
| Option                | Data Type | Description                                                                                    |
+=======================+===========+================================================================================================+
| ``report_frequency``  | integer   | frequency of reported TorchFort training/validation output lines to terminal (default = ``0``) |
+-----------------------+-----------+------------------------------------------------------------------------------------------------+
| ``enable_wandb_hook`` | boolean   | flag to control whether wandb hook is active  (default = ``false``)                            |
+-----------------------+-----------+------------------------------------------------------------------------------------------------+
| ``verbose``           | boolean   | flag to control verbose output from TorchFort (default = ``false``)                            |
+-----------------------+-----------+------------------------------------------------------------------------------------------------+

For more information about the wandb hook, see :ref:`wandb_support-ref`.

.. _optimizer_properties-ref:

Optimizer Properties
~~~~~~~~~~~~~~~~~~~~
The block in the configuration file defining optimizer properties takes the following structure:

.. code-block:: yaml

  optimizer:
    type: <optimizer_type>
    parameters:
      <option> = <value>
    general:
      <option> = value

The :code:`general` block is optional.

The following table lists the available optimizer types:

+----------------+---------------------------------------+
| Optimizer Type | Description                           |
+================+=======================================+
| ``sgd``        | Stochastic Gradient Descent optimizer |
+----------------+---------------------------------------+
| ``adam``       | ADAM optimizer                        |
+----------------+---------------------------------------+


The following table lists the available parameter options by optimizer type:

+----------------+-------------------+-----------+-------------------------------------------------------------------------------------------+
| Optimizer Type | Option            | Data Type | Description                                                                               |
+================+===================+===========+===========================================================================================+
| ``sgd``        | ``learning_rate`` | float     | learning rate (default = ``0.001``)                                                       |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``momentum``      | float     | mometum factor (default = ``0.0``)                                                        |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``dampening``     | float     | dampening for momentum (default = ``0.0``)                                                |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``weight_decay``  | float     | weight decay/L2 penalty (default = ``0.0``)                                               |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``nesterov``      | boolean   | enables Nesterov momentum (default = ``false``)                                           |
+----------------+-------------------+-----------+-------------------------------------------------------------------------------------------+
| ``adam``       | ``learning_rate`` | float     | learning rate (default = ``0.001``)                                                       |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``beta1``         | float     | coefficient used for computing running average of gradient (default = ``0.9``)            |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``beta2``         | float     | coefficient use for computing running average of square of gradient (default = ``0.999``) |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``weight_decay``  | float     | weight decay/L2 penalty (default = ``0.0``)                                               |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``eps``           | float     | term added to denominator to improve numerical stability (default = ``1e-8``)             |
+                +-------------------+-----------+-------------------------------------------------------------------------------------------+
|                | ``amsgrad``       | boolean   | whether to use AMSGrad variant (default = ``false``)                                      |
+----------------+-------------------+-----------+-------------------------------------------------------------------------------------------+

The following table lists the available general options:

+------------------------------+-----------+------------------------------------------------------------------------------------------------+
| Option                       | Data Type | Description                                                                                    |
+==============================+===========+================================================================================================+
| ``grad_accumulation_steps``  | integer   | number of training steps to accumulate gradients between optimizer steps  (default = ``1``)    |
+------------------------------+-----------+------------------------------------------------------------------------------------------------+

.. _lr_schedule_properties-ref:

Learning Rate Schedule Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The block in the configuration file defining learning rate schedule properties takes the following structure:

.. code-block:: yaml

  lr_scheduler:
    type: <schedule_type>
    parameters:
      <option> = <value>

The following table lists the available schedule types:

+----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Schedule Type        | Description                                                                                                                                                  |
+======================+==============================================================================================================================================================+
| ``step``             | Decays learning rate by multiplicative factor every ``step_size`` training iterations                                                                        |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``multistep``        | Decays learning rate by multiplicative factor at user-defined training iteration milestones                                                                  |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``polynomial``       | Decays learning rate by polynomial function                                                                                                                  |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cosine_annealing`` | Decays learning rate using cosine annealing schedule. See PyTorch documentation of                                                                           |
|                      | `torch.optim.lr_scheduler.CosineAnnealingLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html>`_ for more details.  |
+----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

The following table lists the available options by schedule type:

+----------------------+-----------------+-----------------+------------------------------------------------------------------+
| Schedule Type        | Option          | Data Type       | Description                                                      |
+======================+=================+=================+==================================================================+
| ``step``             | ``step_size``   | integer         | Number of training steps between learning rate decay             |
+                      +-----------------+-----------------+------------------------------------------------------------------+
|                      | ``gamma``       | float           | Multiplicative factor of learning rate decay (default = ``0.1``) |
+----------------------+-----------------+-----------------+------------------------------------------------------------------+
| ``multistep``        | ``milestones``  | list of integer | Training step milestones for learning rate decay                 |
+                      +-----------------+-----------------+------------------------------------------------------------------+
|                      | ``gamma``       | float           | Multiplicative factor of learning rate decay (default = ``0.1``) |
+----------------------+-----------------+-----------------+------------------------------------------------------------------+
| ``polynomial``       | ``total_iters`` | integer         | Number of training iterations to decay the learning rate         |
+                      +-----------------+-----------------+------------------------------------------------------------------+
|                      | ``power``       | float           | The power of the polynomial (default = ``1.0``)                  |
+----------------------+-----------------+-----------------+------------------------------------------------------------------+
| ``cosine_annealing`` | ``eta_min``     | float           | Minumum learning rate (default = ``0.0``)                        |
+                      +-----------------+-----------------+------------------------------------------------------------------+
|                      | ``T_max``       | float           | Maximum number of iterations for decay                           |
+----------------------+-----------------+-----------------+------------------------------------------------------------------+

Supervised Learning
===================

The following sections list configuration file blocks specific to supervised learning configuration files.

.. _model_properties-ref:

Model Properties
~~~~~~~~~~~~~~~~~~
The block in the configuration file defining model properties takes the following structure:

.. code-block:: yaml

  model:
    type: <model_type>
    parameters:
      <option> = <value>

The following table lists the available model types:

+-----------------+------------------------------------------------+
| Model Type      | Description                                    |
+=================+================================================+
| ``torchscript`` | Load a model from an exported TorchScript file |
+-----------------+------------------------------------------------+
| ``mlp``         | Use built-in MLP model                         |
+-----------------+------------------------------------------------+


The following table lists the available options by model type:

+-----------------+-----------------+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Model Type      | Option          | Data Type        | Description                                                                                                                                                                        |
+=================+=================+==================+====================================================================================================================================================================================+
| ``torchscript`` | ``filename``    | string           | path to TorchScript exported model file                                                                                                                                            |
+-----------------+-----------------+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``mlp``         | ``layer_sizes`` | list of integers | sequence of input/output sizes for linear layers e.g., ``[16, 32, 4]`` will create two linear layers with input/output of 16/32 for the first layer and 32/4 for the second layer. |
+                 +-----------------+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                 | ``dropout``     | float            | probability of an element to be zeroed in dropout layers (default = ``0.0``)                                                                                                       |
+-----------------+-----------------+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Loss Properties
~~~~~~~~~~~~~~~~~~~~
The block in the configuration file defining loss properties takes the following structure:

.. code-block:: yaml

  loss:
    type: <loss_type>
    parameters:
      <option> = <value>

The following table lists the available loss types:

+-----------+------------------------------------------------------+
| Loss Type | Description                                          |
+=================+================================================+
| ``torchscript`` | Load a loss from an exported TorchScript file  |
+-----------------+------------------------------------------------+
| ``l1``          | L1/Mean Average Error                          |
+-----------------+------------------------------------------------+
| ``mse``         | Mean Squared Error                             |
+-----------------+------------------------------------------------+

The following table lists the available options by loss type:

+-----------------+---------------+-----------+-------------------------------------------------------------------------------------------------------------------+
| Loss Type       | Option        | Data Type | Description                                                                                                       |
+=================+===============+===========+===================================================================================================================+
| ``torchscript`` | ``filename``  | string    | path to TorchScript exported loss file                                                                            |
+-----------------+---------------+-----------+-------------------------------------------------------------------------------------------------------------------+
| ``l1``          | ``reduction`` | string    | Specifies type of reduction to apply to output. Can be either ``none``, ``mean`` or ``sum``. (default = ``mean``) |
+-----------------+---------------+-----------+-------------------------------------------------------------------------------------------------------------------+
| ``mse``         | ``reduction`` | string    | Specifies type of reduction to apply to output. Can be either ``none``, ``mean`` or ``sum``. (default = ``mean``) |
+-----------------+---------------+-----------+-------------------------------------------------------------------------------------------------------------------+


Reinforcement Learning
======================

The following sections list configuration file blocks specific to reinforcement learning system configuration files.

Reinforcement Learning Training Algorithm Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The block in the configuration file defining algorithm properties takes the following structure:

.. code-block:: yaml

  algorithm:
    type: <algorithm_type>
    parameters:
      <option> = <value>
      
The following table lists the available algorithm types:

+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| Algorithm Type | Description                                                                                                                                   |
+================+===============================================================================================================================================+
| ``ddpg``       | Deterministic Policy Gradient. See `DDPG documentation by OpenAI <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>`_ for details |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| ``td3``        | Twin Delayed DDPG. See `TD3 documentation by OpenAI <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ for details               |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| ``sac``        | Soft Actor Critic. See `SAC documentation by OpenAI <https://spinningup.openai.com/en/latest/algorithms/sac.html>`_ for details               |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------+

The following table lists the available options by algorithm type:

+----------------+----------------------------+------------+-------------------------------------------------------------------------------------------+
| Algorithm Type | Option                     | Data Type  | Description                                                                               |
+================+============================+============+===========================================================================================+
| ``ddpg``       | ``batch_size``             | integer    | batch size used in training                                                               |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``nstep``                  | integer    | number of steps for N-step training                                                       |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``nstep_reward_reduction`` | string     | reduction mode for N-step training (see below)                                            |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``gamma``                  | float      | discount factor                                                                           |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``rho``                    | boolean    | weight average factor for target weights (in some frameworks called rho = 1-tau)          |
+----------------+----------------------------+------------+-------------------------------------------------------------------------------------------+
| ``td3``        | ``batch_size``             | integer    | batch size used in training                                                               |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``nstep``                  | integer    | number of steps for N-step training                                                       |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``nstep_reward_reduction`` | string     | reduction mode for N-step training (see below)                                            |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``gamma``                  | float      | discount factor                                                                           |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``rho``                    | float      | weight average factor for target weights (in some frameworks called rho = 1-tau)          |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``num_critics``            | integer    | number of critic networks used                                                            |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``policy_lag``             | integer    | update frequency for the policy in units of critic updates                                |
+----------------+----------------------------+------------+-------------------------------------------------------------------------------------------+
| ``sac``        | ``batch_size``             | integer    | batch size used in training                                                               |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``nstep``                  | integer    | number of steps for N-step training                                                       |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``nstep_reward_reduction`` | string     | reduction mode for N-step training (see below)                                            |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``gamma``                  | float      | discount factor                                                                           |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``alpha``                  | float      | entropy regularization coefficient                                                        |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``rho``                    | boolean    | weight average factor for target weights (in some frameworks called rho = 1-tau)          |
+                +----------------------------+------------+-------------------------------------------------------------------------------------------+
|                | ``policy_lag``             | integer    | update frequency for the policy in units of value updates                                 |
+----------------+----------------------------+------------+-------------------------------------------------------------------------------------------+

The parameter ``nstep_reward_reduction`` defines how the reward is accumulated over N-step rollouts. The options are summarized in a table below (:math:`N` is the value from parameter ``nstep`` described above):

+------------------------------------------------+---------------------------------------------------------------------------------------+
| Reduction Mode                                 | Description                                                                           |
+================================================+=======================================================================================+
| ``sum`` or ``sum_no_skip``                     | :math:`r = \sum_{i=1}^{N^\ast} \gamma^{i-1} r_i`                                      |
+------------------------------------------------+---------------------------------------------------------------------------------------+
| ``mean`` or ``mean_no_skip``                   | :math:`r = \sum_{i=1}^{N^\ast} \gamma^{i-1} r_i / N^\ast`                             |
+------------------------------------------------+---------------------------------------------------------------------------------------+
| ``weighted_mean`` or ``weighted_mean_no_skip`` | :math:`r = \sum_{i=1}^{N^\ast} \gamma^{i-1} r_i / (\sum_{k=1}^{N^\ast} \gamma^{k-1})` |
+------------------------------------------------+---------------------------------------------------------------------------------------+

Here, the value of :math:`N^\ast` depends on whether reduction with or without skip is being chosen. In case of the former, :math:`N^\ast = N` and the replay buffer is searching for trajectories with **at least** :math:`N` steps. If the trajectory terminates earlier, the sample is skipped and a new one is searched. If **all trajectories are shorter** than :math:`N` steps, the replay buffer **will never find** a suitable sample. 

In this case, it is useful to use the modes with the additional suffix ``_no_skip``. In this case, :math:`N^{\ast}` in the formulas will be equal to the minimum of :math:`N` and the number of steps needed to reach the end of the trajectory. The regular and no-skip modes are both useful in different occasions, so it is important to be clear about how the reward structure has to be designed in order to achieve the desired goals.

Replay Buffer Properties
~~~~~~~~~~~~~~~~~~~~~~~~
The block in the configuration file defining algorithm properties takes the following structure:

.. code-block:: yaml

  replay_buffer:
    type: <replay_buffer_type>
    parameters:
      <option> = <value>
      
Currently, only type ``uniform`` is supported. The following table lists the available options:

+---------------------------+-----------------+-----------------+------------------------------------------------------------------+
| Replay Buffer Type        | Option          | Data Type       | Description                                                      |
+===========================+=================+=================+==================================================================+
| ``uniform``               | ``min_size``    | integer         | Minimum number of samples before buffer is ready for training    |
+                           +-----------------+-----------------+------------------------------------------------------------------+
|                           | ``max_size``    | integer         | Maximum capacity                                                 |
+---------------------------+-----------------+-----------------+------------------------------------------------------------------+

Action Properties
~~~~~~~~~~~~~~~~~
The block in the configuration file defining action properties takes the following structure:

.. code-block:: yaml

  action:
    type: <action_type>
    parameters:
      <option> = <value>

The following table lists the available options for every action type for ``ddpg`` and ``td3`` algorithms:

+----------------------------------------------+-------------------+------------+-------------------------------------------------------------------+
| Action Type                                  | Option            | Data Type  | Description                                                       |
+==============================================+===================+============+===================================================================+
| ``space_noise`` or ``parameter_noise``       | ``a_low``         | float      | lower bound for action value                                      |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``a_high``        | float      | upper bound for action value                                      |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``clip``          | float      | clip value for training noise                                     |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``sigma_train``   | float      | standard deviation for gaussian training noise                    |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``sigma_explore`` | float      | standard deviation for gaussian exploration noise                 |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``adaptive``      | bool       | flag to specify whether the standard deviation should be adaptive |
+----------------------------------------------+-------------------+------------+-------------------------------------------------------------------+
| ``space_noise_ou`` or ``parameter_noise_ou`` | ``a_low``         | float      | lower bound for action value                                      |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``a_high``        | float      | upper bound for action value                                      |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``clip``          | float      | clip value for training noise                                     |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``sigma_train``   | float      | standard deviation for Ornstein-Uhlenbeck training noise          |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``sigma_explore`` | float      | standard deviation for Ornstein-Uhlenbeck exploration noise       |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``xi``            | float      | mean reversion parameter for Ornstein-Uhlenbeck noise             |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``dt``            | float      | time-step parameter for Ornstein-Uhlenbeck noise                  |
+                                              +-------------------+------------+-------------------------------------------------------------------+
|                                              | ``adaptive``      | bool       | flag to specify whether the standard deviation should be adaptive |
+----------------------------------------------+-------------------+------------+-------------------------------------------------------------------+

The meaning for most of these parameters should be evident from looking at the details of the implementations for the various RL algorithms linked above. 
However, some parameters require a more detailed explanation: in general, the suffix ``_ou`` refers to stateful noise of Ornstein-Uhlenbeck type with zero drift. This noise type is often used if correlation between time steps is desired and thus popular in reinforcement learning. Check out the `wikipedia page <https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process>`_ for details.

The prefix ``space`` refers to applying the noise to the predicted ation directly. For example, if :math:`p` is our (deterministic) policy function, an exploration action using space noise type is obtained by computing 

.. math::

    \tilde{a} = \mathrm{clip}(p(\theta, s) + \mathcal{N}(0,\sigma_\mathrm{explore}), a_\mathrm{low}, a_\mathrm{high}) 
    
for any input state :math:`s` and policy weights :math:`\theta`. In case of parameter noise, the noise will be applied to each weight of :math:`p` instead. Hence, the noised action is computed  via

.. math::

    \tilde{a} = \mathrm{clip}(p(\theta + \mathcal{N}(0,\sigma_\mathrm{explore}), s), a_\mathrm{low}, a_\mathrm{high}) 
    
The parameter ``adaptive`` specifies whether the noise variance :math:`\sigma` should be taken relative to the magnitude of the action magnitudes or weight magnitudes for space and parameter noise respectively. In terms of the former, this would mean that

.. math::
    
    a &= p(\theta, s)
    
    \tilde{a} &= \mathrm{clip}(a + \sigma_\mathrm{explore}\,\mathcal{N}(0,\|a\|), a_\mathrm{low}, a_\mathrm{high}) 

and analogous for parameter noise.

Whichever noise type and parameters are the best highly depends on the behavior of the environment and therefore we cannot give a general recommendation.

For algorithm type ``sac``, only action bounds are supported as the noise is built into the algorithm and cannot be customized.

Policy and Critic Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The block in the configuration file defining model properties for actor/policy and critic/value are similar to the supervised learning case (see :ref:`model_properties-ref`). In this case, TorchFort supports different model properties for policy and critic. The block configuration looks as follows:

.. code-block:: yaml

  critic_model:
    type: <critic_model_type>
    parameters:
      <option> = <value>

  policy_model:
    type: <policy_model_type>
    parameters:
      <option> = <value>

Refer to the :ref:`model_properties-ref` for available model types and options.

.. note::

    For algorithms which use multiple critics networks such as TD3, the critic model is copied internally ``num_critic`` times and the weights are randomly initialized for each of these models independently.
    
.. note::
    
    In case of SAC algorithm, make sure that the policy network not only returns the mean actions value tensor but also the log probability sigma tensor. As an example see the policy function implementation of `stable baselines <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py>`_.

Learning Rate Schedule Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For reinforcement learning, TorchFort supports different learning rate schedules for policy and critic. The block configuration looks as follows:

.. code-block:: yaml

  critic_lr_scheduler:
    type: <schedule_type>
    parameters:
      <option> = <value>

  policy_lr_scheduler:
    type: <schedule_type>
    parameters:
      <option> = <value>

Refer to the :ref:`lr_schedule_properties-ref` for available scheduler types and options.

