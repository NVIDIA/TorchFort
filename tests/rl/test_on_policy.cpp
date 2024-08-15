#include "torchfort.h"
#include "environments.h"

enum EnvMode { Constant, Predictable, Delayed, Action, ActionState };

std::tuple<float, float> TestSystem(const EnvMode mode, const std::string& system,
				    unsigned int num_explore_iters, unsigned int num_exploit_iters,
				    unsigned int num_eval_iters=100, unsigned int num_grad_steps=1,
				    bool verbose=false) {

  // set seed
  torch::manual_seed(666);
  
  // create dummy env:
  std::vector<int64_t> state_shape{1};
  std::vector<int64_t> action_shape{1};
  std::vector<int64_t> state_batch_shape{1,1};
  std::vector<int64_t> action_batch_shape{1,1};
  std::vector<int64_t> reward_batch_shape{1,1};
  float reward, reward_estimate, p_loss, q_loss;
  bool done;
  bool is_ready;
  float qdiff = 0.;
  float running_reward = 0.;

  // set up tensors:
  torch::Tensor state = torch::zeros(state_shape, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor state_new = torch::empty_like(state);
  torch::Tensor action = torch::zeros(action_shape, torch::TensorOptions().dtype(torch::kFloat32));

  // set up environment
  std::shared_ptr<Environment> env;
  int num_episodes, num_train_iters, episode_length;
  if (mode == Constant) {
    episode_length = 1;
    env = std::make_shared<ConstantRewardEnvironment>(episode_length, state_shape, action_shape, 1.);
  } else if (mode == Predictable) {
    episode_length = 1;
    env = std::make_shared<PredictableRewardEnvironment>(episode_length, state_shape, action_shape);
  } else if (mode == Delayed) {
    episode_length = 2;
    env = std::make_shared<DelayedRewardEnvironment>(episode_length, state_shape, action_shape, 1.);
  } else if (mode == Action) {
    episode_length = 1;
    env = std::make_shared<ActionRewardEnvironment>(episode_length, state_shape, action_shape);
  } else if (mode == ActionState) {
    episode_length = 1;
    env = std::make_shared<ActionStateRewardEnvironment>(episode_length, state_shape, action_shape);
  }
  num_train_iters = num_explore_iters + num_exploit_iters;
  num_episodes = (num_train_iters + num_eval_iters) / episode_length;

  // set up td3 learning systems
  std::string filename = "configs/" + system + ".yaml";
  torchfort_result_t tstat = torchfort_rl_on_policy_create_system("test", filename.c_str(),
								  TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  if (tstat != TORCHFORT_RESULT_SUCCESS) {
    throw std::runtime_error("RL system creation failed");
  }

  // do training loop: initial state
  int iter = 0;
  std::tie(state, reward) = env->initialize();
  for (unsigned int e=0; e<num_episodes; ++e) {
    done = false;
    int i=0;
    while (!done) {
      if (iter < num_explore_iters) {
	// explore
	tstat = torchfort_rl_on_policy_predict_explore("test",
						       state.data_ptr(), 2, state_batch_shape.data(),
						       action.data_ptr(), 2, action_batch_shape.data(),
						       TORCHFORT_FLOAT, 0);
      } else {
	// exploit
	tstat = torchfort_rl_on_policy_predict("test",
					       state.data_ptr(), 2, state_batch_shape.data(),
					       action.data_ptr(), 2, action_batch_shape.data(),
					       TORCHFORT_FLOAT, 0);
      }

      // do environment step
      std::tie(state_new, reward, done) = env->step(action);

      if (iter < num_train_iters) {
	// update replay buffer
	tstat = torchfort_rl_on_policy_update_rollout_buffer("test",
							     state.data_ptr(), 1, state_shape.data(),
							     action.data_ptr(), 1, action_shape.data(), &reward, done,
							     TORCHFORT_FLOAT, 0);

	// perform training step if requested:
	tstat = torchfort_rl_on_policy_is_ready("test", is_ready);
	// iterate till there are no more samples inside the buffer:
	if (is_ready) {
	  for (unsigned int k=0; k<num_grad_steps; ++k) {
	    tstat = torchfort_rl_on_policy_train_step("test", &p_loss, &q_loss, 0);
	  }
	  tstat = torchfort_rl_on_policy_reset_rollout_buffer("test");
	}
      }

      // evaluate policy:
      tstat = torchfort_rl_on_policy_evaluate("test",
					      state.data_ptr(), 2, state_batch_shape.data(),
					      action.data_ptr(), 2, action_batch_shape.data(),
					      &reward_estimate, 2, reward_batch_shape.data(),
					      TORCHFORT_FLOAT, 0);

      if (iter >= num_train_iters) {
	qdiff += std::abs(reward - reward_estimate);
	running_reward += reward;
      }

      if (verbose) {
	std::cout << "episode : " << e
		  << " step: " << i
		  << " state: "  << state.item<float>()
		  << " action: " << action.item<float>()
		  << " reward: " << reward
		  << " q: " << reward_estimate
		  << " done: " << done << std::endl;
      }

      // copy tensors
      state.copy_(state_new);

      // increase counter:
      i++;
      iter++;
    }
  }

  // compute averages:
  qdiff /= float(num_eval_iters);
  running_reward /= float(num_eval_iters);

  if (verbose) {
    std::cout << "Q-difference: " << qdiff << " average reward: " << running_reward << std::endl;
  }

  return std::make_tuple(qdiff, running_reward);
}

int main(int argc, char *argv[]) {

  std::vector<std::string> system_names = {"ppo"};

  for (auto& system : system_names) {
    //TestSystem(Constant, system, 20000, 0, 100, 8, true);

    TestSystem(Predictable, system, 20000, 0, 100, 8, true);

    //TestSystem(Delayed, system, 40000, 0, 100, 8, true);

    //TestSystem(Action, system, 40000, 1000, 100, 8, true);

    //TestSystem(ActionState, system, 40000, 1000, 100, 8, true);
  }

  return 0;
}
