#include "torchfort.h"
#include "environments.h"

enum EnvMode { Constant, Predictable, Delayed, Action, ActionState };

bool TestSystem(const EnvMode mode, const std::string& system, unsigned int num_explore_iters, unsigned int num_exploit_iters) {

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

  // set up tensors:
  torch::Tensor state = torch::zeros(state_shape, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor state_new = torch::empty_like(state);
  torch::Tensor action = torch::zeros(action_shape, torch::TensorOptions().dtype(torch::kFloat32));

  // set up environment
  std::shared_ptr<Environment> env;
  int num_episodes;
  if (mode == Constant) {
    env = std::make_shared<ConstantRewardEnvironment>(1, state_shape, action_shape, 1.);
    num_episodes = num_explore_iters + num_exploit_iters;
  } else if (mode == Predictable) {
    env = std::make_shared<PredictableRewardEnvironment>(1, state_shape, action_shape);
    num_episodes = num_explore_iters + num_exploit_iters;
  } else if (mode == Delayed) {
    env = std::make_shared<DelayedRewardEnvironment>(2, state_shape, action_shape, 1.);
    num_episodes = num_explore_iters + num_exploit_iters;
  } else if (mode == Action) {
    env = std::make_shared<ActionRewardEnvironment>(1, state_shape, action_shape);
    num_episodes = num_explore_iters + num_exploit_iters;
  } else if (mode == ActionState) {
    env = std::make_shared<ActionStateRewardEnvironment>(1, state_shape, action_shape);
    num_episodes = num_explore_iters + num_exploit_iters;
  }

  // set up td3 learning systems
  std::string filename = "configs/" + system + ".yaml";
  torchfort_result_t tstat = torchfort_rl_off_policy_create_system("test", filename.c_str(),
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
	tstat = torchfort_rl_off_policy_predict_explore("test",
							state.data_ptr(), 2, state_batch_shape.data(),
							action.data_ptr(), 2, action_batch_shape.data(),
							TORCHFORT_FLOAT, 0);
      } else {
	// exploit
	tstat = torchfort_rl_off_policy_predict("test",
						state.data_ptr(), 2, state_batch_shape.data(),
						action.data_ptr(), 2, action_batch_shape.data(),
						TORCHFORT_FLOAT, 0);
      }

      // do environment step
      std::tie(state_new, reward, done) = env->step(action);

      // update replay buffer
      tstat = torchfort_rl_off_policy_update_replay_buffer("test",
							   state.data_ptr(), state_new.data_ptr(), 1, state_shape.data(),
							   action.data_ptr(), 1, action_shape.data(), &reward, done,
							   TORCHFORT_FLOAT, 0);

      // perform training step if requested:
      tstat = torchfort_rl_off_policy_is_ready("test", is_ready);
      if (is_ready) {
	tstat = torchfort_rl_off_policy_train_step("test", &p_loss, &q_loss, 0);
      }

      // evaluate policy:
      tstat = torchfort_rl_off_policy_evaluate("test",
					       state.data_ptr(), 2, state_batch_shape.data(),
					       action.data_ptr(), 2, action_batch_shape.data(),
					       &reward_estimate, 2, reward_batch_shape.data(),
					       TORCHFORT_FLOAT, 0);

      std::cout << "episode : " << e
		<< " step: " << i
		<< " state: "  << state.item<float>()
		<< " action: " << action.item<float>()
		<< " reward: " << reward
		<< " q: " << reward_estimate
		<< " done: " << done << std::endl;

      // copy tensors
      state.copy_(state_new);

      // increase counter:
      i++;
      iter++;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {

  std::vector<std::string> system_names = {"ddpg"};

  for (auto& system : system_names) {
    //TestSystem(Constant, system, 20000, 0);

    //TestSystem(Predictable, system, 20000, 0);

    //TestSystem(Delayed, system, 20000, 0);

    TestSystem(Action, system, 20000, 1000);

    //TestSystem(ActionState, system, 20000, 1000);
  }

  return 0;
}
