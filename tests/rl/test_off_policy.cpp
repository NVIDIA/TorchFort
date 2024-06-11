#include "torchfort.h"
#include "environments.h"


bool TestConstantEnv() {
  
  // create dummy env:
  std::vector<int64_t> state_shape{1};
  std::vector<int64_t> action_shape{1};
  std::vector<int64_t> state_batch_shape{1,1};
  std::vector<int64_t> action_batch_shape{1,1};
  std::vector<int64_t> reward_batch_shape{1,1};
  unsigned int num_episodes = 20000;
  unsigned int num_iters_per_episode = 1;
  float reward, reward_estimate, p_loss, q_loss;
  bool final_state=false;
  bool is_ready;

  // set up tensors:
  torch::Tensor state = torch::zeros(state_shape, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor state_new = torch::empty_like(state);
  torch::Tensor action = torch::zeros(action_shape, torch::TensorOptions().dtype(torch::kFloat32));

  // set up environment
  auto env = ConstantRewardEnvironment(state_shape, action_shape, 1.);

  // set up td3 learning systems
  torchfort_result_t tstat = torchfort_rl_off_policy_create_system("constant_td3", "configs/td3.yaml",
								   TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  if (tstat != TORCHFORT_RESULT_SUCCESS) {
    throw std::runtime_error("RL system creation failed");
  }

  // do training loop
  for (unsigned int e=0; e<num_episodes; ++e) {
    for (unsigned int i=0; i<num_iters_per_episode; ++i) {    
      tstat = torchfort_rl_off_policy_predict_explore("constant_td3",
						      state.data_ptr(), 2, state_batch_shape.data(),
						      action.data_ptr(), 2, action_batch_shape.data(),
						      TORCHFORT_FLOAT, 0);

      // do environment step
      std::tie(state_new, reward) = env.step(action);
      
      // update replay buffer
      final_state = (i == num_iters_per_episode-1);
      tstat = torchfort_rl_off_policy_update_replay_buffer("constant_td3",
							   state.data_ptr(), state_new.data_ptr(), 1, state_shape.data(),
							   action.data_ptr(), 1, action_shape.data(), &reward, final_state,
							   TORCHFORT_FLOAT, 0);
      
      // perform training step if requested:
      tstat = torchfort_rl_off_policy_is_ready("constant_td3", is_ready);
      std::cout << i << " " << is_ready << std::endl;
      if (is_ready) {
	tstat = torchfort_rl_off_policy_train_step("constant_td3", &p_loss, &q_loss, 0);
      }
      
      // evaluate policy:
      tstat = torchfort_rl_off_policy_evaluate("constant_td3",
					       state.data_ptr(), 2, state_batch_shape.data(),
					       action.data_ptr(), 2, action_batch_shape.data(),
					       &reward_estimate, 2, reward_batch_shape.data(),
					       TORCHFORT_FLOAT, 0);
      
      std::cout << "episode : " << e << " step: " << i << " state: "  << state.item<float>() << " action: " << action.item<float>() << " reward: " << reward << " q: " << reward_estimate << std::endl;
      
      // copy tensors
      state.copy_(state_new);
    }
  }
  return true;
}


int main(int argc, char *argv[]) {

  TestConstantEnv();
  
  return 0;
}
