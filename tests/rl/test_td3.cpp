#include "torchfort.h"
#include "environments.h"

int main(int argc, char* argv[]) {
  
  // create dummy env:
  torch::IntArrayRef state_shape{1};
  torch::IntArrayRef action_shape{1};

  // set up environment
  auto env = ConstantRewardEnvironment(state_shape, action_shape, 1.);

  // set up td3 learning systems
  torchfort_result_t tstat = torchfort_rl_off_policy_create_system("constant_td3", "configs/td3.yaml",
								   TORCHFORT_DEVICE_CPU, TORCHFORT_DEVICE_CPU);
  if (tstat != TORCHFORT_RESULT_SUCCESS) {
    throw std::runtime_error("RL system creation failed");
  }
  
}
