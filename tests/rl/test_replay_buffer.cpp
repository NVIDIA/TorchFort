#include <torch/torch.h>
#include "internal/rl/replay_buffer.h"

using namespace torchfort;
using namespace torch::indexing;

// helper functions
std::shared_ptr<rl::UniformReplayBuffer> getTestReplayBuffer(int buffer_size, float gamma=0.95, int nstep=1) {

  torch::NoGradGuard no_grad;
  
  auto rbuff = std::make_shared<rl::UniformReplayBuffer>(buffer_size, buffer_size, gamma, nstep, rl::RewardReductionMode::Sum, -1);

  // initialize rng
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(1,5);

  // fill the buffer
  float	reward;
  bool done;
  torch::Tensor state = torch::zeros({1}, torch::kFloat32), state_p, action;
  for (unsigned int i=0; i<buffer_size; ++i) {
    action = torch::ones({1}, torch::kFloat32) * static_cast<float>(dist(rng));
    state_p = state + action;
    reward = action.item<float>();
    done = false;
    rbuff->update(state, action, state_p, reward, done);
    state.copy_(state_p);
  }

  return rbuff;
}

void print_buffer(std::shared_ptr<rl::UniformReplayBuffer> buffp) {
  torch::Tensor stens, atens, sptens;
  float reward;
  bool done;
  for(unsigned int i=0; i<buffp->getSize(); ++i) {
    std::tie(stens, atens, sptens, reward, done) = buffp->get(i);
    std::cout << "entry " << i << ": s = " << stens.item<float>() << " a = " << atens.item<float>()
	      << " s' = " << sptens.item<float>() << " r = " << reward << " d = " << done << std::endl;
  }
}

// check if entries are consistent
bool TestEntryConsistency() {
  // some parameters
  unsigned int batch_size = 32;
  unsigned int buffer_size = 4 * batch_size;

  // get replay buffer
  auto rbuff = getTestReplayBuffer(buffer_size, 0.95, 1);

  // sample
  torch::Tensor stens, atens, sptens, rtens, dtens;
  float state_diff = 0;
  float reward_diff = 0.;
  for (unsigned int i=0; i<4; ++i) {
    std::tie(stens, atens, sptens, rtens, dtens) = rbuff->sample(batch_size);

    // compute differences:
    state_diff += torch::sum(torch::abs(stens + atens - sptens)).item<float>();
    reward_diff += torch::sum(torch::abs(atens - rtens)).item<float>();
  }
  // make sure values are consistent:
  std::cout << "TestEntryConsistency: state-diff " << state_diff << " reward-diff " << reward_diff << std::endl;

  return (state_diff < 1e-7) && (reward_diff < 1e-7);
}


// check if ordering between entries are consistent
bool TestTrajectoryConsistency() {
  // some parameters
  unsigned int batch_size = 32;
  unsigned int buffer_size = 4 * batch_size;

  // get replay buffer
  auto rbuff = getTestReplayBuffer(buffer_size, 0.95, 1);
  
  // get a few items and their successors:
  torch::Tensor stens, atens, sptens, sptens_tmp;
  float reward;
  bool done;
  // get item ad index 
  std::tie(stens, atens, sptens, reward, done) = rbuff->get(0);
  // get next item
  float state_diff = 0.;
  for (unsigned int i=1; i<buffer_size; ++i) {
    std::tie(stens, atens, sptens_tmp, reward, done) = rbuff->get(i);
    state_diff += torch::sum(torch::abs(stens - sptens)).item<float>();
    sptens.copy_(sptens_tmp);
  }
  std::cout << "TestTrajectoryConsistency: state-diff " << state_diff << std::endl;

  return (state_diff < 1e-7);
}

// check if nstep reward calculation is correct
bool TestNstepConsistency() {
  // some parameters
  unsigned int batch_size = 32;
  unsigned int buffer_size = 8 * batch_size;
  unsigned int nstep = 4;
  float gamma = 0.95;

  // get replay buffer
  auto rbuff = getTestReplayBuffer(buffer_size, gamma, nstep);
  
  // sample a batch
  torch::Tensor stens, atens, sptens, rtens, dtens;
  float state_diff = 0;
  float reward_diff = 0.;
  std::tie(stens, atens, sptens, rtens, dtens) = rbuff->sample(batch_size);

  // iterate over samples in batch
  torch::Tensor stemp, atemp, sptemp, sstens;
  float rtemp, reward, gamma_eff, rdiff, sdiff, sstens_val;
  bool dtemp;

  // init differences:
  rdiff = 0.;
  sdiff = 0.;
  for (int64_t s=0; s<batch_size; ++s) {
    sstens = stens.index({s, "..."});
    sstens_val = sstens.item<float>();
    
    // find the corresponding state
    for (unsigned int i=0; i<buffer_size; ++i) {
      std::tie(stemp, atemp, sptemp, rtemp, dtemp) = rbuff->get(i);
      if (std::abs(stemp.item<float>() - sstens_val) < 1e-7) {
	
	// found the right state
	gamma_eff = 1.;
	reward = rtemp;
	for(unsigned int k=1; k<nstep; k++) {
	  std::tie(stemp, atemp, sptemp, rtemp, dtemp) = rbuff->get(i+k);
	  gamma_eff *= gamma;
	  reward += rtemp * gamma_eff;
	}
	break;
      }
    }
    rdiff += std::abs(reward - rtens.index({s, "..."}).item<float>());
    sdiff += torch::sum(torch::abs(sptemp - sptens.index({s, "..."}))).item<float>();
    //std::cout << "reward = " << reward << " sp = " << sptemp.item<float>() << std::endl;
    //std::cout << "reward-reference = " << rtens.index({s, "..."}).item<float>()<< " sp-reference = " << sptens.index({s, "..."}).item<float>() << std::endl;
    //std::cout << "rdiff = " << rdiff << " sdiff = " << sdiff << std::endl;
  }
  // make sure values are consistent:
  std::cout << "TestEntryConsistency: state-diff " << sdiff << " reward-diff " << rdiff << std::endl;
  
  return ((rdiff < 1e-7) && (sdiff < 1e-7));
}


int main(int argc, char* argv[]){

  TestEntryConsistency();
  
  TestTrajectoryConsistency();

  TestNstepConsistency();
}
