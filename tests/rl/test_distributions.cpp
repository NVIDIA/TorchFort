#include <gtest/gtest.h>
#include <torch/torch.h>
#include "internal/rl/distributions.h"

using namespace torchfort;
using namespace torch::indexing;

TEST(NormalDistribution, RandomSampling) {
  // rng
  torch::manual_seed(666);

  // no grad guard
  torch::NoGradGuard no_grad;
  
  // create normal distribution with given shape
  torch::Tensor mutens = torch::empty({4,8}, torch::kFloat32);
  torch::Tensor log_sigmatens = torch::empty({4,8}, torch::kFloat32);

  // fill with random elements
  mutens.normal_();
  log_sigmatens.normal_();
  torch::Tensor sigmatens = torch::exp(log_sigmatens);
  
  auto ndist = rl::NormalDistribution(mutens, sigmatens);
  torch::Tensor sample = ndist.rsample();

  // do direct sampling without reparametrization trick
  torch::Tensor sample_compare = at::normal(mutens, sigmatens);

  // expect that shapes match: I am not sure how to compare the values as well
  EXPECT_NO_THROW(torch::sum(sample-sample_compare).item<float>());
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  
  return RUN_ALL_TESTS();
}
