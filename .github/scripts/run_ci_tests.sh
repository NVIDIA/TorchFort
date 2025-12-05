#!/bin/bash
set -euxo pipefail

cd /opt/torchfort/bin/tests/general
python scripts/setup_tests.py
./test_losses

cd /opt/torchfort/bin/tests/supervised
python scripts/setup_tests.py
./test_checkpoint
./test_training

cd /opt/torchfort/bin/tests/rl
./test_distributions
./test_replay_buffer
./test_rollout_buffer
./test_off_policy --gtest_filter=*L0*
./test_on_policy --gtest_filter=*L0*
