# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse as ap
import math
import numpy as np
from tqdm import tqdm
import torch
from functools import partial
from torch import nn
import torch.nn.functional as F

from models import PolicyFunc

from PyEnvironments import CartPoleEnv

# rendering stuff
import pygame
from pygame import gfxdraw
from moviepy.editor import ImageSequenceClip

# the implementation of the cartpole renderer was taken from
# the OpenAI gym repo: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
class Renderer(object):

    def __init__(self, x_threshold=5, length=0.5):
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.x_threshold = int(x_threshold)
        self.length = length
        
    def render(self, state):

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

def main(args):

    # set seed
    torch.manual_seed(666)

    # CUDA check
    if torch.cuda.is_available():
        torch.cuda.manual_seed(666)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # parameters
    batch_size = 1

    # policy model
    pmodel = torch.jit.load(args.policy_checkpoint)

    # env
    cenv = CartPoleEnv()

    # renderer
    renderer = Renderer()

    # reset and get initial state
    state = cenv.reset()

    frames = []
    for step in tqdm(range(args.num_steps)):

        stens = torch.Tensor(state).unsqueeze(0).to(device)
        atens = pmodel(stens)
        action = atens.item()

        # render state
        imarray = renderer.render(state)
        frames.append(imarray)
        
        # take step
        state_new, reward, terminate = cenv.step(action)

        if terminate:
            break

        state = state_new

    # print number of steps
    print(f"Episode finished with {step} steps")
        
    video = ImageSequenceClip(frames, fps=50)
    video.write_gif(os.path.join(args.output_path, "cartpole.gif"))
    video.write_videofile(os.path.join(args.output_path, "cartpole.mp4"))
    
    
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--policy_checkpoint", type=str, help="Checkpoint for policy to restore", required=True)
    parser.add_argument("--output_path", type=str, help="Directory where to store the generated videos", required=True)
    parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run")
    args = parser.parse_args()

    main(args)
