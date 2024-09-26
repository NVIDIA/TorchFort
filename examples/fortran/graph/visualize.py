# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse as ap
import glob
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import time

def main(args):

    global reffiles, predfiles, artists, triangulation
    print(f"processing files in {args.input_path}...")

    reffiles = sorted(glob.glob(os.path.join(args.input_path, "reference_*.txt")))
    predfiles = sorted(glob.glob(os.path.join(args.input_path, "prediction_*.txt")))

    # Read mesh data
    nodes = np.loadtxt("nodes.txt", skiprows=1)
    triangles = np.loadtxt("connectivity.txt", skiprows=1)
    triangulation = tri.Triangulation(nodes[:,0], nodes[:,1], triangles)

    artists = []

    fig, ((ax1), (ax2)) = plt.subplots(2, 1)
    ax1.set_title("Ground Truth")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax2.set_title("Prediction")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")

    c = ax1.tricontourf(triangulation, np.loadtxt(reffiles[0]), levels=np.linspace(-0.1, 1.0, 15))
    artists += c.collections
    c = ax1.triplot(triangulation, linewidth=0.3, color='black')
    artists.append(c)
    c = ax2.tricontourf(triangulation, np.loadtxt(predfiles[0]), levels=np.linspace(-0.1, 1.0, 15))
    artists += c.collections
    c = ax2.triplot(triangulation, linewidth=0.3, color='black')
    artists.append(c)

    fig.tight_layout()

    def animate(i):
        global reffiles, predfiles, artists, triangulation
        for c in artists:
            try:
              c.remove()
            except:
              pass
        artists.clear()

        c = ax1.tricontourf(triangulation, np.loadtxt(reffiles[i]), levels=np.linspace(-0.1, 1.0, 15))
        artists += c.collections
        c = ax1.triplot(triangulation, linewidth=0.3, color='black')
        artists.append(c)
        c = ax2.tricontourf(triangulation, np.loadtxt(predfiles[i]), levels=np.linspace(-0.1, 1.0, 15))
        artists += c.collections
        c = ax2.triplot(triangulation, linewidth=0.3, color='black')
        artists.append(c)



    ani = FuncAnimation(fig, animate, frames=len(reffiles), repeat=False, interval=1)

    os.makedirs(args.output_path, exist_ok=True)

    def log(i, n):
        print(f"processed {i+1} of {n} frames..." )
    ani.save(os.path.join(args.output_path, "validation_results.gif"), writer=PillowWriter(fps=5), progress_callback=lambda i, n: log(i,n))
    print(f"video written to {os.path.join(args.output_path, 'validation_results.gif')}...")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Directory containing result text files", required=True)
    parser.add_argument("--output_path", type=str, help="Directory to store the generated videos", required=True)
    args = parser.parse_args()

    main(args)

