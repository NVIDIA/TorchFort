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

import argparse as ap
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import time

def main(args):

    global infiles, labelfiles, outfiles, artists
    print(f"processing files in {args.input_path}...")

    infiles = sorted(glob.glob(os.path.join(args.input_path, "input_0*")))
    labelfiles = sorted(glob.glob(os.path.join(args.input_path, "label_0*")))
    outfiles = sorted(glob.glob(os.path.join(args.input_path, "output_0*")))

    artists = []

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_title(r"$u$")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax2.set_title(r"$\nabla \cdot \mathbf{a}u$ (true)")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax3.set_title(r"$\nabla \cdot \mathbf{a}u$ (prediction)")
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$y$")
    ax4.set_title(r"1D sample along dotted line")
    ax4.set_xlabel(r"$x$")

    idata = np.loadtxt(infiles[0])
    ldata = np.loadtxt(labelfiles[0])
    odata = np.loadtxt(outfiles[0])

    c = ax1.contourf(idata)
    try:
      artists += c.collections
    except:
      artists.append(c)
    c = ax1.hlines(idata.shape[0]//2 + 1, 0, idata.shape[1]-1, colors="black", linestyles="dashed")
    artists.append(c)
    c = ax2.contourf(ldata)
    try:
      artists += c.collections
    except:
      artists.append(c)
    c = ax3.contourf(odata)
    try:
      artists += c.collections
    except:
      artists.append(c)
    c, = ax4.plot(idata[idata.shape[0]//2 + 1,:], 'k')
    artists.append(c)
    c, = ax4.plot(ldata[idata.shape[0]//2 + 1,:], 'b')
    artists.append(c)
    c, = ax4.plot(odata[idata.shape[0]//2 + 1,:], 'g.')
    artists.append(c)

    fig.tight_layout()

    def animate(i):
        global infiles, labelfiles, outfiles, artists
        for c in artists:
            c.remove()
        artists.clear()

        idata = np.loadtxt(infiles[i])
        ldata = np.loadtxt(labelfiles[i])
        odata = np.loadtxt(outfiles[i])
        c = ax1.contourf(idata)
        try:
          artists += c.collections
        except:
          artists.append(c)
        c = ax1.hlines(idata.shape[0]//2 + 1, 0, idata.shape[1]-1, colors="black", linestyles="dashed")
        artists.append(c)
        c = ax2.contourf(ldata)
        try:
          artists += c.collections
        except:
          artists.append(c)
        c = ax3.contourf(odata)
        try:
          artists += c.collections
        except:
          artists.append(c)
        c, = ax4.plot(idata[idata.shape[0]//2 + 1,:], 'k')
        artists.append(c)
        c, = ax4.plot(ldata[idata.shape[0]//2 + 1,:], 'b')
        artists.append(c)
        c, = ax4.plot(odata[idata.shape[0]//2 + 1,:], 'g.')
        artists.append(c)

    ani = FuncAnimation(fig, animate, frames=len(infiles), repeat=False, interval=1)

    os.makedirs(args.output_path, exist_ok=True)

    def log(i, n):
        print(f"processed {i+1} of {n} frames..." )
    ani.save(os.path.join(args.output_path, "validation_results.gif"), writer=PillowWriter(fps=5), progress_callback=lambda i, n: log(i,n))
    print(f"video written to {os.path.join(args.output_path, 'validation_results.gif')}...")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Directory containing validation text files", required=True)
    parser.add_argument("--output_path", type=str, help="Directory to store the generated videos", required=True)
    args = parser.parse_args()

    main(args)

