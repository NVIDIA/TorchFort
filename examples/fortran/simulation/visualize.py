# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import h5py as h5
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

    with h5.File(infiles[0], 'r') as f:
        idata = f["data"][...]
    with h5.File(labelfiles[0], 'r') as f:
        ldata = f["data"][...]
    with h5.File(outfiles[0], 'r') as f:
        odata = f["data"][...]

    c = ax1.contourf(idata)
    artists += c.collections
    c = ax1.hlines(idata.shape[0]//2 + 1, 0, idata.shape[1]-1, colors="black", linestyles="dashed")
    artists.append(c)
    c = ax2.contourf(ldata)
    artists += c.collections
    c = ax3.contourf(odata)
    artists += c.collections
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

        with h5.File(infiles[i], 'r') as f:
            idata = f["data"][...]
        with h5.File(labelfiles[i], 'r') as f:
            ldata = f["data"][...]
        with h5.File(outfiles[i], 'r') as f:
            odata = f["data"][...]
        c = ax1.contourf(idata)
        artists += c.collections
        c = ax1.hlines(idata.shape[0]//2 + 1, 0, idata.shape[1]-1, colors="black", linestyles="dashed")
        artists.append(c)
        c = ax2.contourf(ldata)
        artists += c.collections
        c = ax3.contourf(odata)
        artists += c.collections
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
    parser.add_argument("--input_path", type=str, help="Directory containing validation hdf5 files", required=True)
    parser.add_argument("--output_path", type=str, help="Directory to store the generated videos", required=True)
    args = parser.parse_args()

    main(args)

