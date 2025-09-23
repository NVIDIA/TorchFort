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

import sys
import os
import signal
import re
from glob import iglob
import argparse as ap
import subprocess as sp
import wandb
import ruamel.yaml as ry
import time

def signal_handler(sig, frame):
    print("Terminating wandb watcher")
    wandb.finish()
    sys.exit(0)


def spin_wait(cond_func, num_retries, timeout):
    is_ok = True
    for _ in range(num_retries):
        if cond_func():
            break
        else:
            time.sleep(int(timeout / float(num_retries)))
    else:
        is_ok = True
    return is_ok


def find_config(logdir, config_table, model_name, parser):
    # check for all configs we have not visited yet
    config_list = [x for x in iglob(os.path.join(logdir, "*.yaml")) if x not in config_table.values()]
    
    # check the config files for the right config
    config = None
    config_file = None
    for config_file in config_list:
        with open(config_file, 'r') as f:
            model_config = parser.load(f)
            if model_config["identifier"] == model_name:
                del model_config["identifier"]
                config = model_config
                break
            else:
                continue

    return config, config_file
    

def main(args):

    # global parameters
    num_retries = 10
    
    # get logging directory:
    logging_dir = os.getenv("TORCHFORT_LOGDIR", "")
    if not logging_dir:
        raise IOError(f"TORCHFORT_LOGDIR is unset.")
    is_ok = spin_wait(lambda: os.path.isdir(logging_dir), num_retries, args.timeout)
    if not is_ok:
        raise IOError(f"logging directory {logging_dir} does not exist.")
    
    # we need the parser in order to read torchfort configs
    yaml = ry.YAML(typ="safe", pure=True)

    # if logfile path not specified, exit here:
    num_retries = 10
    logfile_exists = True
    errmsg = None
    logfilename = os.path.join(logging_dir, "torchfort.log")
    # the file might not have been created yet, so we retry a couple of times
    is_ok = spin_wait(lambda: os.path.isfile(logfilename), num_retries, args.timeout)
    if not is_ok:
        raise IOError(f"logfile {logfilename} not found. Check your paths or try to increase the timeout variable.")

    print(f"Watching file {logfilename} for updates.", flush=True)
    
    # init wandb
    wandb.init(job_type="train",
               dir=args.wandb_dir,
               project=args.wandb_project,
               group=args.wandb_group,
               entity=args.wandb_entity,
               name=args.run_tag)

    # install signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # regex pattern we are trying to match
    pattern = re.compile(r"^TORCHFORT::WANDB: model: (.*?), step: (\d{1,}), (.*?): (.*)")

    # parse the logfile
    # we keep track of new models popping up:
    config_table = {}
    with open(logfilename) as logfile:
        # get the start time, needed for computing timeouts
        start_time = time.time()

        # store the position of the parser
        last_pos = logfile.tell()
        
        # infinite parsing loop
        while True:
            # read the line
            line = logfile.readline()
            
            # check if line is empty:
            if line:

                # check if line was only partially read,
                # rewind in that case
                if not (os.linesep in line):
                    logfile.seek(last_pos)
                    continue

                # preprocess the line
                line = line.replace(os.linesep, "").strip()
                start_time = time.time()
                lmatch = pattern.match(line)

                # make this fail safe, in case there are some
                # race conditions
                try:
                    if lmatch is not None:
                        # log to wandb
                        mgroups = lmatch.groups()
                        modelname = mgroups[0].strip()
                        step = int(mgroups[1].strip())
                        metric_name = mgroups[2].strip()
                        value = float(mgroups[3].strip())
                        wandb.log({f"{modelname} {metric_name}": value, f"{modelname} {metric_name} step": step})

                        # remember the current position in the file
                        last_pos = logfile.tell()

                        # if the model string was not observed yet, find the new config:
                        if modelname not in config_table.keys():
                            if args.verbose:
                                print(f"New model {modelname} found in logfile, looking up config")
                            config, filename = find_config(logging_dir, config_table, modelname, yaml)
                            if config is not None:
                                if args.verbose:
                                    print(f"Config file for model {modelname} found, logging to wandb")
                                config_table[modelname] = filename
                                wandb.config.update({modelname: config})
                            else:
                                if args.verbose:
                                    print(f"No configuration for model {modelname} found yet")
                                
                except:
                    continue
            else:
                end_time = time.time()
                if (end_time - start_time) > args.timeout:
                    print("Timeout reached, quitting wandb watcher process.", flush=True)
                    wandb.finish()
                    return 
                if args.polling_interval > 0:
                    time.sleep(args.polling_interval)
        
               
if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument("--wandb_dir", type=str, default=None, help="Directory where wandb stores its intermediates.")
    parser.add_argument("--wandb_group", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--run_tag", type=str)
    parser.add_argument("--polling_interval", type=int, default=5, help="Polling interval in seconds with which the file will be polled for updates.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds.")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    
    # launch main app
    main(args)
