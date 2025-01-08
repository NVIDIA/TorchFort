from jinja2 import Environment, FileSystemLoader
import os

template_name = 'script_template.j2'

env = Environment(loader=FileSystemLoader('./'))
template = env.get_template(template_name)

def render_script(context, name, log, bin, exec, def_arg, nodes, types, sizes, opt_args):
    context["name"] = name
    context["log_path"] = log
    context["bin_path"] = bin
    context["exec"] = exec
    context["def_arg"] = def_arg

    for node in nodes:
        context['nodes'] = node

        for type, opt_arg in zip(types, opt_args):
            context['type'] = type
            context['opt_arg'] = opt_arg
            context['sizes'] = sizes[0] if node == 1 else sizes[1]

            with open(f"scripts/{name}_{type}_{node}.sh", "w") as f:
                f.write(template.render(context))

    print(f"{name} scripts rendered successfully!")

# Define rendering context
context = {
    "project_name": "SEEr-Polaris",
    "email": "rstrin4@uic.edu",
    "name": None,
    "type": None,
    "nodes": None,
    "sizes": None,
    "libs": ['/soft/libraries/libtorch/libtorch-2.4.0+cu124/lib', '$HOME/TorchFort-def/lib'],
    "log_path": None,
    "bin_path": None,
    "exec_times": 5,
    "exec": None,
    "def_arg": None,
    "opt_arg": None
}

if 'scripts' not in os.listdir():
    os.mkdir('scripts')

nodes = [1, 2, 4, 8]
types = ['noUM']

sizes = [[16, 32, 64], [32]]
opt_args = ['']

render_script(context, 'train_dis', '$HOME/TorchFort/train_dis_log', '$HOME/TorchFort-def/bin/examples/fortran/simulation', './train_distributed', '--size $size --batch $size', nodes, types, sizes, opt_args)

types = ['UM', 'UMT']
opt_args = ['', '--tuning']

render_script(context, 'train_dis_um', '$HOME/TorchFort/train_dis_log', '$HOME/TorchFort-def/bin/examples/fortran/simulation', './train_distributed_um', '--size $size --batch $size', nodes, types, sizes, opt_args)
