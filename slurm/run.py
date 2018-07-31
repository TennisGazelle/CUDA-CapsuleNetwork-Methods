#!/usr/bin/python3
import os
import argparse
import subprocess

cwd_directory = "../cmake-build-cubix"

batchfileTemplate = "#!/bin/bash\n" \
                    "#SBATCH --ntasks=1\n" \
                    "#SBATCH --output={0}.log\n" \
                    "#SBATCH --gres=gpu:1\n" \
                    "srun " + cwd_directory + "/NeuralNets -g \n"
                    # "#SBATCH --cpus-per-task=8\n" \
                    # "#SBATCH --time=24:00:00\n" \

def slurm_run():
    run_type = "neural_net_output"
    batchfile = batchfileTemplate.format(run_type)
    batchfilename = "CapsNet-{}-job.sh".format(run_type)
    # write this to a file first
    with open(batchfilename, "w+") as f:
        f.write(batchfile)

    # submit batch job
    subprocess.run(["sbatch", batchfilename])
    # erase file
    subprocess.run(["rm", batchfilename])

def recompile():
    my_env = os.environ.copy()
    my_env["CUDA_BIN_PATH"] = "/usr/local/cuda-9.0"
    my_env["PATH"] = "/usr/local/cuda-9.0/bin:" + my_env["PATH"]
    fnull = open(os.devnull, 'w')

    subprocess.run(["rm", "-rf", cwd_directory])
    subprocess.run(["mkdir", cwd_directory])
    subprocess.run(["cmake", ".."], cwd=cwd_directory, env=my_env)
    print("running first make (ignore this; has to be done for some weird reason)...")
    subprocess.run(["make"], cwd=cwd_directory, stdout=fnull, stderr=fnull)
    print("running actual make...")
    subprocess.run(["cmake", "--build", ".", "--target", "NeuralNets", "--", "-j", "4"], cwd=cwd_directory)


if __name__ == '__main__':
    class N:
        pass

    n = N()
    parser = argparse.ArgumentParser(description='Submit this project to SLURM via sbatch.'
                                                 '(with or without recompilation)')
    parser.add_argument('--recomp', action='store_true', help='Fresh Recompile of project (takes longer)')
    args = parser.parse_args(namespace=n)

    if (n.recomp):
        print("fresh recomiling to {}".format(cwd_directory))
        recompile()
    else:
        print("making batchfile and submitting to slurm (sbatch)")
        slurm_run()
