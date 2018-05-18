#!/usr/bin/python3
import os
import subprocess

cwd_directory = "../cmake-build-cubix"

batchfileTemplate = "#!/bin/bash\n" \
                    "#SBATCH --ntasks=1\n" \
                    "#SBATCH --output={0}.log\n" \
                    "#SBATCH --gres=gpu:1\n" \
                    "srun " + cwd_directory + "/NeuralNets \n"
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
    FNULL = open(os.devnull, 'w')

    subprocess.run(["rm", "-rf", cwd_directory])
    subprocess.run(["mkdir", cwd_directory])
    subprocess.run(["cmake", ".."], cwd=cwd_directory, env=my_env)
    print("running first make (ignore this)...")
    subprocess.run(["make"], cwd=cwd_directory, stdout=FNULL, stderr=FNULL)
    print("running actual make...")
    subprocess.run(["cmake", "--build", ".", "--target", "NeuralNets", "--", "-j", "4"], cwd=cwd_directory)


if __name__ == '__main__':
    print("fresh recomiling to {}".format(cwd_directory))
    recompile()
    print("making batchfile and submitting to slurm (sbatch)")
    slurm_run()
