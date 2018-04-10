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
    run_type = "largescale_with_reconstruction"
    batchfile = batchfileTemplate.format(run_type)
    batchfilename = "CapsNet-{}-job.sh".format(run_type)
    # write this to a file first
    with open(batchfilename, "w+") as f:
        f.write(batchfile)

    # submit batch job
    subprocess.run(["sbatch", batchfilename])
    # erase file
    subprocess.run(["rm", "-rf", batchfilename])

def recompile():
    my_env = os.environ.copy()
    my_env["CUDA_BIN_PATH"] = "/usr/local/cuda-9.0"
    my_env["PATH"] = "/usr/local/cuda-9.0/bin:" + my_env["PATH"]

    subprocess.run(["rm", "-rf", cwd_directory])
    subprocess.run(["mkdir", cwd_directory])
    subprocess.run(["cmake", ".."], cwd=cwd_directory, env=my_env)

    # subprocess.run(["sleep", "5s"])
    # subprocess.run(["make"], cwd=cwd_directory, env=my_env)
    # subprocess.run(["cmake", "--build", cwd_directory, "--target", "NeuralNets", "--", "-j", "4"])

if __name__ == '__main__':
    # recompile()
    slurm_run()
