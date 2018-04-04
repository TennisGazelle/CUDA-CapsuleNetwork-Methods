#!/usr/bin/python3
import subprocess

batchfileTemplate = "#!/bin/bash\n" \
                    "#SBATCH --output={0}.log\n" \
                    "#SBATCH --time=24:00:00\n" \
                    "srun ../cmake-build-debug/NeuralNets \n"

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

if __name__ == '__main__':
    slurm_run()
