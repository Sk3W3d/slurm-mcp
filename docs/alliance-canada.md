# Alliance Canada / Compute Canada Notes

This MCP is intended to operate through an existing 2FA-backed SSH session on
Alliance Canada / Compute Canada Slurm clusters. Agents should treat the login
node as a control plane for lightweight commands only.

## Software Environments

Do not assume Conda or Docker are available on these clusters. The default
workflow should use the module system plus a Python virtual environment:

```bash
module load python
python -m venv $VENV_PATH
source $VENV_PATH/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Job scripts should load required modules and activate the virtual environment
inside the Slurm allocation before launching Python code.

## Login Node Policy

Do not run training, evaluation, preprocessing, or other heavy workloads on the
login node. Use the login node for commands such as `sbatch`, `squeue`, `sacct`,
small file inspection, and lightweight Git/status checks.

## Storage

Prefer scratch or project storage for repositories, run directories, logs,
checkpoints, datasets, and temporary outputs. Avoid assuming that `$HOME` has
enough quota for experiment artifacts.

Use `$SLURM_TMPDIR` only for per-job temporary files that can be recreated.

## Job Scripts

Slurm scripts should make the runtime environment explicit:

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load python
source /path/to/venv/bin/activate
cd /path/to/repo

python train.py
```

Cluster-specific GPU, account, partition, and QOS flags should come from the
configured cluster profile rather than from generic assumptions.
