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

With this MCP, use `setup_venv` instead of running these commands by hand. It
installs with `--no-index` from the Alliance wheelhouse, which has prebuilt
wheels for torch, numpy, transformers, and many other packages. Use
`check_wheels` to see which versions exist. Some packages need an extra module
loaded first (for example `arrow` for pyarrow, `opencv` for cv2). Pass those
as `extra_modules`. Use `no_index=False` only for packages that are not in the
wheelhouse.

## Compute Nodes Are Usually Offline

On most Alliance clusters, compute nodes cannot reach the internet. Anything a
job needs from the network (Hugging Face models, datasets, pip packages) has
to be fetched on a login node before the job runs:

1. `hf_download(repo_id)` downloads into the shared `HF_HOME` from a login
   node. Gated models need `hf_store_token` once per cluster.
2. `submit_job` exports `HF_HOME` and sets `HF_HUB_OFFLINE=1`,
   `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`, so
   `from_pretrained("org/name")` loads from the cache.
3. If a job fails with a connection error or "couldn't find it in the cached
   files", the model, revision, or file pattern is missing from the cache.
   Check with `hf_cache_list`.

Login nodes are shared. Downloads and pip installs are fine there; training,
tokenizing large datasets, and long compiles are not.

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
