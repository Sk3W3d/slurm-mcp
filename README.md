# Alliance Slurm MCP

A local MCP server that lets Claude Code (or another MCP client) work on
Alliance Canada clusters: connect through your 2FA-protected SSH login,
sync code, build virtualenvs, cache Hugging Face models for offline
compute nodes, submit and cancel Slurm jobs, and read results.

It does not expose a general remote shell by default. Each tool runs a
fixed, validated command, file access is limited to the directories you
configure, and every remote action is written to a local audit log.

## 1. SSH access

Every cluster needs a `Host` entry in `~/.ssh/config`, and your public key
must be registered in CCDB
(<https://ccdb.alliancecan.ca/ssh_authorized_keys>):

```sshconfig
Host narval
  HostName narval.alliancecan.ca
  User your_cc_username
```

Connect once from a terminal (`ssh narval`) so the host key lands in
`~/.ssh/known_hosts`. The server refuses hosts it cannot verify.

The server has two ways to reuse your Duo-approved login (`ssh_backend` in
`config.yaml`):

- **paramiko** (default on Windows). The server holds the SSH connection
  itself. Claude calls `cluster_login`, you approve the Duo push on your
  phone, and later calls reuse that connection until the server stops. Keys
  are taken from ssh-agent or from `~/.ssh/id_*`. Encrypted keys must be
  loaded into the agent.
- **openssh** (default on Linux, macOS and WSL). Reuses an OpenSSH
  ControlMaster session that you start yourself:

  ```sshconfig
  Host narval
    HostName narval.alliancecan.ca
    User your_cc_username
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 8h
  ```

  Then run `ssh -MNf narval`, approve Duo, and check it with `ssh -O check narval`.

## 2. Install and configure

```bash
python -m pip install -e ".[dev]"
cp config.example.yaml config.yaml   # then edit it
```

For each cluster, `config.yaml` sets the following under `storage`:

| key | used for |
|---|---|
| `remote_repo` | your code on the cluster (Git tools, job scripts, `submit_job` working directory) |
| `run_root` | logs (`run_root/logs`), outputs, and background task folders (`run_root/mcp-tasks`) |
| `venv_root` | virtualenvs made by `setup_venv` |
| `hf_home` | shared Hugging Face cache that jobs read through `HF_HOME` |

Values in `cluster_defaults` apply to every cluster, and each cluster can
override them. `python.module` picks the Lmod Python module (for example
`python/3.11`). `slurm.default_account` fills in `--account`.
`policies.compute_node_internet: false` makes jobs run with the Hugging Face
offline flags.

## 3. Register with Claude Code

```bash
claude mcp add alliance-slurm --scope user \
  -- D:/Projects/slurm-mcp/.venv/Scripts/alliance-slurm-mcp.exe \
  --config D:/Projects/slurm-mcp/config.yaml
```

Add `-e HF_TOKEN=hf_...` if you will need gated models (see `hf_store_token`).
Local file tools (`upload_file`, `download_file`, `sync_local_commits`) may
only touch paths under `local_roots`. When `local_roots` is not set, that is
the directory Claude Code was started in.

## Typical workflow

1. `cluster_login(cluster="narval")`: approve the Duo push. Then run
   `cluster_preflight()`.
2. Get code onto the cluster. Use `remote_git_clone(url)` the first time.
   After that, use `remote_git_pull(dry_run=False)` to pick up changes pushed
   to GitHub, or `sync_local_commits()` to send local commits that aren't
   pushed.
3. `setup_venv(requirements="requirements.txt")` starts a background task.
   Poll it with `remote_task_status(task_id)`.
4. `hf_download("meta-llama/Llama-3.1-8B-Instruct")` starts a background
   task that downloads on the login node into `hf_home`.
5. `job_script_template(command="python train.py")` drafts a script. Save it
   with `write_file("scripts/train.sh", ...)`.
6. `submit_job("train_gpu", "scripts/train.sh", overrides={"time": "04:00:00"})`
   submits the job. Add `dry_run=True` first to preview the `sbatch` command.
7. Check on the job with `job_status`, `tail_job_log`, `list_run_files`,
   `read_file`, `grep_files`, and `download_file`.

## Tools

**Connection and cluster info:** `cluster_login`, `cluster_logout`,
`cluster_auth_status`, `cluster_preflight`, `remote_whoami`, `list_clusters`,
`cluster_info`, `cluster_docs`, `cluster_partitions`, `slurm_accounts`.

**Jobs:** `list_job_profiles`, `submit_job`, `cancel_job` (your own jobs only),
`list_jobs`, `job_status`, `job_history`, `job_logs`, `tail_job_log`,
`job_script_template`.

**Files:** `list_run_files`, `read_file`, `tail_file`, `grep_files`,
`write_file`, `upload_file`, `download_file`. Paths must stay inside the
configured storage roots. Local paths must stay inside `local_roots`.
Transfers are capped at `max_transfer_bytes`.

**Git:** `remote_git_status`, `remote_git_rev`, `remote_git_branch`,
`remote_git_fetch`, `remote_git_pull` (fast-forward only, dry run by default),
`remote_git_checkout`, `remote_git_clone`, `sync_local_commits`. Tools that
change files on the cluster refuse to run when the remote repository has
uncommitted changes. `sync_local_commits` refuses to overwrite remote
history that has diverged from yours.

**Python environments:** `setup_venv`, `venv_info`, `check_wheels`
(`avail_wheels`). Builds follow the Alliance pattern: `module load python`,
`virtualenv --no-download`, `pip install --no-index` from the wheelhouse.
Pass `no_index=False` for packages that are only on PyPI.

**Hugging Face:** `hf_download`, `hf_cache_list`, `hf_store_token`.

**Background tasks:** `remote_task_status`, `list_remote_tasks`,
`cancel_remote_task`. Long login-node work (venv builds, downloads) runs
detached under `run_root/mcp-tasks/<task_id>/`. That folder holds
`task.sh`, `output.log`, and `exit_code`, so status survives reconnects.

**Escape hatch:** `run_remote_command` runs any bash command, but only when
`allow_remote_shell: true` is set.

## Job submission details

Profiles in `config.yaml` set `sbatch` options: `time`, `mem`,
`mem_per_cpu`, `cpus_per_task`, `gpus`, `gpu_type`, `gres`, `nodes`,
`ntasks`, `ntasks_per_node`, `partition`, `account`, `qos`, `constraint`,
`array`, `dependency`, `exclusive`, `signal`, `output`, `error`,
`mail_type`, `mail_user`, `env`, `sbatch_options`. A `submit_job` call can
override any of these with `overrides`.

Jobs start with `--export=ALL,...` plus their environment variables, added
in this order:

1. The Hugging Face cache variables (`HF_HOME`, plus the offline flags when
   compute nodes have no internet).
2. `slurm.job_env`.
3. The profile's `env`.
4. The call's `env`.

Logs go to `run_root/logs/%x-%j.out` and `%x-%j.err` unless the profile sets
`output` or `error`.

## Audit log

Remote actions are appended to `.slurm-mcp-audit.jsonl` next to the config
file. Each entry records the command, return code, and the start of the
output. Data sent over stdin, such as file contents and the HF token, is
not logged. Set `audit_log: null` to turn the log off.

## Development

```bash
python -m pytest
```

`tests/test_server.py` runs the tools against local bash in place of SSH, so
the quoting, file transfers, background tasks, and git-bundle sync are all
exercised for real. On Windows this requires Git Bash.
