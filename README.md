# Alliance Slurm MCP

Local MCP bridge for safe, narrow access to Alliance Canada Slurm clusters.

This first version is intentionally read-only. It checks whether an existing
2FA-backed SSH control connection can be reused, then exposes harmless Slurm
queries such as `squeue` and `sacct`.

## 1. Configure SSH multiplexing

Add entries like this to `~/.ssh/config`:

```sshconfig
Host killarney
  HostName killarney.alliancecan.ca
  User your_cc_username
  ControlMaster auto
  ControlPath ~/.ssh/cm-%C
  ControlPersist 8h
  ServerAliveInterval 60
  ServerAliveCountMax 3
  ForwardAgent no
```

Repeat for `narval`, `trillium`, and `vulcan` as needed, using the host names
recommended by Alliance documentation or your institute.

On Windows, native OpenSSH may not support `ControlMaster`/`ControlPersist`
reliably. The most robust setup is to run the master SSH session in WSL. If the
MCP server itself runs under Windows Python, set this in `config.yaml`:

```yaml
ssh_command:
  - wsl
  - ssh
```

If the MCP server runs inside WSL too, leave `ssh_command` as:

```yaml
ssh_command:
  - ssh
```

## 2. Start a 2FA-backed master SSH session

Run this manually in a terminal:

```bash
ssh -MNf killarney
```

You should receive the usual 2FA prompt. After approval, SSH stays open in the
background as a master connection.

Check it with:

```bash
ssh -O check killarney
```

Then verify non-interactive reuse:

```bash
ssh -o BatchMode=yes killarney whoami
ssh -o BatchMode=yes killarney squeue -u your_cc_username
```

If those commands work without a new 2FA prompt, the MCP server should be able
to reuse the session.

## 3. Configure this MCP server

Copy the example config:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` so each cluster `host` matches an SSH `Host` entry.

Use `cluster_defaults` for shared Alliance Canada / Compute Canada policies,
setup notes, and documentation sources. Individual clusters can add or override
cluster-specific details such as docs, storage paths, accounts, partitions, or
GPU request style.

Operational notes that are useful to both humans and agents can live in local
markdown files such as `docs/alliance-canada.md`, then be referenced from
`cluster_defaults.docs` or a cluster's `docs` list.

## 4. Run locally

Install in editable mode:

```bash
python -m pip install -e .
```

Run the MCP server:

```bash
alliance-slurm-mcp --config config.yaml
```

For Codex or Claude Code, configure the command as the local MCP server command
and pass the same `--config config.yaml` argument.

## Current tools

- `list_clusters()`
- `cluster_info(cluster=None)`
- `cluster_docs(cluster=None)`
- `cluster_auth_status(cluster=None)`
- `remote_whoami(cluster=None)`
- `list_job_profiles(cluster=None)`
- `list_jobs(cluster=None, states=None)`
- `job_status(job_id, cluster=None)`
- `submit_job(profile, script, cluster=None, job_name=None, script_args=None, dry_run=False)`
- `list_run_files(cluster=None, path=None, limit=100, max_depth=4)`
- `read_file(path, cluster=None, max_bytes=20000, base="run_root")`
- `tail_file(path, cluster=None, lines=100, base="run_root")`
- `grep_files(pattern, path=None, cluster=None, limit=50, base="run_root")`
- `job_logs(job_id, cluster=None, limit=20)`
- `tail_job_log(job_id, stream="stdout", cluster=None, lines=100)`

The server intentionally does not expose an arbitrary remote shell.

## Job API shape

Slurm tools return structured dictionaries/lists rather than terminal tables.
`list_jobs` parses `squeue`, while `job_status` combines live `squeue` state
with `sacct` accounting records when available.

Job submission is profile-based. The `script` argument must be a relative path
inside the configured `storage.remote_repo`, such as `scripts/train.slurm`.
Profiles can define common `sbatch` options such as:

- `time`
- `mem`
- `cpus_per_task`
- `gpus`
- `gpu_type`
- `nodes`
- `ntasks`
- `partition`
- `account`
- `qos`
- `constraint`
- `output`
- `error`
- `sbatch_options`

Use `dry_run=True` to preview the remote repository, log directory, and exact
`sbatch` argv without contacting the cluster or submitting a job.

## Log and output inspection

Remote file tools are constrained to the configured `storage.remote_repo` and
`storage.run_root` roots. Relative file paths are resolved under `run_root` by
default. Pass `base="remote_repo"` to inspect repository files.

For example:

```python
list_run_files(limit=20)
tail_job_log("123456", stream="stderr", lines=100)
read_file("logs/train-123456.out", max_bytes=20000)
grep_files("Traceback", path="logs", limit=20)
```

Absolute remote paths are accepted only when they are inside one of the
configured roots.

## Audit log

Remote actions are logged locally as JSONL. By default, the log is written to
`.slurm-mcp-audit.jsonl` next to the config file. Set `audit_log: null` in
`config.yaml` to disable audit logging.
