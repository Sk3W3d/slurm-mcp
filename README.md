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
| `run_root` | job logs (`run_root/logs`), outputs, and the default folder for file tools |
| `venv_root` | virtualenvs made by `setup_venv` |
| `hf_home` | shared Hugging Face cache that jobs read through `HF_HOME` |
| `tasks_root` | background task folders (optional; defaults to `run_root/mcp-tasks`) |

Values in `cluster_defaults` apply to every cluster, and each cluster can
override them key by key. `python.module` picks the Lmod Python module (for
example `python/3.11`). `slurm.default_account` fills in `--account`.
`slurm.cpu_account` replaces it for jobs that request no GPUs, for example
when `default_account` is a GPU-only RAC allocation.
`policies.compute_node_internet: false` makes jobs run with the Hugging Face
offline flags.

### Projects

Storage values are templates, so one config serves many projects:

```yaml
cluster_defaults:
  python:
    venv_name: "{project}"                     # one venv per project
  storage:
    workspace_root: /project/def-yourpi/{user}/code
    remote_repo: "{workspace_root}/{project}"
    run_root: /scratch/{user}/runs/{project}
    venv_root: /home/{user}/venvs
    hf_home: /project/def-yourpi/{user}/hf     # shared by all projects
    tasks_root: /scratch/{user}/.slurm-mcp/tasks
```

`{user}` and `{cluster}` come from the cluster entry, `{project}` from the
selected project, and any other name refers to another storage key. Every
tool that touches project paths takes `project=`. Without it, tools use the
project picked with `project_use`, or `default_project` from the config.

- A new project needs no config edit. Pick a name, then `remote_git_clone` or
  `sync_local_commits` creates its folder.
- Paths without `{project}` (here `hf_home`, `venv_root`, `tasks_root`) are
  shared, so tools that only use them, such as `hf_download` and
  `remote_task_status`, work without a project.
- Keep `run_root` outside `remote_repo`. Untracked log files inside the repo
  make `remote_git_pull` and `remote_git_checkout` refuse to run.
- A project whose checkout doesn't follow the template can override paths
  under `projects:`, for all clusters or one:

  ```yaml
  projects:
    old-paper:
      description: Camera-ready runs.
      clusters:
        narval:
          storage:
            remote_repo: /project/def-yourpi/{user}/old-paper
  ```

- Passing `project=` to a cluster whose storage has no `{project}` is an
  error, so a call never lands in the wrong project's folder. The active
  project is ignored on such clusters.

### Project record

The server keeps a local SQLite record (`state_db`, by default
`.slurm-mcp-state.sqlite3` next to the config file) of what it does for each
project. Any MCP client that uses the same config shares it, so a Codex
session sees the jobs a Claude Code session submitted. The tools fill it in
as a side effect:

| tool | records |
|---|---|
| `submit_job` | job id, profile, script, the repo commit (and whether it had uncommitted changes), and an optional `purpose` |
| `sync_local_commits`, `remote_git_clone`, `remote_git_pull`, `remote_git_checkout` | the commit and branch on the cluster |
| `setup_venv`, `hf_download` | the background task, and the project's venv |
| `job_status`, `list_jobs`, `job_history`, `remote_task_status` | the latest state of recorded jobs and tasks |

It stores facts and caches state. A job's state comes from Slurm and is
saved with the time it was checked (`state_checked_at`), so stale state is
visible as stale. Read it with:

- `project_status`: where the project lives on each cluster, its unfinished
  and recent jobs, background tasks, and latest events. On clusters you're
  connected to, it first refreshes unfinished jobs with one `sacct` call and
  checks unfinished tasks.
- `project_history`: the timeline, e.g. `job_submitted`, `job_finished`,
  `repo_synced`, `task_started` and `task_finished`.
- `project_list`: also lists recorded projects, most recently active first.

Jobs are linked to their project, so `job_logs` and `tail_job_log` find a
recorded job's logs without `project=`. A failed write to the record never
fails the tool. The tool still runs and its result has a `ledger_error` field.
Set `state_db: null` to turn the record off.

**Notes.** The automatic record holds facts. `project_note` adds what only
the agent knows: a result ("val acc 71.2% at epoch 12"), a decision and its
reason, a problem, or the next step. The category is one of `progress`,
`result`, `decision`, `issue` or `next_step`. A note can link to job IDs and
records which MCP client wrote it. `project_status` shows the latest notes.

**Agent instructions.** The server publishes MCP instructions that every
client receives when it connects. They tell the agent to call
`project_use` and `project_status` before working on a project, to pass
`purpose=` when submitting, and to leave `project_note`s after meaningful
steps. The text is `SERVER_INSTRUCTIONS` in `slurm_mcp/server.py`.

**Cluster notes.** Knowledge about the clusters themselves goes in
cluster notes, not project notes: which clusters to prefer and why, hardware,
queue experience, account quirks and quotas.

- **Adding notes.** `cluster_note(text, cluster=None, category=...)` adds one.
  Leave `cluster` unset for general notes such as a preference order. The
  category is one of `preference`, `hardware`, `queue`, `account`, `storage`,
  `software`, `issue` or `tip`. Notes go stale, so
  `cluster_note(..., replaces=<note_id>)` updates one and
  `cluster_note_remove` retires it.
- **Reading notes.** `cluster_notes()` lists them. They also appear in
  `cluster_info`, `project_status` (for the clusters the project uses) and
  `cluster_overview`. The server instructions tell agents to read them before
  choosing a cluster.
- **Jobs per cluster.** `cluster_overview()` is the per-cluster view across
  all projects: each cluster's notes, whether it's connected, and the
  unfinished and recent jobs the server submitted there, each with its
  project and purpose. Job records themselves live in the project record.
- **Syncing.** `project_record_sync` also syncs cluster notes, through
  `_cluster-notes.json` next to the project files.

**More than one machine.** The record is local, so a second machine starts
empty. `project_record_sync(project, cluster)` merges this machine's record
with a copy on the cluster at `storage.record_root/<project>.json` (default
`~/.slurm-mcp/records`), then writes the merged result back. Events are
matched by ID, and for jobs the most recently checked state wins, so
syncing from both machines loses nothing and creates no duplicates. Run it
when you start on a machine that hasn't seen the project, and before
switching machines.

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
2. `project_use("my-paper")` selects the project. `project_list(cluster="narval")`
   shows the projects already on a cluster.
3. Get code onto the cluster. Use `remote_git_clone(url)` the first time.
   After that, use `remote_git_pull(dry_run=False)` to pick up changes pushed
   to GitHub, or `sync_local_commits()` to send local commits that aren't
   pushed.
4. `setup_venv(requirements="requirements.txt")` starts a background task.
   Poll it with `remote_task_status(task_id)`.
5. `hf_download("meta-llama/Llama-3.1-8B-Instruct")` starts a background
   task that downloads on the login node into `hf_home`.
6. `job_script_template(command="python train.py")` drafts a script. Save it
   with `write_file("scripts/train.sh", ...)`.
7. `submit_job("train_gpu", "scripts/train.sh", overrides={"time": "04:00:00"}, purpose="baseline")`
   submits the job. Add `dry_run=True` first to preview the `sbatch` command.
   Later, `project_status()` shows how the project's jobs are doing.
8. Check on the job with `job_status`, `tail_job_log`, `list_run_files`,
   `read_file`, `grep_files`, and `download_file`.

## Tools

**Projects:** `project_use`, `project_list`, `project_status`,
`project_history`, `project_note`, `project_record_sync`. Path-based tools
below also take `project=`.

**Connection and cluster info:** `cluster_login`, `cluster_logout`,
`cluster_auth_status`, `cluster_preflight`, `remote_whoami`, `list_clusters`,
`cluster_info`, `cluster_docs`, `cluster_partitions`, `slurm_accounts`,
`disk_quota` (space and file-count quotas from `diskusage_report`, with
near-quota flags), `config_reload`.

**Jobs:** `list_job_profiles`, `submit_job`, `cancel_job` (your own jobs only),
`list_jobs`, `job_status` (includes Slurm's start estimate for pending jobs),
`job_history`, `job_logs`, `tail_job_log`, `job_script_template`.

**Files:** `list_run_files`, `read_file`, `tail_file`, `grep_files`,
`write_file`, `upload_file`, `download_file`. Paths must stay inside the
configured storage roots. Local paths must stay inside `local_roots`.
Transfers are capped at `max_transfer_bytes`.

**Directories and cluster-to-cluster copies:**

- **`upload_directory`, `download_directory`.** These copy a whole folder,
  packed as an uncompressed tar archive and unpacked on the other side.
  `exclude` takes tar-style patterns such as `[".git", "__pycache__",
  "*.ckpt"]`. An existing destination is refused unless `overwrite=True`,
  which merges into it. Local unpacking refuses entries that would land
  outside the destination: `..` paths, and links pointing elsewhere.
- **`copy_between_clusters`.** This copies a file or folder from one cluster
  to another. Each side has its own base and project. Both clusters must be
  logged in. Login nodes can't log in to each other without 2FA, so the data
  is relayed through this machine: packed on the source, pulled to a
  temporary folder, pushed and unpacked on the destination. Within one
  cluster it's a direct copy on the cluster.
- **Size and speed.** Everything is capped at `max_transfer_bytes` and goes
  through your own connection. For large datasets, use
  [Globus](https://docs.alliancecan.ca/wiki/Globus), which moves data between
  clusters directly.
- **Cleanup and records.** Temporary archives are removed even when a step
  fails. Each transfer is recorded in the project history as
  `files_uploaded`, `files_downloaded` or `files_copied`.

**Git:** `remote_git_status`, `remote_git_rev`, `remote_git_branch`,
`remote_git_log`, `remote_git_diff` (uncommitted changes, or between refs),
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
detached under `tasks_root/<task_id>/` (default `run_root/mcp-tasks`). That folder holds
`task.sh`, `output.log`, and `exit_code`, so status survives reconnects.

**Escape hatch:** `run_remote_command(command, reason)` runs any bash command
on a login node. It's a last resort for when no dedicated tool fits, and it's
off unless `allow_remote_shell: true`. When it's on:

- **Read-only commands run without asking.** These are Slurm queries
  (`squeue`, `sacct`, `sinfo`, `scontrol show`, `seff`, …),
  `diskusage_report`, `module avail`/`spider`, `pip list`, read-only git
  (`status`, `log`, `diff`, `show`, listing branches), and file inspection
  (`ls`, `cat`, `head`, `tail`, `grep`, `find`, `du`, …). This applies only
  when the whole command line is made of them, joined by `|`, `&&`, `||` or
  `;`, with no output redirection (other than to `/dev/null` and `2>&1`),
  `$(...)`, background `&` or `VAR=` prefixes. Arguments are checked too: for
  example, `git branch -D`, `find -delete`, `sort -o` and
  `scontrol update` still need approval. The list is `READ_ONLY_COMMANDS` in
  `slurm_mcp/shell_guard.py`. `remote_shell_readonly_commands` adds names you
  trust, and `remote_shell_auto_approve_readonly: false` turns this off.
- **You approve every other command.** The server shows the command,
  cluster, folder, the agent's reason and why approval is needed, through MCP
  elicitation (a dialog in Claude Code). If you decline or cancel, or the
  client can't prompt, it doesn't run.
- **Clearly destructive commands are always refused**, even when approved.
  That covers recursive deletion of any configured folder or anything above
  it (home, project and scratch roots included), `scancel` (use
  `cancel_job`), recursive `chmod`/`chown`, anything touching `~/.ssh`,
  `ssh`/`scp`/`rsync` to other hosts, `sudo`, `curl … | sh`, and git
  force-push. The checks live in `slurm_mcp/shell_guard.py`. They catch
  accidents, not a determined attacker, so your approval remains the real
  safeguard.
- **Every command is recorded.** The command, its reason and the approval
  go into the audit log, refused ones included, and each command that runs
  also becomes a `shell_command` event in the project record. That shows
  which needs keep coming up and should become dedicated tools.
- **`remote_shell_auto_approve: true` skips the approval prompt.** This is
  dangerous: whatever a mistaken or prompt-injected agent decides to run then
  runs under your account. It's off by default.

## Shared server: one Duo approval per cluster for every client

By default each client (every Claude Code session, Codex) starts its own copy
of the server, so each one holds its own SSH connections and needs its own
Duo push. Instead, run **one shared server** that all clients connect to over
HTTP on your machine.

```powershell
.\scripts\shared-server.ps1                 # start in the background (logs: ~/.slurm-mcp/server.err.log)
.\scripts\shared-server.ps1 -Status         # or -Stop
.\scripts\register-autostart.ps1            # optional: start it at each Windows logon
```

On first start it creates a random token in `~/.slurm-mcp/server-token`, and
every request must send it. Connect the clients:

```bash
# Claude Code (replaces the stdio registration; remove that one first)
claude mcp remove alliance-slurm
claude mcp add --scope user --transport http alliance-slurm http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer $(cat ~/.slurm-mcp/server-token)"
```

```toml
# Codex: ~/.codex/config.toml (set SLURM_MCP_TOKEN to the token, e.g. with setx)
[mcp_servers.alliance-slurm]
url = "http://127.0.0.1:8765/mcp"
bearer_token_env_var = "SLURM_MCP_TOKEN"
tool_timeout_sec = 900   # cluster_login waits up to 2 minutes for Duo
```

On **Linux or macOS**, use the shell versions:

```bash
scripts/shared-server.sh start          # also: stop | status | restart; --port N --config PATH
scripts/install-systemd-user.sh         # optional, Linux: systemd user service (--uninstall to remove)
loginctl enable-linger "$USER"          # optional: keep it running after you log out
```

On a Linux workstation:

- **Set `ssh_backend: paramiko` in that machine's config.** The default there
  is `openssh`, which can't answer Duo, so `cluster_login` couldn't send the
  push.
- **Give it its own config.** It needs Linux paths for `local_roots`, and
  `ssh <cluster>` once per cluster to record host keys.
- **Keep the machines' records in sync** with `project_record_sync`. The
  project record is per machine.

**When Duo happens.** Once per cluster, the first time any client needs it.
After that, every session and agent reuses the connection, and `/mcp` in
Claude Code only reconnects the client, without a new push. A new push is
needed only when the shared server restarts (reboot, or loading new server
code) or when a connection breaks (laptop sleep, network drop, login node
reboot). Then one client logs in again for everyone.

**What's shared:**

- **Shared by all clients:** SSH connections, the project record, cluster
  notes, and the login history.
- **Per client session:** the active project (`project_use`), and approval
  dialogs for shell commands, which appear in the session that ran the
  command.

Slow calls, like a Duo wait or a transfer, run in worker threads, so they
don't block other clients.

**Security.** The server only listens on loopback (`--host` refuses anything
else), and requests without the token get 401. Anything running as you that
can read the token file can use your open cluster connections, much as
anything running as you could use your SSH key.

## Automation nodes (optional, needs approval)

Some clusters have an **automation node** that accepts a constrained SSH key
without Duo
([Alliance docs](https://docs.alliancecan.ca/wiki/Automation_in_the_context_of_multifactor_authentication)).
Alliance grants access **by request**, and states that these nodes "are not
suitable for usage of AI agents". Ask support before turning this on.

How the server uses them:

- **When.** If the cluster's login connection is up, everything uses it.
  Otherwise, with `robot_enabled: true` and a `robot_host` for the cluster,
  tools connect to the automation node on their own, without a Duo push.
- **Which tools.** Commands that the node's wrapper (`allowed_commands.sh`)
  accepts go there: Slurm (`squeue`, `sacct`, `sbatch`, `scancel`, `sinfo`),
  git status, fetch, pull and checkout, and file reads, writes, listings and
  transfers over SFTP. The wrapper runs commands without a shell, so the
  server sends them as single commands. `submit_job` uses `--chdir` and an
  absolute script path, and file tools use SFTP instead of
  `find`/`stat`/`tail`.
- **What still needs the login node.** Anything needing a shell: the
  `run_remote_command` tool, `grep_files`, venv builds, HF downloads, directory
  transfers, `sync_local_commits`, `slurm_accounts` and preflight. These say
  so and ask for `cluster_login`. The wrapper reports rejected commands with
  exit code 0, so the server detects its rejection message instead.
- **Logging in.** `cluster_login(method=...)` takes `"login"` (Duo), `"robot"`
  (key only) or `"auto"`. With `auto`, the server tries Duo first, and if the
  push isn't approved in time and `robot_fallback_on_mfa_timeout: true`, it
  connects the automation node instead.
- **History.** `cluster_auth_status` shows both connections and the recent
  login history per method. Connection attempts are recorded as `connection`
  events. A failed automation login isn't retried for 10 minutes, since
  clusters throttle repeated logins.

Setup:

1. Create a dedicated key without a passphrase. Unattended use can't type
   one, which is why the CCDB line pins it to your IP:
   `ssh-keygen -t ed25519 -N "" -f ~/.ssh/alliance_robot_ed25519`.
2. Add it in CCDB with all three constraints, for example:
   `restrict,from="x.y.z.*",command="/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/allowed_commands.sh" ssh-ed25519 AAAA...`.
   `from=` must be your public IP with at least the first three parts fixed.
   Set `robot_ip_version` to match the address family the mask uses.
3. Record the nodes' host keys in a separate file with `ssh-keyscan -4
   <robot host> >> ~/.ssh/known_hosts_alliance_robots`. Check the
   fingerprints first.
4. Set `robot_host` per cluster. The documented nodes are:
   - `robot.fir.alliancecan.ca`, `robot.narval.alliancecan.ca`,
     `robot.nibi.alliancecan.ca` and `robot.rorqual.alliancecan.ca`;
   - `robot.tamia.ecpia.ca`;
   - `robot2.scinet.utoronto.ca` for Trillium, and
     `trig-robot1.scinet.utoronto.ca` for Trillium GPU.

   Then set `robot_enabled: true`, and optionally
   `robot_fallback_on_mfa_timeout: true`.

## Job submission details

Profiles in `config.yaml` set `sbatch` options: `time`, `mem`,
`mem_per_cpu`, `cpus_per_task`, `gpus`, `gpu_type`, `gres`, `nodes`,
`ntasks`, `ntasks_per_node`, `partition`, `account`, `qos`, `constraint`,
`array`, `dependency`, `exclusive`, `signal`, `output`, `error`,
`mail_type`, `mail_user`, `env`, `sbatch_options`. A `submit_job` call can
override any of these with `overrides`.

Jobs start with `--export=ALL,...` plus their environment variables, added
in this order (later ones win):

1. `MCP_PROJECT`, `MCP_REPO_DIR` and `MCP_RUN_DIR`, so a script can write to
   `$MCP_RUN_DIR/checkpoints` instead of hard-coding paths. The prefix avoids
   clashing with variables such as `$PROJECT` that some Alliance clusters set.
2. The Hugging Face cache variables (`HF_HOME`, plus the offline flags when
   compute nodes have no internet).
3. `slurm.job_env`.
4. The profile's `env`.
5. The call's `env`.

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
