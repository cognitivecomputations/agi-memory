<!--
title: Execution Backends
summary: Run the existing execution tools locally, over exact SSH, or in ephemeral remote containers
read_when:
  - "You want Hexis to run work on another machine"
  - "You need to configure SSH or remote Docker execution"
  - "An execution tool ran on the wrong machine or cannot connect"
section: operations
-->

# Execution Backends

Hexis exposes one stable set of execution tools: `shell`, `safe_shell`,
`run_script`, and `execute_code`. The active execution profile decides where
they run. `local` is the default and preserves the existing worker behavior.
An operator can explicitly add and select either:

- `ssh`: a workspace on one exact SSH host; or
- `docker_remote`: a new, ephemeral container on an exact remote Docker daemon
  reached through SSH.

There is no automatic placement and no silent fallback. If Hexis cannot read
the selected profile or connect to it, the tool fails without running the
command somewhere else.

## See the current choice

```bash
hexis execution status
hexis execution status --json
```

Status is read-only. It checks whether the required local binaries and named
key files are visible, but it does not connect to a host. The built-in state is:

```text
Active execution profile: local
* local (local) — ready locally
No remote connections were opened.
```

## Prepare SSH deliberately

Use a dedicated key and a dedicated known-hosts file. Hexis invokes OpenSSH
with `-F /dev/null`, `BatchMode=yes`, `IdentitiesOnly=yes`, and strict host-key
checking. It therefore does not consume an ambient SSH config, try unrelated
keys, or accept a new host key in the background.

Create a key if the target does not already have one intended for Hexis:

```bash
ssh-keygen -t ed25519 -f "$HOME/.ssh/hexis_execution" -C hexis-execution
chmod 600 "$HOME/.ssh/hexis_execution"
```

Install the public key through the target's normal administrative path. Then
make the first connection yourself, verify the displayed fingerprint against a
trusted source, and record it in a separate file:

```bash
ssh -F /dev/null \
  -i "$HOME/.ssh/hexis_execution" \
  -o IdentitiesOnly=yes \
  -o UserKnownHostsFile="$HOME/.ssh/hexis_execution_known_hosts" \
  runner@build.example true
```

Do not substitute `StrictHostKeyChecking=no` or blindly trust `ssh-keyscan`.
The remote workspace must already exist. `run_script` maps a local script's
workspace-relative path to the same relative path under that remote root; Hexis
does not silently copy or overwrite a checkout. If you want a copy, synchronize
it explicitly first—for example, `rsync -az ./ runner@build.example:/srv/project/`.

## Add and select an SSH profile

Saving a profile does not activate it and does not open a connection:

```bash
hexis execution add-ssh build \
  --host build.example \
  --user runner \
  --port 22 \
  --workspace /srv/project \
  --identity-file "$HOME/.ssh/hexis_execution" \
  --known-hosts-file "$HOME/.ssh/hexis_execution_known_hosts"

hexis execution test build
hexis execution use build
```

`test` is the explicit network action. It prints a fixed marker, the effective
workspace, and the selected Python executable. A failed check reports the SSH,
host-key, workspace, or Python failure in place. Add `--replace` only when you
intend to replace an existing profile.

SSH commands run under a small target-side Python supervisor. The supervisor
owns one process group, enforces the selected timeout on the target, bounds its
captured output, sends TERM and then KILL to that exact group when necessary,
and reports whether cleanup occurred. Closing only the local SSH client is not
treated as proof that remote work stopped.

## Add remote Docker

Remote Docker accepts only an `ssh://USER@HOST[:PORT]` endpoint. Passwords in
the URL, daemon TCP endpoints, public network mode, and ambient SSH config are
rejected. The workspace is an existing absolute path on the remote Docker
host; it is bind-mounted at `/workspace` by default.

Pull the chosen image on the remote daemon yourself. Tool calls use
`--pull=never`, so a mutable registry tag cannot change merely because a tool
ran. A digest-pinned reference is the strongest choice.

```bash
DOCKER_HOST=ssh://runner@build.example \
DOCKER_SSH_COMMAND="ssh -F /dev/null -i $HOME/.ssh/hexis_execution -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$HOME/.ssh/hexis_execution_known_hosts" \
docker pull python:3.13@sha256:REPLACE_WITH_VERIFIED_DIGEST

hexis execution add-docker container-build \
  --docker-host ssh://runner@build.example \
  --image python:3.13@sha256:REPLACE_WITH_VERIFIED_DIGEST \
  --workspace /srv/project \
  --identity-file "$HOME/.ssh/hexis_execution" \
  --known-hosts-file "$HOME/.ssh/hexis_execution_known_hosts"

hexis execution test container-build
hexis execution use container-build
```

Every call gets a uniquely named, labeled, `--rm` container. Network access is
`none` unless the operator chose `--network bridge` while creating the profile.
On timeout or cancellation, Hexis removes only that exact owned container. A
named volume retains picklable `execute_code` session variables while no
container is running; this is the hibernated state and has no compute cost.

## Tool behavior

The tool names and input schemas do not change when a profile changes:

| Tool | Placement behavior |
|------|--------------------|
| `shell`, `safe_shell` | Run the requested command in the selected workspace; the existing command policy and approval gate still apply |
| `run_script` | Validate the local path, then run the same workspace-relative path remotely; it never uploads or overwrites the file |
| `execute_code` | Keep session variables locally or in expiring remote state; values that Python cannot pickle are returned in `not_persisted` instead of being silently lost |

`safe_shell` parses one allowlisted command and invokes it as direct argv on
every backend. Shell operators, substitution, and redirection are therefore
inert text, while mutating modes such as `find -delete`, output-file flags, and
branch/tag/remote changes are rejected before placement.

The local REPL can bridge back into other Hexis tools. A remote Python process
cannot safely call back through that in-process bridge, so `tool_use` raises an
explicit error telling the agent to call the tool directly in a separate turn.
Ordinary Python imports and file access follow the selected host or container's
own permissions.

Remote `execute_code` state expires after `execution.repl_state_ttl_hours`
(seven days by default). SSH state lives in the remote user's cache directory.
Remote-Docker state lives in the profile's named volume. Code and command
arguments remain subject to the ordinary tool audit and approval policy.

## Return local or remove a profile

Changing the active profile affects new calls only:

```bash
hexis execution use local
hexis execution remove build
```

An active profile cannot be removed. Removing a profile does not delete its
remote workspace, SSH cache, or Docker volume. Hexis prints the preserved
volume name; inspect it and run `docker --host ssh://USER@HOST volume rm NAME`
yourself only if you intend to destroy that state.

Add, replace, select, and remove decisions also enter the ordinary change
journal with profile name, backend type, and prior/new selection. Keys,
known-host contents, command text, and code are not copied into that entry.

## Worker placement

The identity and known-hosts paths must be readable by the process that runs
the Hexis worker. Host user services are the cleanest path:

```bash
hexis service status
hexis service install
```

If workers remain in Docker, mount the chosen files read-only at the exact
configured paths. A missing or over-permissive identity fails the tool and
prints the correction; it never triggers an ambient-key or local-execution
fallback.

## Troubleshooting

| Symptom | What to do |
|---------|------------|
| `refused to run locally` | Restore database access so Hexis can prove the selected profile, then retry |
| Identity mode error | Run the exact printed `chmod 600 PATH`, verify the worker can read it, then retry |
| Host-key error | Connect manually with the same key and known-hosts file, verify the fingerprint, then retry `hexis execution test NAME` |
| Remote workspace missing | Create or synchronize the configured absolute path, then rerun the test |
| Docker image missing | Pull the exact configured reference on the selected remote daemon; Hexis intentionally uses `--pull=never` |
| Python unavailable | Install Python on the SSH host or choose an image containing it; use `--python PATH` when adding/replacing the profile |
| Need an immediate safe fallback | Explicitly run `hexis execution use local`; Hexis will not make that placement decision itself |
