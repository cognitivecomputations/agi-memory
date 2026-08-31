---
name: code-execution
description: Run programs, shell commands, and scripts — compile and execute code, test programs in installed languages, work with files in the workspace
category: system
requires:
  tools: [safe_shell, execute_code]
contexts: [heartbeat, chat]
aliases: [code, script, compute, calculate, run, python]
bound_tools: [safe_shell, execute_code, run_script, read_file, write_file, list_directory, glob, grep, edit_file, shell, browser]
---

# Code Execution

All four tools use the operator-selected execution profile. Do not ask for or
invent a backend argument—the public tool schemas stay stable. If a result says
the selected SSH or remote-Docker profile is unavailable, report its exact
recovery step; never retry by assuming local placement. `run_script` requires
the same workspace-relative file to exist on the remote target. Remote
`execute_code` persists picklable session variables, reports unpicklable names,
and cannot call the local `tool_use` bridge; call that tool directly instead.

Use this skill to actually run things: compile a program, execute a script,
test code someone shares with you, check what a command outputs, or work with
files. Running code is how claims about code become knowledge — a program
that compiles and passes its tests teaches you more than any amount of
reading it.

## Choosing the tool

1. `safe_shell` — one shell command with guarded defaults. The right first
   reach for compiling, running installed binaries, checking versions, and
   inspecting the environment (`which`, `ls`, language toolchains on PATH).
2. `execute_code` — run a code snippet directly in a supported runtime
   without touching the filesystem.
3. `run_script` — a saved multi-step script when the work has real structure.
4. `write_file` + `safe_shell` — the classic loop for languages the runtime
   doesn't embed: write the source file, invoke its compiler or interpreter
   from the shell, read the output.

## Practice

- State what you ran and show the real output — including failures. A
  compile error is a finding, not an embarrassment.
- Working files belong in the workspace directory; clean up what has no
  lasting value.
- When a language or binary is missing from PATH, say so plainly and name
  what you checked (`which <name>`). Absence of a toolchain is a fact about
  the environment, never something to paper over with imagined output.
- Long or destructive operations deserve a sentence of intent before you
  run them.
