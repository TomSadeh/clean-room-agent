# USB Air-Gap Intermediary: Design Overview

## Purpose

A minimal, auditable transfer verification layer between Jane's machine and the outside world. Prevents outbound exfiltration and inbound contamination at the physical boundary.

## Hardware

One dedicated machine. Minimal Linux install. No network interface — disabled in BIOS, not just software. Two USB ports: one for the source drive (read-only mount), one for the clean output drive. Nothing else runs on this machine.

## Core Principle

Never copy. Always reconstruct. The intermediary reads content from the source drive, validates and filters it, then writes a new clean artifact to the output drive. No byte-for-byte transfer ever occurs. What comes out is built from scratch using only the validated content.

## Inbound Path (External → Jane)

**Input:** USB containing cloned git repositories (mature codebases for bootstrapping/training).

**Process:**
1. Mount source USB read-only.
2. Walk file tree. Build manifest of everything found.
3. Whitelist filter by extension. Only plain text source files (.py, .ts, .js, and whatever else is explicitly listed). Everything not on the whitelist is dropped and logged.
4. For each whitelisted file: read as raw bytes, validate UTF-8 encoding, re-encode to clean UTF-8. Reject files that aren't valid text.
5. Content scan: flag files with anomalous patterns — unusual comment density, strings resembling LLM instructions, embedded base64, abnormal line lengths. Flagged files go to a quarantine manifest for human review, not automatic rejection.
6. Git history reconstruction: extract commit metadata (author, date, message) and diffs only. Reconstruct a new git repo from scratch using extracted data. No .git/hooks, no workflows, no CI configs, no binary objects. The output repo has the same logical history but is structurally new.
7. Hash every output file. Write full manifest with hashes.
8. Write to clean output USB.
9. Produce human-readable transfer log: what was copied, what was dropped, what was flagged, all hashes.

**Human step:** Review the transfer log before plugging the clean USB into Jane's machine. No automation crosses the air gap.

## Outbound Path (Jane → External)

**Input:** USB containing files Tom explicitly chose to extract from Jane's machine.

**Process:**
1. Mount source USB read-only.
2. Walk file tree. Build manifest.
3. Whitelist filter — same principle. Only plain text source files and plain text logs.
4. Re-encode all text to clean UTF-8. Strip all file metadata (timestamps, permissions, extended attributes).
5. Content scan for steganographic patterns: unusual whitespace sequences, abnormal variable naming entropy, embedded data in comments, polyglot file structures. Flag for human review.
6. Hash everything. Write manifest.
7. Write to clean output USB.
8. Produce transfer log.

**Human step:** Review the transfer log. This is the more sensitive direction — take time with it.

## What the intermediary explicitly does NOT do

- Execute anything from either USB. It reads bytes and writes bytes.
- Use any external dependencies. Standard library only.
- Connect to any network, ever.
- Make decisions autonomously. Anomalies are flagged for human review, never auto-resolved.
- Run silently. Every action logged, every decision explained.

## What gets dropped unconditionally

- Binary files of any kind.
- Executable files.
- Git hooks, CI/CD configs, GitHub Actions workflows.
- Dotfiles and hidden directories (except .git history data, which is reconstructed not copied).
- Anything not on the explicit extension whitelist.

## The script itself

One file. No dependencies. Small enough to read in full and understand completely. The entire security model depends on the intermediary being simple enough that a human can audit it by reading it. If the script is too complex to hold in your head, it's too complex.

## Logging

The transfer log is the audit trail. It answers: what crossed the boundary, in what direction, when, what was dropped, what was flagged, and the hash of every file that made it through. Keep all transfer logs permanently, even if you delete the transferred files.

## Update policy

The intermediary script changes rarely. Every change is a diff you review manually. The intermediary does not self-update. It does not receive updates from Jane's machine. Updates come from you, on a USB you prepared yourself, reviewed before loading.
