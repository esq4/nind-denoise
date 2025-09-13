1. Publish the consolidated follow-up refactor plan for nind-denoise and request confirmation. (done)
2. Triage current pylint findings and correctness risks in legacy helpers under src\nind_denoise\common\libs.

- Re-run pylint focused on src\nind_denoise\common, src\nind_denoise\tools, src\nind_denoise\models, src\nind_denoise\networks, and
  src\nind_denoise to generate an up-to-date, file-by-file issue list.
- Classify issues by severity: correctness (undefined names, inconsistent returns), safety (broad except, dangerous
  defaults), and style (long lines, trailing whitespace).
- Select a minimal, low-risk subset for the first pass (correctness and safety first).

3. Implement minimal, safe fixes in src\nind_denoise\common\libs to address correctness and safety without changing
   behavior.

- Add missing imports or define missing symbols flagged by pylint (e.g., undefined names).
- Replace dangerous default mutable arguments with None-sentinels and initialize inside functions.
- Normalize and make return paths consistent across branches.
- Narrow broad exception handlers and avoid bare excepts; log errors appropriately.
- Ensure final newlines, remove trailing whitespace, and reflow lines > 100 chars for readability.
- Prefer f-strings over str.format and literal {}/[] constructions where suggested by pylint.

4. Subprocess safety and logging review (pipeline utilities).

- Audit uses of run_cmd and subprocess calls to ensure check=True (or equivalent) and robust error handling.
- Confirm Path-to-str conversions are explicit and cwd is used consistently to avoid path confusion.
- Add unit tests that simulate failure modes for run_cmd without invoking external binaries.

5. CLI and orchestration readability (optional, no behavior change).

- Consider splitting large CLI functions in src\denoise.py into smaller helpers to reduce too-many-args/locals while
  keeping Typer interface stable.
- Add smoke tests to assert CLI module importability and basic option parsing.

6. Configuration and imports cleanup.

- Trim unused imports in src\nind_denoise\config.py and other modules where applicable.
- Remove ad-hoc import-outside-toplevel occurrences where lazy import is no longer needed.

7. Tests: expand small, fast unit coverage (no external tools).

- Add tests for small pure functions: get_output_extension, resolve_output_paths, get_stage_filepaths, and XMP
  transforms edge cases.
- Use tmp_path and in-memory data; ensure tests clean up artifacts.
- Introduce pytest markers for integration tests (darktable-cli, gmic) and keep them skipped by default; document -m
  integration usage.

8. Validation loop.

- Run black, pylint, and targeted pytest on modified paths to ensure quality and prevent regressions.
- Address any regressions or flake findings; iterate until clean.

9. Documentation touch-ups.

- Update README or .junie\guidelines.md only if behavior or developer workflow meaningfully changes; keep minimal.

10. Prepare PR and changelog notes.

- Summarize changes, list impacted modules, and include before/after lint snapshots for transparency.