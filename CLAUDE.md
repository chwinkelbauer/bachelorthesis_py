# Project rules

- `docs/FRAMEWORK.md` is the living specification of this project. **Every change** to code, model, mathematics,
  parameters, presets, metrics or experimental design must update the affected sections of that file (including the
  status table) and add a dated line to its changelog in the same change.
- Present design changes, challenge them scientifically, and get the user's approval before implementing larger changes.
- Keep the folder structure (configs/, economy/, agents/, learning/, analysis/, scripts/, tests/, docs/). Rewrite in place;
  no parallel copies of modules.
- Run `python -m unittest discover -s tests -t .` after changes; all tests must pass.
- No good may be hard-coded or rewarded as money; money metrics stay symmetric over goods and null-referenced.
