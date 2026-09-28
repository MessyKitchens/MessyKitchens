## Summary

What does this change do, and why?

## Checklist

- [ ] `python -m pytest -q` and `python -m ruff check .` pass locally
- [ ] `python scripts/dev/check_release.py` passes on a clean export (README section 7)
- [ ] Tests cover behavior changes
- [ ] README, docs, and `docs/release/manifest.json` are updated when an interface, artifact contract, or the file inventory changes
- [ ] No datasets, checkpoints, generated outputs, credentials, or private paths are included
- [ ] Third-party code keeps its attribution in `THIRD_PARTY_NOTICES.md`
