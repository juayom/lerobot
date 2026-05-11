# Root loose-files cleanup (2026-05-11)

## Kept at repository root intentionally
These are normal repository/package metadata or install/build files and should usually stay at root:
- .dockerignore
- .gitattributes
- .gitignore
- .pre-commit-config.yaml
- CODE_OF_CONDUCT.md
- CONTRIBUTING.md
- LICENSE
- MANIFEST.in
- Makefile
- README.md
- SECURITY.md
- docs-requirements.txt
- pyproject.toml
- requirements-macos.txt
- requirements-ubuntu.txt
- requirements.in
- setup.py
- uv.lock

## Moved out of repository root
These looked like one-off utilities, temporary scripts, or model assets rather than required root metadata:
- get-pip.py
- recover_shoulder_lift.py
- tmp_release_follower_torque.py
- test.py
- sam_vit_h_4b8939.pth

## New location
- backups/2026-05-11_root-loose-files_cleanup/root_loose_files/
