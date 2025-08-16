# Contributing

Thanks for helping improve the Sri Lanka Rain & Energy project! This guide describes how to contribute effectively.

## Branching model
- Create a feature branch from `main`: `feature/<short-name>`
- Open a pull request (PR) early as a draft; link to any issues.
- Keep PRs focused (small and reviewable).

## Code standards
- Python: prefer type hints and clear naming.
- Notebooks: restart and run all before committing to keep a clean execution order.
- Avoid committing large binaries; use the `artifacts/` folder for models/outputs (gitignored).

## Testing and validation
- Manually test the Streamlit app locally: `streamlit run app.py`.
- For notebooks, ensure the end-to-end run completes without errors.

## Reviews
- Request at least one reviewer.
- Address comments promptly; use follow-up commits (avoid force-push to shared branches).

## Releases / Deployments
- Streamlit Cloud automatically installs from `requirements.txt`.
- Keep `requirements.txt` minimal and pinned.
