# FEGA project page

This folder is a standalone Jekyll project page based on the UKP Academic Project Page template.

## Refresh the RAVEL atlas data

The interactive atlas is derived from the checked-in RAVEL artifacts, not hand-authored page data. From the repository root, run this in the FEGA Python environment after results change:

```bash
python page/scripts/export_results.py
```

It reads `results/fega/ravel/` and writes compact page assets below `page/assets/generated/`. It does not modify `.paper/` or `results/`.

## Run locally

From the repository root:

```powershell
cd page
jekyll serve --host 127.0.0.1 --port 4000
```

The first start can spend about 30 seconds resolving Ruby dependencies. Keep this terminal open and wait until it prints `Server address: http://127.0.0.1:4000/`; then open [http://127.0.0.1:4000](http://127.0.0.1:4000). Stop the server with `Ctrl+C`.

`jekyll build` also creates `_site/`, but it does **not** start a web server. If `serve` prints an error or returns to the prompt, copy that terminal output before closing it.

If Jekyll dependencies are not already installed, run `bundle install` once and start it with `bundle exec jekyll serve --host 127.0.0.1 --port 4000`.
