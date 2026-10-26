# Deploying

The site is published by GitHub Pages from the `gh-pages` branch of
[afiq-aswadi/bft-pmc](https://github.com/afiq-aswadi/bft-pmc), at
<https://afiq-aswadi.github.io/bft-pmc/>. This folder is a git repository on
that branch, so an update is a rebuild, a commit and a push:

```bash
python3 build_pmc.py && python3 build_site.py --no-figures
git add -A && git commit -m "Describe the change"
git push origin gh-pages
```

## Links

The page is plain static files with relative links, so it works at any
sub-path with no configuration. Everything it links to outside itself is
online:

- **Paper**: the OpenReview page, <https://openreview.net/forum?id=qZ5gMk6Gpd>.
- **Supplement**: `icl2_supplementary.pdf` on the code repository's `main`.
- **Run settings**: each configuration's `run_record.json`, from the copies in
  the code repository's `paper_data/camera_ready/pmc_diagnostics/`
  (`RECORDS_URL` in `build_pmc.py`).

The figure archive links to no source images, because the full-resolution
originals live only in the authors' workspace.

## Turning Pages on

Once, by a repository admin: **Settings → Pages → Build and deployment**, set
Source to "Deploy from a branch", branch `gh-pages`, folder `/ (root)`, and
save. Or, with admin rights:

```bash
gh api -X POST repos/afiq-aswadi/bft-pmc/pages -f 'source[branch]=gh-pages' -f 'source[path]=/'
```

## Notes

- **Size.** 29 MB, mostly `figures/` (25 MB). A default clone of bft-pmc
  downloads this branch too, and each full rebuild rewrites every WebP, so
  rebuild figures only when they change: `build_site.py --no-figures`
  refreshes the data alone.
- **`.nojekyll`** makes Pages serve the files as they are.
- **Caching.** Asset URLs carry a content hash, so a redeploy cannot leave a
  visitor on a stale script. Run the build before committing to refresh the
  stamps.
