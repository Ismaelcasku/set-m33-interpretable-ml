# Publish this directory to GitHub

Suggested repository name: `set-m33-interpretable-ml`.

After choosing/creating the empty GitHub repository:

```bash
git init
git add .
git commit -m "Paper-aligned SET-M33 code release"
git branch -M main
git remote add origin git@github.com:<ACCOUNT>/set-m33-interpretable-ml.git
git push -u origin main

git tag -a v1.0.0 -m "SET-M33 paper-aligned code release"
git push origin v1.0.0
```

Before making the repository public, decide whether to add an explicit software
license. The supplied release intentionally does not change the current default
copyright status.

After the final Zenodo update, add the GitHub URL and final data-record version
to the manuscript Data Availability statement and, if desired, add the
repository URL/archival DOI to `CITATION.cff`.
