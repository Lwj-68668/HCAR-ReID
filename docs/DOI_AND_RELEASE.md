# DOI and Release Workflow

Use this workflow to satisfy journal requirements for permanent, citable, and reproducible artifacts.

## 1) Prepare Public Repository

- Ensure README, requirements, and docs are complete.
- Ensure private or licensed raw datasets are not committed.
- Add `CITATION.cff` with final metadata.

## 2) Create a Versioned GitHub Release

1. Push code to GitHub.
2. Create an annotated tag (example: `v1.0.0`).
3. Create a GitHub Release from that tag.
4. Include release notes with:
   - Manuscript title
   - Main results
   - Exact training/evaluation commands

## 3) Mint Code DOI via Zenodo

1. Sign in to Zenodo.
2. Link Zenodo with your GitHub account.
3. Enable this repository in Zenodo GitHub settings.
4. Publish GitHub Release `v1.0.0`.
5. Zenodo creates an archive and mints a DOI.
6. Copy both links:
   - Version DOI (for specific release)
   - Concept DOI (for all versions)

## 4) Data DOI Strategy

Because benchmark datasets may have redistribution restrictions:

- Do not rehost restricted raw data unless license allows.
- Publish a separate Zenodo record for reproducibility metadata:
  - split files
  - preprocessing scripts
  - experiment logs
  - checksum files
- Use that record DOI as your "data DOI / data statement DOI".

## 5) Update Manuscript and Repository Text

Update placeholders in:

- `README.md`
- `CITATION.cff`
- manuscript abstract/footnote/availability statement

Required language to keep:

- The code is directly related to the manuscript submitted to *The Visual Computer*.
- Readers are encouraged to cite the manuscript when using the released artifacts.

## 6) Final Pre-Submission Checklist

- GitHub link is public and permanent.
- DOI link resolves correctly (`https://doi.org/...`).
- README includes dependency and usage instructions.
- Reproducibility docs and commands are present.
- Manuscript contains GitHub + DOI links.
