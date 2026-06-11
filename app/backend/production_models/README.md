# Production model artifacts

Place downloaded mllabiome-ii model ZIP artifacts in this directory.

Model artifacts are intentionally not committed to the Git repository. Download them from the GitHub Release named `models-v0.1.0` or another compatible model-artifact release.

Expected filenames:

```text
ibs_mpma_e_model_v0.1.0.zip
scz_mpma_e_model_v0.1.0.zip
crc_mpma_e_model_v0.1.0.zip
ghs_mpma_e_model_v0.1.0.zip
ibd_mpma_e_model_v0.1.0.zip
dm_mpma_e_model_v0.1.0.zip
```

After downloading, restart the backend and check:

```bash
curl http://localhost:8000/api/inference/models
```
