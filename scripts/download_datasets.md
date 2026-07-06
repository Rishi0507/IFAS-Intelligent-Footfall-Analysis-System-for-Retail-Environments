# Dataset download guide

The three stage-1 datasets all require manual download (academic / form-submission
access). After download, upload the zips to Colab via the `training_pipeline.ipynb`
upload cell — the notebook handles organization inline.

## PETA (PEdAttridute dataset)

- 19,000 pedestrian images, 8,700+ persons, 105 binary attributes
- **Where to request**: https://mmlab.ie.cuhk.edu.hk/projects/PETA.html
- **What you'll get**: a .zip with image folders + either `PETA.mat` or `Label.txt` files
- **Gender attribute**: `personalMale` (1=male, 0=female) in PETA.mat, or gender field in Label.txt

## RAP (Richly Annotated Pedestrian Attribute dataset)

- ~42,000 images, 27,000+ persons, 92 attributes
- **Where to request**: https://rap.ideal-x.org/  (free for academic use, requires email registration)
- **What you'll get**: a .zip with image folder + `RAP_annotation.mat`
- **Gender attribute**: column 0 in the `attribute_labels` matrix (1=male, 2=female)

## SCface (Surveillance Camera Face dataset)

- 130 subjects, 5 cameras each, ~4,160 images total
- **Where to download**: http://www.scface.org/  (free for research, requires form submission)
- **Gender labels**: SCface has **no public gender labels**. The training notebook auto-labels via CLIP, you correct manually.

## Expected disk usage

| Dataset  | Images  | Disk   |
|----------|---------|--------|
| PETA     | ~19,000 | ~1.5 GB |
| RAP      | ~42,000 | ~3.5 GB |
| SCface   | ~4,160  | ~250 MB |
| Combined | ~65,000 | ~5.3 GB |
