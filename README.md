# mushroom
A workflow for 3D serial section registration and analysis

Under active development

## Library installation

Although not always neccessary, we recommend installing into a fresh environment when possible to avoid installation issues.

```bash
conda create -n mushroom -c conda-forge -y python=3.11 jupyter
conda activate mushroom
pip install git+https://github.com/ding-lab/mushroom.git
```

## Tutorial

API subject to change...

[Registering multi-modal serial section data](https://github.com/ding-lab/mushroom/blob/main/notebooks/tutorials/data_preperation_and_registration.ipynb)

[Running Mushroom](https://github.com/ding-lab/mushroom/blob/main/notebooks/tutorials/mushroom_tutorial.ipynb)

[Output analysis](https://github.com/ding-lab/mushroom/blob/main/notebooks/tutorials/output_analysis.ipynb)

## Manuscript materials

Location of materials for **3D Imaging and Multimodal Spatial Characterization of the Precancer-to-Cancer Transition in Breast and Prostate**

[H&E protein prediction model training](https://github.com/ding-lab/mushroom/blob/main/notebooks/manuscript/submission_v1/he_channel_prediction.ipynb)

[Serial section registration workflow](https://github.com/ding-lab/mushroom/blob/main/notebooks/manuscript/submission_v2/step1_register_datasets.ipynb)

[3D ROI data generation and quantification](https://github.com/ding-lab/mushroom/blob/main/notebooks/manuscript/submission_v2/step2_roi_data_gen.ipynb)

[Automated 2D epithelial region detection and quantification](https://github.com/ding-lab/mushroom/blob/main/notebooks/manuscript/submission_v2/step5_region_characterization.ipynb)

[DEG plotting and analyses](https://github.com/ding-lab/mushroom/blob/main/notebooks/manuscript/submission_v2/step6_deg_analysis.ipynb)

[Automated region connectivity for HT704B1](https://github.com/ding-lab/mushroom/blob/main/notebooks/manuscript/submission_v1/analyses_brca_v3.ipynb)

## Previous versions

Location of materials for **Tumour evolution and microenvironment interactions in 2D and 3D space**

[Neighborhood model training](https://github.com/ding-lab/mushroom/blob/subclone-resubmission/notebooks/manuscript/submission_v1/step3_train_mushroom_cancer_v2.ipynb)

[Analysis](https://github.com/ding-lab/mushroom/blob/subclone-resubmission/notebooks/projects/subclone_paper/step7_figure6_revisions_v2.ipynb)
