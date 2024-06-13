# Semi-Truths
Dataset for Evaluating AI-Generated Image Detectors on Various Magnitudes of Change

<img width="500px" src="./figures/mag_of_change_head_fig.png" alt="Different measures of magnitudes of change presented in SemiTruths: Area Ratio and Semantic Change" />

## Get Started

```
conda create -n semi python=3.10
conda activate semi
pip install -r requirements.txt
```

## Generating the SemiTruths Dataset

To demonstrate our pipeline for creating highly descriptive AI-Augmented images for evaluating detectors, we release a subset of input data used for this project as well as all relevant scripts.

The input data sample is sourced from existing semantic segmentation datasets (ADE20K, CityScapes, CelebAHQ, HumanParsing, OpenImages, SUNRGBD).

To simulate the data generation process on a small sample:
```
python generate_semi_truths.py 
```
NOTE: An argparse is defined within this script that can be edited to reflect different data locations.

<img width="500px" src="./figures/final_full_pipeline.png" alt="Diagram of the SemiTruths image augmentation process." />


## Stress-Testing

We release the stress-testing inference pipeline that was used to get predictions from SOTA AI-generated image detectors on the Semi-Truths dataset. Run:

```
cd stress_testing
python run_1.py
```
NOTE: An argparse is defined within this script that must be edited to reflect data locations.

