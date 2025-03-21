# Unsupervised Grading Functions for Bowel Sound Activity

This repository provides core functions used in our study for unsupervised scoring of bowel sound activity using spectral features extracted from audio recordings.

The grading system is based on cluster-level analysis and unsupervised similarity measurements. These functions were used to evaluate the degree of gastrointestinal activity based on the resemblance of spectral features to reference clusters.

## 📌 Included Functions

- `cosine_similarity(x, c)`:  
  Computes the cosine similarity between a test observation `x` and a cluster center `c`.

- `knn_cluster_grade(observation, cluster_data, k=2)`:  
  Calculates the average Euclidean distance between an observation and its `k` nearest neighbors in a cluster.

- `cluster_membership_strength(observation, cluster_observations)`:  
  Returns the inverse of the average Euclidean distance between an observation and all members of a cluster. Used as a membership score.

## 🧪 Usage

These functions were used to evaluate multiple spectral feature vectors of equal-length audio windows in our unsupervised analysis. The aim was to score each segment based on how closely it resembled a non-activity cluster.


## 🛠 Requirements

- Python 3.x
- numpy
- scikit-learn

You can install the required libraries using:

```bash
pip install numpy scikit-learn
```

## 📜 License

This code is shared under the MIT License. You are free to use and modify it with attribution.

## 📖 Citation

If you use this code in your research, please cite the related article (link will be added upon acceptance).
