# Netflix Content Segmentation (Task 4)

## Overview
An unsupervised machine learning project that segments the Netflix catalog (8,790 titles) into meaningful content groups using K-Means clustering and Principal Component Analysis (PCA).

## Methodology
- **Feature Engineering (45 Features):** Combined `release_year`, numeric `duration` (`duration_num`), binary content type (`is_tv_show`), and 42 multi-label encoded genre tags (`listed_in`).
- **Standardization:** Applied `StandardScaler` across all 45 features prior to clustering so continuous variables and binary genre indicators contribute on a uniform scale.
- **Optimal Cluster Selection:** Evaluated $K = 2$ through $8$ using the Elbow Method (inertia) and Silhouette Score, selecting **$K = 5$** (Inertia: `325,948`, Silhouette Score: `0.1442`) as the optimal balance between granularity and cluster cohesion.
- **Visualization:** Projected the 45-dimensional scaled feature space onto 2 principal components using PCA to visualize cluster separation.

## Identified Content Segments ($K = 5$)
| Cluster | Size | Dominant Type | Avg Release Year | Avg Duration | Primary Content Profile |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 849 | TV Show | 2016.2 | 2.1 Seasons | Kids' & Family TV Series |
| **1** | 5,317 | Movie | 2012.9 | 99.9 min | Mainstream & International Feature Films |
| **2** | 825 | Movie | 2014.1 | 95.6 min | Independent & International Dramas |
| **3** | 1,180 | TV Show | 2016.7 | 1.7 Seasons | International TV Dramas |
| **4** | 619 | TV Show | 2017.5 | 1.4 Seasons | Recent Reality TV Series |

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt