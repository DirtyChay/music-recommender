# Music-Recommender

This project aims to create song recommendations as a personalized playlist using the Million Song Dataset combined with
Nest User data.

Link to dataset:

http://millionsongdataset.com/pages/getting-dataset/#subset

## Environment Setup (Local)

Downloading the subset from the official site is complicated, so please download the 3 files from the Google Drive
folder and place all files in the main project directory before running the code.
https://drive.google.com/drive/folders/1hGWJrezEpd7SMbizWebN_Ku5nl1LAUAh

**Required libraries:**

- **Python ≥ 3.13**
- **pandas**
- **polars**
- **pyarrow**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn**

## Environment Setup (Google Collab):

- The environment for this project is google colab + Google Drive
- The data was uploaded onto Google Drive
- In Google Drive create drive a folder called 'DSE'
- Then create a shortcut to the drive folder: https://drive.google.com/drive/folders/1t-1ueI5ETe1Io3TsyBkqkf3gaaftR9Al
- Run mount code in Google Collab
- Read csvs

## Google Collab Files:
- File 1: Data Retrieval and Aggregation
  https://colab.research.google.com/drive/1-q-dqyzmLsRcgv1YBy9cNQehbUDZ_7W9#scrollTo=pFw4IqMhIPGU 

- File 2: Scaling and One Hot Encoding**
  https://colab.research.google.com/drive/1im-2VShAmXn2CdvcDh29SkmVWCuXrMyw#scrollTo=kSa132xItemW
  
- File 3: PCA, Clusterng and Model Evaluation
  https://colab.research.google.com/drive/1t6gh9YIIY5zEJQHlJ6jL0e3jdtfI5q70#scrollTo=9096ff6b

- Data: https://drive.google.com/drive/folders/1hGWJrezEpd7SMbizWebN_Ku5nl1LAUAh?usp=drive_link




my current work starts here:

## Methods:

For a complete walkthrough of the project, including detailed explanations, figures, and code, refer to the [full project report](INSERT_REPORT_LINK_HERE).

### 1. Exploratory Data Analysis (EDA)
- **Data Sources:** Million Song Dataset – track metadata and user-song interactions.
- **Purpose:** Understand data structure, identify missing values, outliers, and patterns.
- **Visualizations Used:**
  - Numerical features: Histograms, strip plots, correlation heatmaps.
  - Categorical features: Bar charts, pie charts.
  - Example insights:
    - Artist familiarity and hotness had few missing values (-1 indicates missing).
    - Many songs had missing or zero year values, removed during preprocessing.
    - Popularity of genres: Rock most common, followed by electronic and pop.
    - Right-skewed distribution of song plays with extreme outliers captured via stripplots.

### 2. Data Preprocessing
- **Goal:** Prepare two main datasets for modeling and playlist generation:
  1. **Song-level DataFrame:** Contains user-song interactions merged with metadata and genre labels.
     - Dropped rows with missing genres and year = 0.
     - Created a new feature: `total_track_time = duration * plays`.
     - Saved as Parquet for easy access.
  2. **User-level DataFrame:** Aggregated user listening statistics for clustering.
     - Calculated total plays, total listening time, number of unique songs/artists.
     - Determined preference-based features: most-listened artist, favorite genres, most common release year, most-played song.
- **Feature Preparation:**
  - Dropped NaNs.
  - **Numerical variables:** Scaled using Min-Max and Robust scaling (Robust chosen for outliers).
  - **Categorical variables:** One-hot encoding for favorite genres and artist.
- **Dimensionality Reduction:**
  - Applied PCA to numerical features, retaining components explaining 90% variance.
  - Applied UMAP 
- **Cluster Preparation:**
  - Used the elbow method on PCA-transformed data to find optimal k for K-means.
  - Constructed heatmaps to visualize cluster distributions across different k values.

### 3. Clustering Models
- **K-means Clustering**
  - Tested different numbers of clusters
  - Key parameters: n_clusters=(9 or 26), random_state=42, n_init='auto'
  - Evaluated using silhouette score on a subset of data and visualizations.
  - Output: Cluster labels assigned to each user.
- **HDBSCAN**
  - HDBSCAN automatically detects clusters and labels low-density points as noise.
  - Key parameters: min_cluster_size=200, min_samples=10, n_neighbors=50, min_dist=0.
  - Evaluated using visualizations
  - Output: Cluster labels assigned to each user.
    

### 4. Playlist Generation
- Merged user with cluster assigments with song-level data.
- Computed global song popularity.
- For each cluster for each algorithm (KMeans and HDBSCAN):
  - Ranked songs by cluster-specific popularity.
  - Selected top 20 songs per cluster
- Output: Cluster-level playlists for recommendation.
- User Top 20: Output User's top 20 songs
- Compare cluster playlists to user top songs

---

## Results
*Content to be added from the results section of the full report.*

## Discussion
*Content to be added from the discussion section of the full report.*

## Conclusion
*Content to be added from the conclusion section of the full report.*

## Statement of Collaboration
*add* 






