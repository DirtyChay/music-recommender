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

- Python ≥ 3.13
- pandas
- matplotlib
- seaborn
- networkx

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


## 1. Preprocesssing:

**Transformation:**
We transformed the raw user-song interaction data into meaningful user-level features to prepare for clustering and recommendation tasks.  This involved aggregating each user’s activity to compute statistics such as total play count, total listening time, and the number of unique songs and artists. We also identified each user’s favorite attributes, including their most-played song, favorite artist, preferred genres, and most-listened-to release year. We calculated the total listening time as total plays (per song) x track duration. (File 1)

**Scaling:**
To scale our numerical features we tested MinMaxScaler and RobustScaler, we proceeded with using RobustScaler since it is not as highly sensitive to outliers like MinMax. (File 2)

**Encoding:**
We one hot encode our categorical features (genre and artist) using pd.getdummies. Due to lack of computing power, we only keep the one hot encoded columns representing the genre. This is because for our artist category, we have 14975 unique artists and including these all as features was very costly. (File 2)

**Feature expansion:**
We apply PCA to the user data, which transforms the original features into a new set of orthogonal components that capture the most important patterns in the data. We set n_components to 0.9 which automatically selects the number of components that explain 90% of the variance in the data. We reduce dimensionality and keep important feature information thus facilitating the clustering process. (File 3)

## 2. Model Evaluation:

As we are using clustering, we cannot check for accuracy. However, we were able to capture 92.2% of the total variance for our model using PCA. Then, we use the elbow method to find a suitable k for our k means model. To evaluate the clusters, we produce a pairplot of the model using a random sample and observe how the clusters look across the principal components. 

**Model Steps**
1.Run PCA on the preprocessed data
2.Use the elbow method to find a k value
3.Run kmeans with the dataframe created in the PCA step

**Evaluation:**

**The Elbow Method:**
This is an indirect method of evaluating the model. Since this step calculates inertia, it helps us visualize how the model is reacting to different cluster numbers. Before running this and finding out the best k, we set the number of clusters to 30. After running the elbow method, we were able to find the right balance for our model, since too few clusters causes high inertia and too many clusters can cause overfitting and add unnecessary complexity. We found that the optimal number of clusters for our model was 9. 

**Visual Inspection:**
Once we conducted PCA and compared scatter plots before and after. Before PCA, we saw that the model clustered groups mainly by total_plays and total_track_time. Although they are important features, we did not want the model to simply group by these, since someone who has a 1000 minutes of Taylor Swift theoretically should not be grouped with someone with 1000 minutes of Metallica. PCA in fact did improve clustering.

**Sample Silhouette Score:**
This score helps us understand how similar each point is to its own cluster versus other clusters. If the score is close to 1, it indicates that our points are well clustered, if it's around 0 the points may be on the boundary between clusters, and if it's negative then the point may be in the wrong cluster. Running this is very expensive, so we selected a random sample of 50k points. The sample silhouette score was approx 0.34. This indicates that the clusters capture some structure in the user data but there is overlap between clusters. The clusters are still useful in helping us identify groups by listening patterns and genre. With just genre, artist, and listening metrics, the feature space is pretty limited, so clusters will naturally not be very great.

## 3: Questions:

**Where does your model fit in the fitting graph?**

Since we conducted unsupervised machine learning we do not plot a learning curve. We did however, use the elbow method to find the best number of clusters for our model. We noticed two elbows, a clear elbow at 9 and a potential elbow at 15.
We visualized the number of points in each cluster for different k values and that validated our choice of k=9 as higher numbers of clusters resulted in several tiny clusters which we could not justify given our general intuition for how people tend to listen to music.


**What are the next models you are thinking of and why?**

We are thinking of building a density based clustering model for our next model because our data is dense in some regions and sparse in others. We want to be able to capture the variance in the density. This may also make the clusters more representative of user behavior as that presents itself in regular shaped/spherical clusters (as in K Means) but rather presents in irregular shapes with varying densities. 


## 4. Conclusion Section:

Our model clusters similar users together, because we want to map a user to a cluster, then pull the top songs in that cluster and recommend them to the user as a playlist. However, as we do not have users to test on, we will not know if we actually succeeded in creating a robust recommendation model.

What we can speak to is that we are somewhat happy with how the clusters look in general, but notice some strangeness: our largest cluster has 531,864 datapoints, and our smallest cluster only has 62 datapoints. We looked into the statistics for the clusters, and found our outlier cluster #6 (with 62 users) listens to songs at a much higher rate than people in other clusters (perhaps they are bots) and on average have played 614 songs each. The big cluster #0, which contains half the users at ~500,000, has a significantly lower number, 22 songs on average played per user. It's possible that this cluster comprises primarily people trying out the music platform and listening to songs for an hour or two and then leaving. 

**Some more detailed findings:**

Cluster Sizes:
Cluster Size and Engagement: The clusters show a wide range in the number of unique users, from very large clusters like Cluster 0 with 531,864 users to very small ones like Cluster 6 with only 62 users
Large Clusters (0, 4, 5, 8): These clusters contain the majority of users. While they have high total plays and track times, their average plays per user and average track time per user are relatively lower. 
Small/Medium Clusters (1, 2, 3, 6, 7): These clusters, despite having fewer users, exhibit significantly higher average engagement per user. Cluster 6 stands out with an extremely high average plays per user 614.87 plays and average track time per user with 166,681.74 seconds
Engagement Levels across Clusters:
Highly Engaged Clusters (1, 2, 3, 6, 7): These clusters show much higher average plays and track times per user compared to the larger clusters. Cluster 3 and Cluster 6, in particular, demonstrate really high engagement. 
Moderately Engaged Clusters (0, 4, 5, 8): These clusters represent the bulk of the user base, with lower per-user averages, possibly representing casual listeners.

In summary, the KMeans clustering segmented users into groups with different listening behaviors. We observe a trade-off where smaller clusters exhibit exceptionally high engagement per user, while larger clusters represent a broader, more moderately engaged audience. This suggests that the model captured varying degrees of user dedication and potentially diverse musical preferences, which are valuable for targeted recommendations.


**To improve the model we might try the following:**

Include more features so that clusters are more dependent on tastes (songs, genres) rather than activity (number of songs played)
We ran one-hot encoding on the artists so that each user would have their favorite artist tracked. However, this added too many columns and affected PCA performance. Perhaps we could try acquiring better resources in our cloud environment to run PCA. Once we have run PCA, we could save these results. 
We could use the artist_term data (which has terms assigned to artists, with a total of ~1K unique terms) to augment our dataframes and use one hot encoding on those instead of the artist names to reduce the size of newly added features from ~10K to ~1K for artists
Similarly, we ran one-hot encoding on the songs to track a user's favorite songs, but this added too many columns. Perhaps acquiring better resources (more RAM/processing on the cloud) would allow us to include these columns and run PCA.
We could try creating more features in the preprocessing step, such as “favorite artist hotness” and “favorite artist familiarity” to the dataframe we are running the model on.









