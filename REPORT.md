# Introduction to Your Project

## Introduction

Music is an integral part of many people’s lives. It’s on during the commute to work, playing in the background at coffee shops, and echoing through stores—we encounter it everywhere. Everyone has varying tastes in music, some enjoy rock and roll while others might prefer the sound of smooth jazz. Before the explosion of digital data in recent years, people would curate their playlists manually or discover new music by conducting research. Now, information about musical works has become widely available. As a result, music platforms have started leveraging this data to enhance the listening experience. Modern platforms now offer features such as personalized song recommendations, auto-generated playlists, collaborative playlists, and even AI DJs. All of these rely on analyzing both user behavior and song characteristics to tailor music to individual tastes. However, sometimes the “personalization” can miss the mark. 

This project does not involve predicting exactly what songs a user will listen to, instead, its objective was to generate a set of recommended songs (a playlist) for each user based on observed user–song interaction data. This was conducted using the million song dataset (http://millionsongdataset.com/) metadata and user-song interaction data. First, data exploration was conducted, the data was preprocessed, then two clustering algorithms were implemented and the results were analyzed. Music data is rich and complex, so it creates a really good environment for exploring clustering algorithms and this sector of machine learning while working on something we are passionate about. In this project, the focus is on improving the data and optimizing the models in order to create clusters that appropriately reflect different song/music preferences. Good predictive models allow organizations to make data driven decisions, increase efficiency, and reduce risks. In our particular case, ‘predicting’ what customers will like can create a personalized, enhanced user experience and give us valuable insights on user listening behavior. 

---

# Methods

This section will include the exploration results, preprocessing steps, models chosen in the order they were executed. You should also describe the parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, additional models are optional. Please note that models can be the same i.e. DNN but different versions of it if they are distinct enough. Changes can not be incremental. You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. 

## Data Exploration

Your report should include relevant figures of your choosing to help with the narration of your story, including legends (similar to a scientific paper). For reference you search machine learning and your model in google scholar for reference examples.  (3 points). You can also add figures of plots obtained after EDA.

The data that was leveraged in the project from the Million Song Dataset was the publicly available song metadata and a user-song interactions table. Prior to conducting data cleaning and preprocessing, the data was visualized in order to develop an initial understanding of the dataset. EDA helps reveal patterns and relationships within the data, allowing us to identify potential issues such as missing values, outliers, or incorrect data types. 

### The Track Metadata Dataframe

This dataframe was extracted by connecting to the SQLite track metadata database and extracting all records from the songs table. Duplicate entries were removed based on song_id and we added genre by merging the metadata with genre labels that came in a separate file. Using the describe function, we saw the dataframe has 999,056 rows and 16 columns. The numerical features in the dataframe are displayed below, which are: song duration, artist_familiarity, artist_hotness, year, track_id, shs_perf (the performance of the song in another dataset called second hand dataset), and shs_work (the identifier of the song in the second hand dataset). 

Immediately we noticed that features like artist familiarity, artist hotness and the shs performance score have minimum values of -1 or how year has a minimum value of 0 which could indicate missing values. So the distributions were plotted for further inspection.

It turns out that -1 is the missingness indicator for artist hotness and artist familiarity, there are only 12 unknown values for artist hotness and 180 unknown values for artist familiarity, this is good since the more known data we have the better the insights. For the year variable, there is a large amount of unknowns since a song cannot have a year of 0. We uncovered that 484,284 songs have a year of 0. If we remove 0s then the distribution is much clearer, showing songs ranging from the 1950s up until the early 2000s.

To explore the categorical labels, we looked at the genre labels via seaborn bar plots, which have majority and minority genres. The top majority and minority genre is rock, and the other most popular among both are electronic, pop. Metal is a more prevalent minority genre. 

We also explored genre missingness using seaborn pie plots and noticed a large portion of missing data

### The User-Song Dataframe

Using the describe function we saw the original dataframe has 48,373,586 rows and 3 columns, the columns are user_id, song_id and number of plays. We merge this dataframe with the track metadata dataframe from above to be able to gain more data on the songs users are playing. The dataframe has 384,546 unique songs and 1,019,318 unique users. 

We plotted the distribution of song plays, which showed a right skewed distribution, so the majority of the plays are smaller than larger. Additionally, we used a stripplot as well in order to capture the outliers that weren’t being captured in the histogram and we can see some very large play counts, the largest visible play count nearing 700,000. 

We looked at the genre distributions of the songs within the user-song dataframe with rock still heavily in the lead. 

For the user–song interaction dataframe, via a seaborn pie plot, it was observed that 61.7% of the songs had identifiable genre labels, which is higher than the proportion observed in the full track-metadata dataframe. This is attributed to the lower number of unique songs included in the interaction dataframe. 

In order to find out more about the artists that were being listened to in the dataset, we visualized the top 15 most listened to artists using a barplot. Noticing once again the popularity of rock. 

Other visualizations that were created were correlation heatmaps to see the correlation between numerical variables, if any. But we noticed very low correlations between most variables except artist hotness and artists similarity and shs perf and shs work which was expected. These visualizations and their code can be found in the milestone 2 plots notebook. 

## Data Preprocessing

The data preprocessing section focused on outputting two dataframes: a song dataframe with song information and non aggregated user interactions and a user level dataframe for clustering.

### The Song Dataframe

For the first dataframe, mirroring what happened in the data exploration section above, the data (track metadata, user-song data, genre labels) was loaded into the notebook. We then merged the track metadata with the genre labels on track id, then merged this with the user-song interaction data on song id. We then drop the rows with null genres and year as 0. We also created a new feature called total track time which multiplies the duration of a track with the total plays. This dataframe gets saved at the end of the notebook as a parquet file for easy access in subsequent steps. So what this is, is per every row we have a user id with a song they’ve listened to and all the song metadata. 

### The User Level Dataframe

We took the dataset above and created a new one with user level info, so we had to do some aggregations. We grouped by user_id, and we computed user listening statistics. For each user, we calculated their total number of plays, total listening time, number of unique songs, and number of unique artists. We also identified preference-based attributes by selecting the most frequently occurring values within each user’s listening history. These included their most-listened artist, favorite majority and minority genres, most common release year among the tracks they listened to, and their most-played song. Together, these aggregated features formed a sort of “profile” for each user and were used as inputs for the clustering algorithms. 

To prepare the user-level features for clustering, we dropped NaNs and then applied two different scaling methods to all numerical variables: Min-Max scaling and Robust scaling. Min–Max scaling transforms each feature to a 0–1 range, while Robust scaling standardizes values based on the median and interquartile range, making it less sensitive to extreme outliers. We computed both versions to compare their effects. Because the user-level features contained several skewed distributions and meaningful outliers like heavy listeners. Robust scaling was selected as the more appropriate normalization method for subsequent modeling. Because the dataset included some categorical features: user’s favorite majority genre, minority genre, and favorite artist. We converted these variables into numerical form using one-hot encoding. We saved the processed user dataset in Parquet format to enable efficient storage and future access. The encoded DataFrame was written to the shared ‘DSE 220 Final Project’ Google Drive folder.

For the next step of preprocessing, we loaded the processed user dataset from the Parquet file. Due to RAM issues, we had to sample 500,000 rows of data instead of working with the entire dataset and we also had to drop the artist one hot encoded columns that we created prior since it created thousands. Subsequently, to reduce dimensionality and capture the most relevant patterns in the data, we applied Principal Component Analysis (PCA) on the numerical features, excluding the user_id column as it was not relevant for clustering. We retained enough components to explain 90% of the variance in the data (set parameter n_components=0.9). The PCA transformation produced a new set of features, with each principal component represented as a separate column in a new DataFrame. Next, to determine the optimal number of clusters for our dataframe, we applied the elbow method using the PCA-transformed features. We evaluated k-means clustering across a range of cluster counts from 1 to 30 and recorded the corresponding inertia, which measures the within-cluster sum of squares. For each value of k, we also tracked the number of users assigned to each cluster. The resulting inertia values were plotted against the number of clusters via a line chart to visualize the "elbow point," which indicates the best number of clusters.

To better understand how the k-means algorithm distributed users across clusters at different values of k, we constructed a matrix that records the number of data points assigned to each cluster for every tested cluster count. This matrix was converted into a DataFrame, where each row represents a specific k value and each column corresponds to a cluster ID. We then visualized this distribution using two heatmaps. The first heatmap displays the raw counts of users in each cluster and the other is the log scaled heatmap. 

## Model 1: KMeans Clustering

To quickly explain, K-means clustering is an unsupervised machine learning method used to group data points into a number of clusters based on similarity. The algorithm works by assigning observations to the nearest cluster centroid and then updating the centroids to minimize the within-cluster variance. After identifying the optimal number of clusters, we decided to run the model with 9 and 26 clusters on separate iterations using scikit-learn's k-means algorithm. A new column was added to the user DataFrame to indicate each user’s cluster assignment. We evaluated the model by computing the silhouette score on a subset of 50,000 data points, due to the computational cost of evaluating the full dataset. Clusters were visualized across different principal components using pairplots and focused scatter plots, which are discussed further in the results section.

## Model 2: HDBSCAN

For our second approach, we applied UMAP (Uniform Manifold Approximation and Projection) and then a HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise). UMAP was used to reduce the data into 15-dimensions (set parameter n_component = 15), then UMAP looks at the 50 nearest neighbors of each point (set parameter n_neighbor = 50) and points cluster tightly(set parameter min_dist=0.0). We then applied HDBSCAN, a density-based clustering algorithm that automatically identifies clusters and labels low-density points as noise. The algorithm ran with the minimum number of points in a cluster for it to be considered valid at 200 (set parameter min_cluster_size=200), the number of samples needed to be considered a core point was configured at 10 (set min_samples=10) and we use all GPU to speed up computations (set n_jobs = -1). We assigned the resulting cluster labels to the user dataframe in a new column called hdb_cluster. We then inspected the clustering results by counting the number of unique clusters and examining the distribution of points across all clusters. To visualize how the clusters relate to basic listening metrics, we created a histogram of the HDBSCAN labels and a scatter plot comparing total_play_count and unique_artist_count against cluster assignments which we will further discuss in the results section. Lastly, we assign the HDBSCAN cluster numbers to each user in the user dataframe and drop unnecessary columns. The resulting dataframe has user_id, the KMeans cluster assignment and the HDBSCAN cluster assignment.

## Playlist Generation

To generate ‘playlists’ for each cluster, we first combined the two dataframes we ended up with, the song dataframe and the user clusters dataframe. This merge created a single dataset that included each user’s plays, the corresponding song attributes, and the cluster assignments from both KMeans and HDBSCAN. We then selected the relevant variables to keep. Next, we computed global popularity by summing the total number of plays for each song across all users. This provides context on whether a track is widely listened to overall or only within specific clusters.

For playlist creation, we aggregated the data separately for KMeans and HDBSCAN. For each clustering method, we grouped the data by cluster and song, summed the play counts within each cluster, and sorted songs in descending order of cluster-specific popularity. From these ranked lists, we selected the top 20 most-played songs for every cluster. Finally, we joined these top songs back with the song metadata. So we have 20 songs and their info for each cluster for each clustering algorithm. To evaluate this, we looked at specific user ids, created a dataframe with that user’s top 20 songs and compared that to the songs in the user’s assigned cluster playlist. We then further evaluated the clusters. 

---

# Results Section

This will include the results from the methods listed above (C). You will have figures here about your results as well. No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section. (5 points)

<!-- Leave blank for now -->

---

# Discussion Section

This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how to correctly scrutinize things and find short comings. 

<!-- Leave blank for now -->

---

# Conclusion

This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts. (3 points)

<!-- Leave blank for now -->

---

# Statement of Collaboration

This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. Every person should be listed in the following format:

**Name: Title: Contribution**  
If the person contributed nothing then just put in writing: Did not participate in the project. 

<!-- Leave blank for now -->
