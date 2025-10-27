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

Link to data exploration notebook (question 4):
https://colab.research.google.com/drive/1R78wYIx8ubgndlbqqMjH89J9ISlaoF28#scrollTo=LhDklquNoH-B

Link to data plots noteboook(question 5):
https://colab.research.google.com/drive/1EWx3bRGiWkc8kpmrDeuQ_j1jzZ2e1f5w#scrollTo=_Ty_supesYTh

Explanation to question 6:

In order to create some of the visuals we wanted to, we had to preprocess the data. We merged tables to bring in genre
info, user and song info. After looking more at the data we realize we need to bring in genre information some other way
since right now 70% of songs do not have a genre associated with them. We also aim to bring in lyric and audio features
from another file to further enrich our data. We could potentially extract meaning from lyrics to add a sentiment score
to songs and see how that affects popularity. Incorporating audio features will help us group songs better. Our year
column also has a lot of missing values, about half of the songs do not have a release year, so we may need to work on a
subset. Additionally we have missing values for artist hotness and artist familiarity so conducting imputation for the
missing data might be necessary. For the clustering part, we started some of the preprocessing since we wanted to
visualize our user data better, we want to cluster users together by the songs they listen to to create consumer
profiles so from the data we derived our users' favorite songs, genres, artist, and song year. 

