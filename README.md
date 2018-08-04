# Recommender-System

## Project:Movie Recommender System using Stacked Auto Encoder

## Install

This project requires Python and the following Python libraries installed:

   - NumPy
   - Pandas
   - torch

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/#linux). distribution of Python, which already has the above packages and more included. 
you will also need to have pytorch deep learning framework for this project and you install the [pytorch](https://pytorch.org/). 

## Code

Template code is provided in the recommender_system.py or recommender_system.ipynb files. You will also be required to use  the 'movies.dat',users.dat,and ratings.dat dataset files to complete your work. you can access these dataset in 'ml-1m' folder.

## Run

In a terminal or command window, and run one of the following commands:

python recommender_system.py

or

jupyter notebook recommender_system.ipynb

This will open the Jupyter Notebook software and project file in your browser.

## Data

The main aim of this project is to predict the rating a user would give a movie. For this purpose we will use the famous MovieLens dataset. [MovieLens](https://grouplens.org/datasets/movielens/) is a web based recommender system and online community that recommends movies for its users to watch.

More specifically we will use the ml-1m.zip dataset that contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users. The import file we need is ratings.dat. This file contains 1,000,209 lines all having the following format: user_id::movie_id::rating:time_stamp.

For example the first line in ratings.dat:

1::595::5::978824268 

means that user Nr. 1 gave movie Nr. 595 a five star rating. The time stamp can be ignored because it won’t be used.

You should read the README file in 'm1-1m' file folder for further explanation of the dataset.

## Implementation

read more about Autoencoder here [guide](https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798) 

