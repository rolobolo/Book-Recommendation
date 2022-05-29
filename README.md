### Book Recommendation: Hybrid(collaborative+content)
# Book-Recommendation System

This system recommends books to user by combining percentile score of content and collaborative filtering algorithm.


## Dataset Used

 - [Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
 


## Algorithms 

#### Popular in the Whole Collection: 
Sorted the dataset according to mean ratings each of the books have received in non-increasing order and then recommended top n books.
#### User-Item Collaborative Filtering Recommendation:
Collaborative Filtering Recommendation System works by considering user ratings and finds cosine similarities in ratings by several users to recommend books. To implement this, took only those books' data that have at least 50 ratings in all.
#### Content Based Recommendation: 
This system recommends books by calculating similarities in Book Titles. For this, TF-IDF feature vectors were created for unigrams and bigrams of Book-Titles; only those books' data has been considered which are having at least 80 ratings.
#### Hybrid Approach (Collaborative+Content) Recommendation: 
A hybrid recommendation system was built using the combination of both content-based filtering and collaborative filtering systems. A percentile score is given to the results obtained from both content and collaborative filtering models and is combined to recommend top n books.
