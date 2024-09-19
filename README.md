The project aims to perform customer segmentation using unsupervised learning methods, specifically hierarchal clustering. 
The dataset consists of customer demographics, including gender and city, which will be used along with additional transactional features to cluster customers into meaningful groups.
The steps for this project :
1- Extracting Data : ( read - head - tail - info - describe )
2- preprocessing : ( datetime format - recency - frequency - Merge all features - remove duplicate values - replace null vaules ) ,Numerical features ( transaction_count, total_coupons_burnt) are standardized using StandardScaler. Categorical features (city_id, gender_id) are transformed into one-hot encoded variables using OneHotEncoder. These transformations are applied using a fit Transform within a Pipeline to ensure consistency across training and prediction phases
3- Hierarchical clustering is a method of cluster analysis that seeks to build a hierarchy of clusters. 
 Agglomerative Clustering:
    This is the most common type. It starts with each data point as a separate cluster and iteratively merges the closest pairs of clusters until only one cluster remains or a specified number of clusters is reached.
        Steps:
        Calculate the distance between each pair of clusters.
        Merge the two closest clusters.
        Repeat until the desired number of clusters is achieved.
4- Compute the pairwise distances between observations
5- Linkage 
  The way distances between clusters are calculated affects the results. Common linkage criteria include:
        Single Linkage: Distance between the closest points of two clusters.
        Complete Linkage: Distance between the furthest points of two clusters.
        Average Linkage: Average distance between all points in two clusters.
        Ward's Linkage: Minimizes the total within-cluster variance.
6-  Evaluation : Silhouette Score:  providing insight into the compactness and separation of clusters
7- Segment Analysis : the optimal number of clusters is determined, each data point (customer_id). Analysts can then perform segment analysis to uncover characteristics of each group.
8- visualization :To display the relationship between segment and feature selection
