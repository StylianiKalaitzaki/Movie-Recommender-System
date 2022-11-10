import sys
import math
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances 

def selectKMostSimilar(x, k, n):
    similarity_array = x.copy()
    k_most_similar = [0] * len(similarity_array)
    indexes = []
    similarity_array[n] = - float('inf') #assign similarity with itself as -infinity

    for i in range(k):
        maxi = np.argmax(similarity_array) #index of max
        similarity_array[maxi]= - float('inf') #replace max with a very small number
        k_most_similar[maxi] = 1
        indexes.append(maxi)

    return k_most_similar,indexes

def computeSimilarityMatrix(x,metric):
    ratings_matrix = x.copy()
    if metric == "jaccard":
        # Item Similarity Matrix using Jaccard similarity
        
        jaccard = pairwise_distances(ratings_matrix.astype(bool),ratings_matrix.astype(bool),metric="jaccard",n_jobs=-1)
        similarity_matrix = 1 - jaccard

    else: # adjusted cosine
        
        # Substract row mean from each element in row
        adjusted_matrix=ratings_matrix.sub(ratings_matrix.mean(axis=1), axis=0).fillna(0)

        # Similarity Matrix using Cosine similarity as a similarity measure between All Users/Items
        similarity_matrix = cosine_similarity(adjusted_matrix)
        similarity_matrix[np.isnan(similarity_matrix)] = 0
    
    return similarity_matrix

def computePredictedRatings(similarity_matrix, k_most_similar, ratings_table, bin_matrix):
    #similarity_matrix is sparse and contains similarities only in the positions of the k most similar
    #k_most_similar contains the indexes of the the k most similar
   
    predicted_ratings = np.dot(similarity_matrix, ratings_table)

    for i, x in enumerate(predicted_ratings):
        ar = []
        for j,y in enumerate(x):
            tmp = similarity_matrix[i].copy()
            for smlr in k_most_similar[i]:
            # e.g. if a similar user has not rated the movie j , the similarity will not be added on the denominator
                if bin_matrix[smlr,j] == 0 : 
                    tmp[smlr] = 0
            if sum(tmp)!=0:
                ar.append(y/sum(tmp))
            else:
                ar.append(0)
        predicted_ratings[i] = ar
    
    return predicted_ratings

def ratingsToBinary(ratings):
    bin_matrix = ratings.copy()
    bin_matrix['rating_x'] = bin_matrix['rating_x'].apply(lambda x: 1 if x > 0 else 0)
    bin_matrix = bin_matrix.pivot(index = 'movieId', columns = 'userId', values = 'rating_x').fillna(0)
    return bin_matrix.to_numpy().astype(int),bin_matrix.T.to_numpy().astype(int)

def S1(R,binary,dummy,metric):
    #R: table of ratings
    #dummy: test table with movieId as index

    print("Starting S1...")
    print("Computing S1 similarity...", end = ' ')
    if metric == "jaccard":
        item_similarity = computeSimilarityMatrix(binary,metric)
    else:
        item_similarity = computeSimilarityMatrix(R,metric)
    

    k_item_similarity = [] #binary
    k_most_similar_items = [] #indexes

    for i,x in enumerate(item_similarity):
        y,indexes = selectKMostSimilar(x,k,i)
        k_item_similarity.append(np.multiply(x, y))
        k_most_similar_items.append(indexes)
    print("Done")

    print("Computing S1 predictions...", end = ' ')
    predicted_ratings = computePredictedRatings(k_item_similarity, k_most_similar_items, R, binary)
    
    S1_final_ratings = np.multiply(predicted_ratings, dummy)
    print("Done")
    #print("************S1 Final Ratings************")
    #print(S1_final_ratings.T)

    # The movies with predicted rating (S1) > 2.5 are marked as 1 for prediction from S2
    X = S1_final_ratings.copy() 
    X = X[X > 2.5].fillna(0)
    X = X[X == 0].fillna(1)
    return X

def S2(R_NaN,R,binary,dummy,metric):
    ''' 
    dummy: a table with movieId as index (The result of S1). This function uses its transpose.
    R_NaN: the table of ratings with NaN values
    R: the table of ratings with 0s instead of NaN
    '''
    print("Starting S2...")
    
    print("Computing S2 similarity...", end = ' ')
    user_similarity = computeSimilarityMatrix(R_NaN,metric)
    

    k_user_similarity = [] #binary
    k_most_similar_users = [] #indexes

    for i,x in enumerate(user_similarity):
        y,indexes = selectKMostSimilar(x,k,i)
        k_user_similarity.append(np.multiply(x, y))
        k_most_similar_users.append(indexes)
    
    print("Done")

    print("Computing S2 predictions...", end = ' ')
    predicted_ratings_S2 = computePredictedRatings(k_user_similarity, k_most_similar_users, R, binary)
    
    S2_final_ratings = np.multiply(predicted_ratings_S2, dummy.T)

    print("Done")
    #print("************S2 Final Ratings************")
    #print(S2_final_ratings)
    return S2_final_ratings


### Calculates predictions for all unknown ratings in the original dataset before the split
def predictActualUnknownRatings(ratings,ratings_tableNaN,user_data,bin_matrix):
    
    d_ratings = ratings.copy()
    d_ratings['rating'] = d_ratings['rating'].apply(lambda x: 0 if x > 0 else 1)
    item_item_table = d_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(1)
    
    ### Here starts S1
    #item_item_table = S1(user_data.T,bin_matrix,item_item_table,"jaccard")
    item_item_table = S1(user_data.T,bin_matrix,item_item_table,"adjusted cosine")

    ### Here starts S2
    pred = S2(ratings_tableNaN,user_data,bin_matrix.T,item_item_table,"adjusted cosine")
    return pred



    
if len(sys.argv)!= 3:
    print("arguments: recommender_system.py <number of nearest neighbors> <train set percentage (e.g. 90, 80 etc)>")
    sys.exit()
k = int(sys.argv[1],base=10)
p = int(sys.argv[2],base=10)



# read file
ratings = pd.read_csv('data/ratings.csv')
ratings=ratings.drop(columns=['timestamp'])

# keep movies with at least 5 ratings
df=pd.DataFrame(ratings.groupby("movieId")["rating"].count().reset_index(name="Total Ratings"))
ratings=ratings.merge(df,on="movieId", how='inner')
ratings=ratings.loc[ratings["Total Ratings"]>4]

# prepare train and test datasets
X_train, X_test = train_test_split(ratings, random_state = 0,train_size = (p/100), test_size = .10)
user_data = pd.merge(X_train, X_test, how="outer", on=["userId","movieId"] )

bin_matrix,bin_matrixT = ratingsToBinary(user_data.drop(columns = ['rating_y']))

ratings_tableNaN =user_data.pivot(index = 'userId', columns = 'movieId', values = 'rating_x')
user_data = user_data.pivot(index = 'userId', columns = 'movieId', values = 'rating_x').fillna(0)

# copy of train and test datasets
dummy_train = X_train.copy()
dummy_test = X_test.copy()

dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)


right_merged = pd.merge(dummy_train, dummy_test, how="outer", on=["userId","movieId"] )
test_item_item_table = right_merged.pivot(index = 'movieId', columns = 'userId', values = 'rating_y').fillna(0)

'''
preds = predictActualUnknownRatings(ratings,ratings_tableNaN,user_data,bin_matrix)
print(preds)
'''

### Evaluation
print("******************** Evaluation ********************")

### **** S1 ****

#test_item_item_table = S1(user_data.T,bin_matrix,test_item_item_table,"adjusted cosine")
test_item_item_table = S1(user_data.T,bin_matrix,test_item_item_table,"jaccard")

### **** S2 ****
'''S2 takes as input the item-item table but uses the transpose'''
test_user_final_rating = S2(ratings_tableNaN,user_data,bin_matrixT,test_item_item_table,"adjusted cosine")


n = np.count_nonzero(test_user_final_rating)
test = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating')

# RMSE Score
test_user_final_rating = test_user_final_rating[test_user_final_rating > 0]

diff_sqr_matrix = (test - test_user_final_rating)**2
sum_of_squares_err = diff_sqr_matrix.sum().sum()

rmse = np.sqrt(sum_of_squares_err/n)
print("RMSE: ",rmse)

# Mean abslute error
mae = np.abs(test_user_final_rating - test).sum().sum()/n
print("MAE: ",mae)


#Total number of relevant items(>3.5) in Test Set
total_relevant_items = X_test['rating'][X_test['rating'] > 3.5].count()

# The items in the Test Set that are relevant(>3.5) are marked as 1
actual_ratings = pd.merge(X_train, X_test, how="outer", on=["userId","movieId"] )
actual_ratings = actual_ratings.pivot(index = 'userId', columns = 'movieId', values = 'rating_y').fillna(0)

Y = actual_ratings.copy() 
Y = Y[Y > 3.5].fillna(0)
Y = Y[Y == 0].fillna(1)


# The items in the Predicted Set that are relevant(>3.5) are marked as 1
Z = test_user_final_rating.copy() 
Z = Z[Z > 3.5].fillna(0)
Z = Z[Z == 0].fillna(1)

recom_relevant_items = np.count_nonzero(np.multiply(Z, Y).to_numpy())
total_recom_items = np.count_nonzero(Z.to_numpy())


if total_relevant_items != 0:
    recall = recom_relevant_items / total_relevant_items
else:
    recall = 0
print("Recall: ",recall)
if total_recom_items != 0:
    precision = recom_relevant_items / total_recom_items
else:
    precision = 0
print("Precision: ",precision)
