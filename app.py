import operator
import pickle
from typing import List, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


def getTopRecommendations(bookID, Numbers):
    collaborative = []
    row = reverseIndexMap[bookID]
    # print("Input Book:")
    # printBookDetails(bookID)

    # print("\nRECOMMENDATIONS:\n")

    mn = 0
    similar = []
    for i in np.argsort(pairwiseSimilarity[row])[:-2][::-1]:
        if books[books['ISBN'] == indexMap[i]]['Book-Title'].values[0] not in similar:
            if mn >= Numbers:
                break
            mn += 1
            similar.append(books[books['ISBN'] == indexMap[i]]['Book-Title'].values[0])
            # printBookDetails(indexMap[i])
            collaborative.append(books[books['ISBN'] == indexMap[i]]['Book-Title'].values[0])
    return collaborative


def recommend(selected_book_name, Numbers):
    normalized_df = tfidf_matrix.astype(np.float32)
    cosine_similarities = cosine_similarity(normalized_df, normalized_df)
    isbn = books.loc[books['Book-Title'] == selected_book_name].reset_index(drop=True).iloc[0]['ISBN']
    content = []

    idx = popular_book.index[popular_book['ISBN'] == isbn].tolist()[0]
    similar_indices = cosine_similarities[idx].argsort()[::-1]
    similar_items = []
    for i in similar_indices:
        if popular_book['Book-Title'][i] != selected_book_name and popular_book['Book-Title'][
            i] not in similar_items and len(
                similar_items) < Numbers:
            similar_items.append(popular_book['Book-Title'][i])
            content.append(popular_book['Book-Title'][i])
    k = list(books['Book-Title'])
    m = list(books['ISBN'])
    collaborative = getTopRecommendations(m[k.index(selected_book_name)], Numbers)
    z = list()
    K = float(1 / Numbers)
    for x in range(Numbers):
        z.append(1 - K * x)

    dictISBN = {}
    for x in collaborative:
        dictISBN[x] = z[collaborative.index(x)]

    for x in content:
        if x not in dictISBN:
            dictISBN[x] = z[content.index(x)]
        else:
            dictISBN[x] += z[content.index(x)]

    ISBN = dict(sorted(dictISBN.items(), key=operator.itemgetter(1), reverse=True))
    recommend_books = []
    w = 0
    for x in ISBN.keys():
        if w >= Numbers:
            break
        w += 1
        recommend_books.append(x)
    return recommend_books


popular_book = pickle.load(open('popular_book.pkl', 'rb'))
populardf = pickle.load(open('populardf.pkl', 'rb'))
df_popular = pickle.load(open('df_popular.pkl', 'rb'))
k = pickle.load(open('k.pkl', 'rb'))
m = pickle.load(open('m.pkl', 'rb'))
P = pickle.load(open('P.pkl', 'rb'))
st.title('Book Recommender System')

tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
books_dict = pickle.load(open('books_dict.pkl', 'rb'))
reverseIndexMap = pickle.load(open('reverseIndexMap.pkl', 'rb'))
pairwiseSimilarity = pickle.load(open('pairwiseSimilarity.pkl', 'rb'))
indexMap = pickle.load(open('indexMap.pkl', 'rb'))
books = pd.DataFrame(books_dict)
Books = pickle.load(open('popular_book.pkl', 'rb'))


selected_book_name = st.selectbox('Select Book:', popular_book['Book-Title'].values)


r_input = st.number_input('No of Recommendations:', min_value=1, max_value=50, step=1)

if st.button('Show Recommendation'):
    recommendations = recommend(selected_book_name, r_input)
    # recommendations = recommend('Animal Farm')
    for i in recommendations:
        st.write(i)


book_name = list(populardf['Book-Title'].values)
votes = list(populardf['num_ratings'].values)
ratings = list(populardf['avg_ratings'].values)
url = list(df_popular['Image-URL-M'].values)
for i in range(22):
    ratings[i] = "{:.2f}".format(ratings[i])
st.header('Top Popular Books')
col = st.columns(5)
for i in range(5):
    with col[i]:
        st.subheader(book_name[i])
        st.write('Votes:', votes[i], 'Ratings:', ratings[i])
        st.image(url[i])
col1 = st.columns(5)
for i in range(5):
    with col1[i]:
        st.subheader(book_name[i + 5])
        st.write('Votes:', votes[i + 5], 'Ratings:', ratings[i + 5])
        st.image(url[i+5])
col2 = st.columns(5)
for i in range(5):
    with col2[i]:
        st.subheader(book_name[i + 10])
        st.write('Votes:', votes[i + 10], 'Ratings:', ratings[i + 10])
        st.image(url[i+10])

# FEEDBACK
st.sidebar.title('Feedback')
rating = 'Null'
rating = st.sidebar.selectbox('Are You Happy with the Example', ('Yes', 'No', 'Not Sure'))
if rating == 'Yes':
    st.sidebar.success('Thank You for Selecting Yes')
elif rating == 'No':
    st.sidebar.info('Sincere Apologies')


elif rating == 'Not Sure':
    st.sidebar.info('')



st.sidebar.write('Rate Your Experience')
get_number = st.sidebar.slider("Select a Number", 1, 10)
st.sidebar.write('Rated', get_number)
