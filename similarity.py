# -------------------------------------------------------------------------
# AUTHOR: Jasmit Mahajan
# FILENAME: similarity.py
# SPECIFICATION: This program will get most similar documents and terms and will render the highest cosine similarity.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Defining the documents
doc1 = "Soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "I support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document matrix
# [soccer, my, favorite, sport, I, like, one, support, olympic, game]
terms_list = np.array(['soccer', 'my', 'favorite', 'sport', 'I', 'like', 'one', 'support', 'olympic', 'games'])
docs_list = np.array([doc1, doc2, doc3, doc4])
vec = CountVectorizer(vocabulary=terms_list, analyzer="word")
X = vec.fit_transform(docs_list)
df = X.toarray()

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here

cos_sim = cosine_similarity(df)
max_sim = 0
max_pair = [-1, -1]
for i in range(len(cos_sim)):
    for j in range(len(cos_sim[i])):
        if i != j:
            temp_var = max_sim
            max_sim = max(max_sim, cos_sim[i][j])
            if temp_var < max_sim:
                max_pair = [i, j]

# Print the highest cosine similarity following the template below
# The most similar documents are: doc1 and doc2 with cosine similarity = x

# --> Add your Python code here
print("The most similar documents are doc" + str(max_pair[0]) + " and doc" + str(max_pair[1]) + " with cosine similarity = " + str(max_sim))