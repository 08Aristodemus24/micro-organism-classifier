import csv
import numpy as np
import tqdm
import pickle
import json


# for NLP data
def load_corpus(path: str):
    """
    reads a text file and returns the text
    """

    with open(path, 'r', encoding='utf-8') as file:
        corpus = file.read()

    return corpus

def get_chars(corpus: str):
    """
    returns a list of all unique characters found
    in given corpus
    """

    chars = sorted(list(set(corpus)))

    return chars

def load_lookup_array(path: str):
    """
    reads a text file containing a list of all unique values
    and returns this
    """

    with open(path, 'rb') as file:
        char_to_idx = pickle.load(file)

    return char_to_idx

def save_lookup_array(path: str, uniques: list):
    """
    saves and writes all the unique list of values to a
    a file for later loading by load_lookup_array()
    """

    with open(path, 'wb') as file:
        pickle.dump(uniques, file)

def save_meta_data(path: str, meta_data: dict):
    """
    saves dictionary of meta data such as hyper 
    parameters to a .json file
    """

    with open(path, 'w') as file:
        json.dump(meta_data, file)
        file.close()

def load_meta_data(path: str):
    """
    loads the saved dictionary of meta data such as
    hyper parameters from the created .json file
    """

    with open(path, 'r') as file:
        meta_data = json.load(file)
        file.close()

    return meta_data

def save_model(model, path: str):
    """
    saves partcularly an sklearn model in a .pkl file
    for later testing
    """

    with open(path, 'wb') as file:
        pickle.dump(model, file)
        file.close()

def load_model(path: str):
    """
    loads the sklearn model stored in a .pkl file
    for later testing
    """

    with open(path, 'rb') as file:
        model = pickle.load(file)
        file.close()

    return model

def construct_embedding_dict(emb_path):
    """
    returns an embedding dictionary populated with all the unique 
    values and their respective vector representations from the 
    file of the pretrained embeddings 

    creates the embedding dictionary from the text file containing
    the pre-trained embeddings which are in the format below:

    the 0.1234 0.423 -0.1324 ... -0.231
    hello 0.1234 0.423 -0.1324 ... -0.231
    nice 0.1234 0.423 -0.1324 ... -0.231
    """

    with open(emb_path, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONE)

        # map(float, line[1:]) takes the read line which is 
        # an array of the flaot values excluding the word and
        # passes each of its elements to the callback we 
        # provided which was `float`
        embedding_dict = {line[0]: np.array(list(map(float, line[1:]))) for line in reader}

    # gets the first item of the created dictionary
    first_key = next(iter(embedding_dict))

    # access the value or vector of the given the 
    # key and get its length
    emb_dim = embedding_dict[first_key].shape[0]

    return embedding_dict, emb_dim

def construct_embedding_matrix(val_to_index, embedding_dict, emb_dim):
    """
    Constructs the embedding matrix upon finishing the phase of 
    constructing the embedding dictionary. So that reading the
    embeddings is only done once to increase time efficiency
    """

    # oov words (out of vacabulary words) will be mapped to 0 vectors
    # this is why we have a plus one always to the number of our words in 
    # our embedding matrix since that is reserved for an unknown or OOV word
    vocab_len = len(val_to_index) + 1

    # initialize it to 0
    embedding_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in tqdm.tqdm(val_to_index.items()):
        # skip if, if index is already equal to the number of
        # words in our vocab. A break statement if you will
        if index < vocab_len:
            # if word does not exist in the pretrained word embedding itself
            # then return an empty array
            vector = embedding_dict.get(word, [])

            # if in cases vect is indeed otherwise an empty array due 
            # to the word existing in the pretrained word embeddings
            # then place it in our embedding matrix. Otherwise its index
            # where a word does not exist will stay a row of zeros
            if len(vector) > 0:
                embedding_matrix[index] = vector[:emb_dim]

    return embedding_matrix

def get_cat_cols(df):
    """
    returns all categorical columns/features names
    as a list
    """

    cols = df.columns

    num_cols = df._get_numeric_data().columns.to_list()

    # get complement of set of columns and numerical columns
    cat_cols = list(set(cols) - set(num_cols))
    
    return cat_cols

def get_top_models(models_train, models_cross, pool_size: int=10, model_type: str="regressor"):
    """
    takes in the dataframes returned by either LazyClassifier or LazyPredict
    e.g. clf = LazyRegressor(
        verbose=0, 
        ignore_warnings=True, 
        custom_metric=None, 
        regressors=[LinearRegression, Ridge, Lasso, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR])
    models_train, predictions_train = clf.fit(ch_X_trains, ch_X_trains, ch_Y_trains, ch_Y_trains)
    models_cross, predictions_cross = clf.fit(ch_X_trains, ch_X_cross, ch_Y_trains, ch_Y_cross)

    args:
        models_train - 
        models_cross - 
        pool_size - number of rows to take into consideration when merging the
        dataframes of model train and cross validation metric values
    """

    # rename columns for each dataframe to avoid duplication during merge
    for col in models_train.columns:
        models_train.rename(columns={f"{col}": f"Train {col}"}, inplace=True)
        models_cross.rename(columns={f"{col}": f"Cross {col}"}, inplace=True)

    # merge both first pool_size rows of training and cross 
    # validation model dataframes
    models_train = models_train[:pool_size].reset_index()
    models_cross = models_cross[:pool_size].reset_index()
    
    # merge model dataframes on 'Model' column
    top_models = models_train.merge(models_cross, how='inner', left_on='Model', right_on='Model')
    top_models.sort_values(by="Cross Adjusted R-Squared" if model_type == "regressor" else "Cross F1 Score")

    return top_models