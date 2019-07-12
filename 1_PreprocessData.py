#!/usr/bin/env python
# coding: utf-8

# # Preprocess Data
# This notebook contains materials to parse raw python files into function and docstring pairs, tokenize both function and dosctring into tokens, and split these pairs into a train, valid and test set.  
# 
# *This step is optional, as we provide links to download pre-processed data at various points in the tutorial.  However, you might find it useful to go through these steps in order to understand how the data is prepared.*
# 
# If you are using the recommended approach of using a `p3.8xlarge` instance for this entire tutorial you can use this docker container to run this notebook: [hamelsmu/ml-gpu](https://hub.docker.com/r/hamelsmu/ml-gpu/).
# 
# Alternatively, if you wish to speed up *this notebook* by using an instance with lots of cores (because everything in this notebook is CPU bound), you can use this container [hamelsmu/ml-cpu](https://hub.docker.com/r/hamelsmu/ml-gpu/).

# In[1]:


import ast
import glob
import re
from pathlib import Path

import astor
import pandas as pd
import spacy
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

from general_utils import apply_parallel, flattenlist

EN = spacy.load('en')


# ## Download and read  raw python files
# 
# The first thing we will want to do is to gather python code.  There is an open dataset that Google hosts on [BigQuery](https://cloud.google.com/bigquery/) that has code from open source projects on Github.  You can use [bigquery](https://cloud.google.com/bigquery/) to get the python files as a tabular dataset by executing the following SQL query in the bigquery console:
# 
# ```{sql}
# SELECT 
#  max(concat(f.repo_name, ' ', f.path)) as repo_path,
#  c.content
# FROM `bigquery-public-data.github_repos.files` as f
# JOIN `bigquery-public-data.github_repos.contents` as c on f.id = c.id
# JOIN (
#       --this part of the query makes sure repo is watched at least twice since 2017
#       SELECT repo FROM(
#         SELECT 
#           repo.name as repo
#         FROM `githubarchive.year.2017` WHERE type="WatchEvent"
#         UNION ALL
#         SELECT 
#           repo.name as repo
#         FROM `githubarchive.month.2018*` WHERE type="WatchEvent"
#         )
#       GROUP BY 1
#       HAVING COUNT(*) >= 2
#       ) as r on f.repo_name = r.repo
# WHERE 
#   f.path like '%.py' and --with python extension
#   c.size < 15000 and --get rid of ridiculously long files
#   REGEXP_CONTAINS(c.content, r'def ') --contains function definition
# group by c.content
# ```
# 
# 
# Here is a link to the [SQL Query](https://bigquery.cloud.google.com/savedquery/506213277345:009fa66f301240e5ad9e4006c59a4762) incase it is helpful.  The raw data contains approximate 1.2 million distinct python code files.
# 
# **To make things easier for this tutorial, the folks on the Google [Kubeflow team](https://kubernetes.io/blog/2017/12/introducing-kubeflow-composable/) have hosted the raw data for this tutorial in the form of 10 csv files, available at the url: https://storage.googleapis.com/kubeflow-examples/code_search/raw_data/00000000000{i}.csv as illustrated in the below code:**

# In[3]:


df = pd.read_csv(f'https://storage.googleapis.com/kubeflow-examples/code_search/raw_data/000000000001.csv')
# df = pd.read_csv('./data/github.csv')

df['nwo'] = df['repo_path'].apply(lambda r: r.split()[0])
df['path'] = df['repo_path'].apply(lambda r: r.split()[1])
df.drop(columns=['repo_path'], inplace=True)
df = df[['nwo', 'path', 'content']]
df.head()

df = df[0:7000, :]

# In[4]:


# Inspect shape of the raw data


# ## Functions to parse data and tokenize
# 
# Our goal is to parse the python files into (code, docstring) pairs.  Fortunately, the standard library in python comes with the wonderful [ast](https://docs.python.org/3.6/library/ast.html) module which helps us extract code from files as well as extract docstrings.  
# 
# We also use the [astor](http://astor.readthedocs.io/en/latest/) library to strip the code of comments by doing a round trip of converting the code to an [AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree) and then from AST back to code. 

# In[5]:


def tokenize_docstring(text):
    "Apply tokenization using spacy to docstrings."
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]


def tokenize_code(text):
    "A very basic procedure for tokenizing code strings."
    return RegexpTokenizer(r'\w+').tokenize(text)


def get_function_docstring_pairs(blob):
    "Extract (function/method, docstring) pairs from a given code blob."
    pairs = []
    try:
        module = ast.parse(blob)
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        for _class in classes:
            functions.extend([node for node in _class.body if isinstance(node, ast.FunctionDef)])

        for f in functions:
            source = astor.to_source(f)
            docstring = ast.get_docstring(f) if ast.get_docstring(f) else ''
            function = source.replace(ast.get_docstring(f, clean=False), '') if docstring else source

            pairs.append((f.name,
                          f.lineno,
                          source,
                          ' '.join(tokenize_code(function)),
                          ' '.join(tokenize_docstring(docstring.split('\n\n')[0]))
                         ))
    except (AssertionError, MemoryError, SyntaxError, UnicodeEncodeError):
        pass
    return pairs


def get_function_docstring_pairs_list(blob_list):
    """apply the function `get_function_docstring_pairs` on a list of blobs"""
    return [get_function_docstring_pairs(b) for b in blob_list]


# The below convience function `apply_parallel` parses the code in parallel using process based threading.  Adjust the `cpu_cores` parameter accordingly to your system resources!

# In[6]:


pairs = flattenlist(apply_parallel(get_function_docstring_pairs_list, df.content.tolist(), cpu_cores=32))


# In[7]:


assert len(pairs) == df.shape[0], f'Row count mismatch. `df` has {df.shape[0]:,} rows; `pairs` has {len(pairs):,} rows.'
df['pairs'] = pairs
df.head()


# ## Flatten code, docstring pairs and extract meta-data

# Flatten (code, docstring) pairs

# In[8]:


# flatten pairs
df = df.set_index(['nwo', 'path'])['pairs'].apply(pd.Series).stack()
df = df.reset_index()
df.columns = ['nwo', 'path', '_', 'pair']


# Extract meta-data and format dataframe.  
# 
# We have not optimized this code.  Pull requests are welcome!

# In[9]:



df['function_name'] = df['pair'].apply(lambda p: p[0])
df['lineno'] = df['pair'].apply(lambda p: p[1])
df['original_function'] = df['pair'].apply(lambda p: p[2])
df['function_tokens'] = df['pair'].apply(lambda p: p[3])
df['docstring_tokens'] = df['pair'].apply(lambda p: p[4])
df = df[['nwo', 'path', 'function_name', 'lineno', 'original_function', 'function_tokens', 'docstring_tokens']]
df['url'] = df[['nwo', 'path', 'lineno']].apply(lambda x: 'https://github.com/{}/blob/master/{}#L{}'.format(x[0], x[1], x[2]), axis=1)
df.head()


# ## Remove Duplicates

# In[10]:



# remove observations where the same function appears more than once
before_dedup = len(df)
df = df.drop_duplicates(['original_function', 'function_tokens'])
after_dedup = len(df)

print(f'Removed {before_dedup - after_dedup:,} duplicate rows')


# In[11]:




# ## Separate function w/o docstrings

# In[13]:


def listlen(x):
    if not isinstance(x, list):
        return 0
    return len(x)

# separate functions w/o docstrings
# docstrings should be at least 3 words in the docstring to be considered a valid docstring

with_docstrings = df[df.docstring_tokens.str.split().apply(listlen) >= 3]
without_docstrings = df[df.docstring_tokens.str.split().apply(listlen) < 3]


# ## Partition code by repository to minimize leakage between train, valid & test sets. 
# Rough assumption that each repository has its own style.  We want to avoid having code from the same repository in the training set as well as the validation or holdout set.

# In[14]:


grouped = with_docstrings.groupby('nwo')


# In[15]:


# train, valid, test splits
train, test = train_test_split(list(grouped), train_size=0.87, shuffle=True, random_state=8081)
train, valid = train_test_split(train, train_size=0.82, random_state=8081)


# In[16]:


train = pd.concat([d for _, d in train]).reset_index(drop=True)
valid = pd.concat([d for _, d in valid]).reset_index(drop=True)
test = pd.concat([d for _, d in test]).reset_index(drop=True)


# In[17]:


print(f'train set num rows {train.shape[0]:,}')
print(f'valid set num rows {valid.shape[0]:,}')
print(f'test set num rows {test.shape[0]:,}')
print(f'without docstring rows {without_docstrings.shape[0]:,}')


# Preview what the training set looks like.  You can start to see how the data looks, the function tokens and docstring tokens are what will be fed downstream into the models.  The other information is important for diagnostics and bookeeping.

# In[18]:


train.head()


# ## Output each set to train/valid/test.function/docstrings/lineage files
# Original functions are also written to compressed json files. (Raw functions contain `,`, `\t`, `\n`, etc., it is less error-prone using json format)
# 
# `{train,valid,test}.lineage` are files that contain a reference to the original location where the code was retrieved. 

# In[39]:


def write_to(df, filename, path='./data/processed_data/'):
    "Helper function to write processed files to disk."
    out = Path(path)
    out.mkdir(exist_ok=True)
    df.function_tokens.to_csv(out/'{}.function'.format(filename), index=False)
    df.original_function.to_json(out/'{}_original_function.json.gz'.format(filename), orient='values', compression='gzip')
    if filename != 'without_docstrings':
        df.docstring_tokens.to_csv(out/'{}.docstring'.format(filename), index=False)
    df.url.to_csv(out/'{}.lineage'.format(filename), index=False)


# In[40]:


# write to output files
write_to(train, 'train')
write_to(valid, 'valid')
write_to(test, 'test')
write_to(without_docstrings, 'without_docstrings')


# ## The pre-processed data is also hosted on Google Cloud, at the following URLs:

# In[24]:


# # cool trick to send shell command results into a python variable in a jupyter notebook!
# files = ! ls ./data/processed_data/ | grep -E '*.function$|*.docstring$|*.lineage$|*_original_function.json.gz$'

# # print the urls
# urls = [f'https://storage.googleapis.com/kubeflow-examples/code_search/data/{f}' for f in files]
# for s in urls:
#     print(s)

