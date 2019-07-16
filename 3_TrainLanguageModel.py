#!/usr/bin/env python
# coding: utf-8

# Environment is available as a publically available docker container: `hamelsmu/ml-gpu`
# 
# ### Pre-Requisite: Make Sure you have the right files prepared from Step 1
# 
# You should have these files in the root of the `./data/processed_data/` directory:
# 
# 1. `{train/valid/test.function}` - these are python function definitions tokenized (by space), 1 line per function.
# 2. `{train/valid/test.docstring}` - these are docstrings that correspond to each of the python function definitions, and have a 1:1 correspondence with the lines in *.function files.
# 3. `{train/valid/test.lineage}` - every line in this file contains a link back to the original location (github repo link) where the code was retrieved.  There is a 1:1 correspondence with the lines in this file and the other two files. This is useful for debugging.
# 
# 
# ### Set the value of `use_cache` appropriately.  
# 
# If `use_cache = True`, data will be downloaded where possible instead of re-computing.  However, it is highly recommended that you set `use_cache = False` for this tutorial as it will be less confusing, and you will learn more by runing these steps yourself.  This notebook was run on AWS on a `p3.8xlarge` in approximately 8 hours.

# In[5]:


# # Optional: you can set what GPU you want to use in a notebook like this.  
# # Useful if you want to run concurrent experiments at the same time on different GPUs.
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[2]:


use_cache = False


# In[ ]:


## Download pre-processed data if you want to run from tutorial from this step.##
from general_utils import get_step2_prerequisite_files

if use_cache:
    get_step2_prerequisite_files(output_directory = './data/processed_data')


# # Build Language Model From Docstrings
# 
# The goal is to build a language model using the docstrings, and use that language model to generate an embedding for each docstring.  

# In[4]:


import torch,cv2
from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
from general_utils import save_file_pickle, load_file_pickle
import logging
from pathlib import Path
from fastai.text import *

source_path = Path('./data/processed_data/')

with open(source_path/'train.docstring', 'r') as f:
    trn_raw = f.readlines()

with open(source_path/'valid.docstring', 'r') as f:
    val_raw = f.readlines()
    
with open(source_path/'test.docstring', 'r') as f:
    test_raw = f.readlines()


# Preview what the raw data looks like: here are 10 docstrings

# In[52]:


trn_raw[:10]


# ## Pre-process data for language model
# 
# We will use the class  `build_lm_vocab` to prepare our data for the language model

# In[3]:


vocab = lm_vocab(max_vocab=50000,
                 min_freq=10)

# fit the transform on the training data, then transform
trn_flat_idx = vocab.fit_transform_flattened(trn_raw)


# Look at the transformed data

# In[55]:


trn_flat_idx[:10]


# In[56]:


[vocab.itos[x] for x in trn_flat_idx[:10]]


# In[57]:


# apply transform to validation data
val_flat_idx = vocab.transform_flattened(val_raw)


# Save files for later use

# In[60]:


if not use_cache:
    vocab.save('./data/lang_model/vocab_v2.cls')
    save_file_pickle('./data/lang_model/trn_flat_idx_list.pkl_v2', trn_flat_idx)
    save_file_pickle('./data/lang_model/val_flat_idx_list.pkl_v2', val_flat_idx)


# ## Train Fast.AI Language Model
# 
# This model will read in files that were created and train a [fast.ai](https://github.com/fastai/fastai/tree/master/fastai) language model.  This model learns to predict the next word in the sentence using fast.ai's implementation of [AWD LSTM](https://github.com/salesforce/awd-lstm-lm).  
# 
# The goal of training this model is to build a general purpose feature extractor for text that can be used in downstream models.  In this case, we will utilize this model to produce embeddings for function docstrings.

# In[16]:


vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
trn_flat_idx = load_file_pickle('./data/lang_model/trn_flat_idx_list.pkl_v2')
val_flat_idx = load_file_pickle('./data/lang_model/val_flat_idx_list.pkl_v2')


# In[17]:


if not use_cache:
    fastai_learner, lang_model = train_lang_model(model_path = './data/lang_model_weights_v2',
                                                  trn_indexed = trn_flat_idx,
                                                  val_indexed = val_flat_idx,
                                                  vocab_size = vocab.vocab_size,
                                                  lr=3e-3,
                                                  em_sz= 500,
                                                  nh= 500,
                                                  bptt=20,
                                                  cycle_len=1,
                                                  n_cycle=3,
                                                  cycle_mult=2,
                                                  bs = 200,
                                                  wd = 1e-6)
    
elif use_cache:    
    logging.warning('Not re-training language model because use_cache=True')


# In[18]:


if not use_cache:
    fastai_learner.fit(1e-3, 3, wds=1e-6, cycle_len=2)


# In[19]:


if not use_cache:
    fastai_learner.fit(1e-3, 2, wds=1e-6, cycle_len=3, cycle_mult=2)


# In[20]:


if not use_cache:
    fastai_learner.fit(1e-3, 2, wds=1e-6, cycle_len=3, cycle_mult=10)


# Save language model and learner

# In[21]:


if not use_cache:
    fastai_learner.save('lang_model_learner_v2.fai')
    lang_model_new = fastai_learner.model.eval()
    torch.save(lang_model_new, './data/lang_model/lang_model_gpu_v2.torch')
    torch.save(lang_model_new.cpu(), './data/lang_model/lang_model_cpu_v2.torch')


# # Load Model and Encode All Docstrings
# 
# Now that we have trained the language model, the next step is to use the language model to encode all of the docstrings into a vector. 

# ** Note that checkpointed versions of the language model artifacts are available for download: **
# 
# 1. `lang_model_cpu_v2.torch` : https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model/lang_model_cpu_v2.torch 
# 2. `lang_model_gpu_v2.torch` : https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model/lang_model_gpu_v2.torch
# 3. `vocab_v2.cls` : https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model/vocab_v2.cls

# In[59]:


from lang_model_utils import load_lm_vocab
vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
idx_docs = vocab.transform(trn_raw + val_raw, max_seq_len=30, padding=False)
lang_model = torch.load('./data/lang_model/lang_model_gpu_v2.torch', 
                        map_location=lambda storage, loc: storage)


# In[60]:


lang_model.eval()


# **Note:** the below code extracts embeddings for docstrings one docstring at a time, which is very inefficient.  Ideally, you want to extract embeddings in batch but account for the fact that you will have padding, etc. when extracting the hidden states.  For this tutorial, we only provide this minimal example, however you are welcome to improve upon this and sumbit a PR!

# In[61]:


def list2arr(l):
    "Convert list into pytorch Variable."
    return V(np.expand_dims(np.array(l), -1)).cpu()

def make_prediction_from_list(model, l):
    """
    Encode a list of integers that represent a sequence of tokens.  The
    purpose is to encode a sentence or phrase.

    Parameters
    -----------
    model : fastai language model
    l : list
        list of integers, representing a sequence of tokens that you want to encode

    """
    arr = list2arr(l)# turn list into pytorch Variable with bs=1
    model.reset()  # language model is stateful, so you must reset upon each prediction
    hidden_states = model(arr)[-1][-1] # RNN Hidden Layer output is last output, and only need the last layer

    #return avg-pooling, max-pooling, and last hidden state
    return hidden_states.mean(0), hidden_states.max(0)[0], hidden_states[-1]


def get_embeddings(lm_model, list_list_int):
    """
    Vectorize a list of sequences List[List[int]] using a fast.ai language model.

    Paramters
    ---------
    lm_model : fastai language model
    list_list_int : List[List[int]]
        A list of sequences to encode

    Returns
    -------
    tuple: (avg, mean, last)
        A tuple that returns the average-pooling, max-pooling over time steps as well as the last time step.
    """
    n_rows = len(list_list_int)
    n_dim = lm_model[0].nhid
    avgarr = np.empty((n_rows, n_dim))
    maxarr = np.empty((n_rows, n_dim))
    lastarr = np.empty((n_rows, n_dim))

    for i in tqdm_notebook(range(len(list_list_int))):
        avg_, max_, last_ = make_prediction_from_list(lm_model, list_list_int[i])
        avgarr[i,:] = avg_.data.numpy()
        maxarr[i,:] = max_.data.numpy()
        lastarr[i,:] = last_.data.numpy()

    return avgarr, maxarr, lastarr


# In[64]:


avg_hs, max_hs, last_hs = get_embeddings(lang_model, idx_docs)


# ### Do the same thing for the test set

# In[63]:


idx_docs_test = vocab.transform(test_raw, max_seq_len=30, padding=False)
avg_hs_test, max_hs_test, last_hs_test = get_embeddings(lang_model, idx_docs_test)


# # Save Language Model Embeddings For Docstrings

# In[65]:


savepath = Path('./data/lang_model_emb/')
np.save(savepath/'avg_emb_dim500_v2.npy', avg_hs)
np.save(savepath/'max_emb_dim500_v2.npy', max_hs)
np.save(savepath/'last_emb_dim500_v2.npy', last_hs)


# In[64]:


# save the test set embeddings also
np.save(savepath/'avg_emb_dim500_test_v2.npy', avg_hs_test)
np.save(savepath/'max_emb_dim500_test_v2.npy', max_hs_test)
np.save(savepath/'last_emb_dim500_test_v2.npy', last_hs_test)


# ** Note that the embeddings saved to disk above have also been cached and are are available for download: **
# 
# Train + Validation docstrings vectorized:
# 
# 1. `avg_emb_dim500_v2.npy` : https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model_emb/avg_emb_dim500_v2.npy
# 2. `max_emb_dim500_v2.npy` : https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model_emb/last_emb_dim500_v2.npy
# 3. `last_emb_dim500_v2.npy` : https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model_emb/max_emb_dim500_v2.npy
# 
# Test set docstrings vectorized:
# 
# 1. `avg_emb_dim500_test_v2.npy`: https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model_emb/avg_emb_dim500_test_v2.npy
# 
# 2. `max_emb_dim500_test_v2.npy`: https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model_emb/last_emb_dim500_test_v2.npy
# 
# 3. `last_emb_dim500_test_v2.npy`: https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model_emb/max_emb_dim500_test_v2.npy

# # Evaluate Sentence Embeddings
# 
# One popular way of evaluating sentence embeddings is to measure the efficacy of these embeddings in downstream tasks like sentiment analysis, textual similarity etc.  Usually you can use general-purpose benchmarks such as the examples outlined [here](https://github.com/facebookresearch/SentEval) to measure the quality of your embeddings.  However, since this is a very domain specific dataset - those general purpose benchmarks may not be appropriate.  Unfortunately, we have not designed downstream tasks that we can open source at this point.
# 
# In the absence of these downstream tasks, we can at least sanity check that these embeddings contain semantic information by doing the following:
# 
# 1. Manually examine similarity between sentences, by supplying a statement and examining if the nearest phrase found is similar. 
# 
# 2. Visualize the embeddings.
# 
# We will do the first approach, and leave the second approach as an exercise for the reader.  **It should be noted that this is only a sanity check -- a more rigorous approach is to measure the impact of these embeddings on a variety of downstream tasks** and use that to form a more objective opinion about the quality of your embeddings.
# 
# Furthermroe, there are many different ways of constructing a sentence embedding from the language model.  For example, we can take the average, the maximum or even the last value of the hidden states (or concatenate them all together).  **For simplicity, we will only evaluate the sentence embedding that is constructed by taking the average over the hidden states** (and leave other possibilities as an exercise for the reader). 

# ### Create search index using `nmslib` 
# 
# [nmslib](https://github.com/nmslib/nmslib) is a great library for doing nearest neighbor lookups, which we will use as a search engine for finding nearest neighbors of comments in vector-space.  
# 
# The convenience function `create_nmslib_search_index` builds this search index given a matrix of vectors as input.
# 
# 

# In[78]:


from general_utils import create_nmslib_search_index
import nmslib
from lang_model_utils import Query2Emb
from pathlib import Path
import numpy as np
from lang_model_utils import load_lm_vocab
import torch


# In[79]:


# Load matrix of vectors
loadpath = Path('./data/lang_model_emb/')
avg_emb_dim500 = np.load(loadpath/'avg_emb_dim500_test_v2.npy')


# In[67]:


# Build search index (takes about an hour on a p3.8xlarge)
dim500_avg_searchindex = create_nmslib_search_index(avg_emb_dim500)


# In[68]:


# save search index
dim500_avg_searchindex.saveIndex('./data/lang_model_emb/dim500_avg_searchindex.nmslib')


# Note that if you did not train your own language model and are downloading the pre-trained model artifacts instead, you can similarly download the pre-computed search index here: 
# 
# https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model_emb/dim500_avg_searchindex.nmslib

# After you have built this search index with nmslib, you can do fast nearest-neighbor lookups.  We use the `Query2Emb` object to help convert strings to the embeddings: 

# In[80]:


dim500_avg_searchindex = nmslib.init(method='hnsw', space='cosinesimil')
dim500_avg_searchindex.loadIndex('./data/lang_model_emb/dim500_avg_searchindex.nmslib')


# In[81]:


lang_model = torch.load('./data/lang_model/lang_model_cpu_v2.torch')
vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')

q2emb = Query2Emb(lang_model = lang_model.cpu(),
                  vocab = vocab)


# The method `Query2Emb.emb_mean` will allow us to use the langauge model we trained earlier to generate a sentence embedding given a string.   Here is an example, `emb_mean` will return a numpy array of size (1, 500).

# In[82]:


query = q2emb.emb_mean('Read data into pandas dataframe')
query.shape


# **Make search engine to inspect semantic similarity of phrases**.  This will take 3 inputs:
# 
# 1. `nmslib_index` - this is the search index we built above.  This object takes a vector and will return the index of the closest vector(s) according to cosine distance.  
# 2. `ref_data` - this is the data for which the index refer to, in this case will be the docstrings. 
# 3. `query2emb_func` - this is a function that will convert a string into an embedding.

# In[83]:


class search_engine:
    def __init__(self, 
                 nmslib_index, 
                 ref_data, 
                 query2emb_func):
        
        self.search_index = nmslib_index
        self.data = ref_data
        self.query2emb_func = query2emb_func
    
    def search(self, str_search, k=3):
        query = self.query2emb_func(str_search)
        idxs, dists = self.search_index.knnQuery(query, k=k)
        
        for idx, dist in zip(idxs, dists):
            print(f'cosine dist:{dist:.4f}\n---------------\n', self.data[idx])


# In[84]:


se = search_engine(nmslib_index=dim500_avg_searchindex,
                   ref_data = test_raw,
                   query2emb_func = q2emb.emb_mean)


# ## Manually Inspect Phrase Similarity
# 
# Compare a user-supplied query vs. vectorized docstrings on test set.  We can see that similar phrases are not exactly the same, but the nearest neighbors are reasonable.  

# In[85]:


import logging
logging.getLogger().setLevel(logging.ERROR)


# In[86]:


se.search('read csv into pandas dataframe')


# In[87]:


se.search('train a random forest')


# In[88]:


se.search('download files')


# In[89]:


se.search('start webserver')


# In[90]:


se.search('send out email notification')


# In[91]:


se.search('save pickle file')


# ### Visualize Embeddings (Optional)
# 
# We highly recommend using [tensorboard](https://www.tensorflow.org/versions/r1.0/get_started/embedding_viz) as way to visualize embeddings.  Tensorboard contains an interactive search that makes it easy (and fun) to explore embeddings.  We leave this as an exercise to the reader.

# In[ ]:




