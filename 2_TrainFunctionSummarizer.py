#!/usr/bin/env python
# coding: utf-8

# Environment is available as a publically available docker container: `hamelsmu/ml-gpu`
# 
# ### Pre-requisite: Familiarize yourself with sequence-to-sequence models
# 
# If you are not familiar with sequence to sequence models, please refer to [this tutorial](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8).
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
# If `use_cache = True`, data will be downloaded where possible instead of re-computing.  However, it is highly recommended that you set `use_cache = False` for this tutorial as it will be less confusing, and you will learn more by runing these steps yourself. **This notebook takes approximately 4 hours to run on an AWS `p3.8xlarge` instance.**

# In[12]:


use_cache = False


# In[2]:


# # Optional: you can set what GPU you want to use in a notebook like this.  
# # Useful if you want to run concurrent experiments at the same time on different GPUs.
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[4]:


# This will allow the notebook to run faster
from pathlib import Path
from general_utils import get_step2_prerequisite_files, read_training_files
from keras.utils import get_file
OUTPUT_PATH = Path('./data/seq2seq/')
OUTPUT_PATH.mkdir(exist_ok=True)


# # Read Text From File
# 
# We want to read in raw text from files so we can pre-process the text for modeling as described in [this tutorial](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8)

# In[13]:


if use_cache:
    get_step2_prerequisite_files(output_directory = './data/processed_data')

# you want to supply the directory where the files are from step 1.
train_code, holdout_code, train_comment, holdout_comment = read_training_files('./data/processed_data/')


# In[14]:


# code and comment files should be of the same length.

assert len(train_code) == len(train_comment)
assert len(holdout_code) == len(holdout_comment)


# # Tokenize Text
# 
# In this step, we are going to pre-process the raw text for modeling.  For an explanation of what this section does, see the [Preapre & Clean Data section of this Tutorial](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8)

# In[7]:


from ktext.preprocess import processor

if not use_cache:    
    code_proc = processor(hueristic_pct_padding=.7, keep_n=20000)
    t_code = code_proc.fit_transform(train_code)

    comment_proc = processor(append_indicators=True, hueristic_pct_padding=.7, keep_n=14000, padding ='post')
    t_comment = comment_proc.fit_transform(train_comment)

elif use_cache:
    logging.warning('Not fitting transform function because use_cache=True')


# **Save tokenized text** (You will reuse this for step 4)

# In[10]:


import dill as dpickle
import numpy as np

if not use_cache:
    # Save the preprocessor
    with open(OUTPUT_PATH/'py_code_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(code_proc, f)

    with open(OUTPUT_PATH/'py_comment_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(comment_proc, f)

    # Save the processed data
    np.save(OUTPUT_PATH/'py_t_code_vecs_v2.npy', t_code)
    np.save(OUTPUT_PATH/'py_t_comment_vecs_v2.npy', t_comment)


# Arrange data for modeling

# In[5]:



from seq2seq_utils import load_decoder_inputs, load_encoder_inputs, load_text_processor


encoder_input_data, encoder_seq_len = load_encoder_inputs(OUTPUT_PATH/'py_t_code_vecs_v2.npy')
decoder_input_data, decoder_target_data = load_decoder_inputs(OUTPUT_PATH/'py_t_comment_vecs_v2.npy')
num_encoder_tokens, enc_pp = load_text_processor(OUTPUT_PATH/'py_code_proc_v2.dpkl')
num_decoder_tokens, dec_pp = load_text_processor(OUTPUT_PATH/'py_comment_proc_v2.dpkl')


# If you don't have the above files on disk because you set `use_cache = True` you can download the files for the above function calls here:
# 
#  - https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_t_code_vecs_v2.npy
#  - https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_t_comment_vecs_v2.npy
#  - https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_code_proc_v2.dpkl
#  - https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_comment_proc_v2.dpkl

# # Build Seq2Seq Model For Summarizing Code
# 
# We will build a model to predict the docstring given a function or a method.  While this is a very cool task in itself, this is not the end goal of this exercise.  The motivation for training this model is to learn a general purpose feature extractor for code that we can use for the task of code search.

# In[6]:


from seq2seq_utils import build_seq2seq_model


# The convenience function `build_seq2seq_model` constructs the architecture for a sequence-to-sequence model.  
# 
# The architecture built for this tutorial is a minimal example with only one layer for the encoder and decoder, and does not include things like [attention](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf).  We encourage you to try and build different architectures to see what works best for you!

# In[7]:


seq2seq_Model = build_seq2seq_model(word_emb_dim=800,
                                    hidden_state_dim=1000,
                                    encoder_seq_len=encoder_seq_len,
                                    num_encoder_tokens=num_encoder_tokens,
                                    num_decoder_tokens=num_decoder_tokens)


# In[8]:


seq2seq_Model.summary()


# ### Train Seq2Seq Model

# In[9]:


from keras.models import Model, load_model
import pandas as pd
import logging

if not use_cache:

    from keras.callbacks import CSVLogger, ModelCheckpoint
    import numpy as np
    from keras import optimizers

    seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.00005), loss='sparse_categorical_crossentropy')

    script_name_base = 'py_func_sum_v9_'
    csv_logger = CSVLogger('{:}.log'.format(script_name_base))

    model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                       save_best_only=True)

    batch_size = 1100
    epochs = 16
    history = seq2seq_Model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.12, callbacks=[csv_logger, model_checkpoint])


# In[ ]:


if use_cache:
    logging.warning('Not re-training function summarizer seq2seq model because use_cache=True')
    # Load model from url
    loc = get_file(fname='py_func_sum_v9_.epoch16-val2.55276.hdf5',
                   origin='https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_func_sum_v9_.epoch16-val2.55276.hdf5')
    seq2seq_Model = load_model(loc)
    
    # Load encoder (code) pre-processor from url
    loc = get_file(fname='py_code_proc_v2.dpkl',
                   origin='https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_code_proc_v2.dpkl')
    num_encoder_tokens, enc_pp = load_text_processor(loc)
    
    # Load decoder (docstrings/comments) pre-processor from url
    loc = get_file(fname='py_comment_proc_v2.dpkl',
                   origin='https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/py_comment_proc_v2.dpkl')
    num_decoder_tokens, dec_pp = load_text_processor(loc)
    


# Note that the above procedure will automatically download a pre-trained model and associated artifacts from https://storage.googleapis.com/kubeflow-examples/code_search/data/seq2seq/ if `use_cache = True`.  
# 
# Otherwise, the above code will checkpoint the best model after each epoch into the current directory with prefix `py_func_sum_v9_`

# # Evaluate Seq2Seq Model For Code Summarization
# 
# To evaluate this model we are going to do two things:
# 
# 1.  Manually inspect the results of predicted docstrings for code snippets, to make sure they look sensible.
# 2.  Calculate the [BLEU Score](https://en.wikipedia.org/wiki/BLEU) so that we can quantitately benchmark different iterations of this algorithm and to guide hyper-parameter tuning.

# ### Manually Inspect Results (on holdout set)

# In[15]:


from seq2seq_utils import Seq2Seq_Inference
import pandas as pd

seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                 decoder_preprocessor=dec_pp,
                                 seq2seq_model=seq2seq_Model)

demo_testdf = pd.DataFrame({'code':holdout_code, 'comment':holdout_comment, 'ref':''})
seq2seq_inf.demo_model_predictions(n=15, df=demo_testdf)


# ### Comment on manual inspection of results:
# 
# The predicted code summaries are not perfect, but we can see that the model has learned to extract some semantic meaning from the code.  That's all we need to get reasonable results in this case.  

# ### Calculate BLEU Score (on holdout set)
# 
# BLEU Score is described [in this wikipedia article](https://en.wikipedia.org/wiki/BLEU), and is a way to measure the efficacy of summarization/translation such as the one we conducted here.  This metric is useful if you wish to conduct extensive hyper-parameter tuning and try to improve the seq2seq model.

# In[24]:


# This will return a BLEU Score
seq2seq_inf.evaluate_model(input_strings=holdout_code, 
                           output_strings=holdout_comment, 
                           max_len=None)


# # Save model to disk
# 
# Save the model to disk so you can use it in Step 4 of this tutorial.

# In[16]:


seq2seq_Model.save(OUTPUT_PATH/'code_summary_seq2seq_model.h5')


# In[ ]:




