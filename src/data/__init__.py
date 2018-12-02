from .embed import TextData
from .loader import standardize_dataset, mk_dataloader
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

###
### Private:
###
__NEWSGROUPS = fetch_20newsgroups(subset='test')
__textdata = TextData(__NEWSGROUPS.data, __NEWSGROUPS.target)
__textdata.embed()
_, __target, __embedding = __textdata.getter()   

(__embedding_trn, __embedding_vld,
 __target_trn, __target_vld) = train_test_split(__embedding, 
                                                __target,
                                                train_size = 700,
                                                test_size = 400,
                                                stratify = __target,
                                                random_state = 999
                                               )


###
### Public API:
###
training_batches = mk_dataloader(standardize_dataset(__embedding_trn, 
                                                     __target_trn)
                                )
validati_batches = mk_dataloader(standardize_dataset(__embedding_vld, 
                                                     __target_vld)
                                )