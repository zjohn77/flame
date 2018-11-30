from .embed import *
from .loader import *
from sklearn.model_selection import train_test_split

textdata = TextData(NEWSGROUPS.data, NEWSGROUPS.target)
textdata.embed()
_, target, embedding = textdata.getter()   

embedding_trn, embedding_vld, target_trn, target_vld = train_test_split(embedding, 
                                                                        target,
                                                                        stratify = target, 
                                                                        test_size = .5
                                                                       )

training_batches = mk_dataloader(standardize_dataset(embedding_trn, 
                                                     target_trn)
                                )
validati_batches = mk_dataloader(standardize_dataset(embedding_vld, 
                                                     target_vld)
                                )