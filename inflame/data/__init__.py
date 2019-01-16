"""
The data_pipeline function strings together bits and pieces of text processing
functionality that are defined in the util folder. It takes
data and target, which are two lists of texts, embeds them in GloVe,
and lastly converts them to pytorch DataLoader objects.   
"""
from .util import NLP, standardize_dataset, mk_dataloader
from sklearn.model_selection import train_test_split

def data_pipeline(data, target: 'lists of texts'):
   '''Split into train/test; embed text into vectors; then create Dataloaders.'''
   # Stratified sample the data and target into training and validation datasets.
   (data_trn, data_vld,
   target_trn, target_vld) = train_test_split(data, 
                                              target,
                                              train_size = .75,
                                              test_size = .25,
                                              stratify = target,
                                              random_state = 999
                                             )

   # Embed data.
   embedding_trn = NLP(data_trn).embed()
   embedding_vld = NLP(data_vld).embed()

   # Standardize numpy tensors and convert them to pytorch tensors.
   training_batches = mk_dataloader(standardize_dataset(embedding_trn, 
                                                        target_trn
                                                       )
                                   )
   validati_batches = mk_dataloader(standardize_dataset(embedding_vld, 
                                                        target_vld
                                                       )
                                   )
   
   return training_batches, validati_batches