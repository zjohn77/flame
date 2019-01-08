from .util.embed import TextData
from .util.loader import standardize_dataset, mk_dataloader
from sklearn.model_selection import train_test_split

def data_pipeline(data, target: 'lists of texts'):
   (data_trn, data_vld,
   target_trn, target_vld) = train_test_split(data, 
                                              target,
                                              train_size = .75,
                                              test_size = .25,
                                              stratify = target,
                                              random_state = 999
                                             )
   textdata_trn = TextData(data_trn, target_trn)
   textdata_vld = TextData(data_vld, target_vld)

   textdata_trn.embed()
   textdata_vld.embed()

   _, target_trn, embedding_trn = textdata_trn.getter()   
   _, target_vld, embedding_vld = textdata_vld.getter()   

   training_batches = mk_dataloader(standardize_dataset(embedding_trn, 
                                                        target_trn
                                                       )
                                   )
   validati_batches = mk_dataloader(standardize_dataset(embedding_vld, 
                                                        target_vld
                                                       )
                                   )
   return training_batches, validati_batches