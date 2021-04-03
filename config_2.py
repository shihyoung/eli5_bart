import os
# ROOT = '/home/sureear'
ROOT = os.getcwd()

eli5_precomputed_train = os.path.join(ROOT,'eli5_train_precomputed_dense_docs_15doc.json')
eli5_precomputed_valid = os.path.join(ROOT,'eli5_valid_precomputed_dense_docs_15doc.json')
eli5_precomputed_test = os.path.join(ROOT,'eli5_test_precomputed_dense_docs_15doc.json')

not_key = os.path.join(ROOT,'eli5_train_not_key_1.json')
not_key_final = os.path.join(ROOT,'eli5_train_not_key_only_remove_dupl.json')

eli5_precomputed_train_final = os.path.join(ROOT,'eli5_train_precomputed_only_triple.json')
eli5_precomputed_valid_final = os.path.join(ROOT,'eli5_valid_precomputed_only_triple.json')
eli5_precomputed_test_final = os.path.join(ROOT,'eli5_test_precomputed_only_triple.json')

eli5_precomputed_train_final_remove_dupl = os.path.join(ROOT, 'eli5_train_precomputed_only_triple_remove_dupl.json')
eli5_precomputed_train_final_remove_dupl_2 = os.path.join(ROOT, 'eli5_train_precomputed_only_triple_remove_dupl_2.json')
eli5_precomputed_valid_final_remove_dupl = os.path.join(ROOT, 'eli5_valid_precomputed_only_triple_remove_dupl.json')
eli5_precomputed_test_final_remove_dupl = os.path.join(ROOT, 'eli5_test_precomputed_only_triple_remove_dupl.json')

model_save_name = os.path.join(ROOT,'workspace/2021_ETRI_0.1v')

eli5_precomputed_train_with_triple = [os.path.join(ROOT,'tmp/eli5_train_precomputed_dense_docs_15doc_triple.json'),
os.path.join(ROOT,'tmp/eli5_train_precomputed_dense_docs_15doc_triple_2.json'),
os.path.join(ROOT,'tmp/eli5_train_precomputed_dense_docs_15doc_triple_3.json'),
os.path.join(ROOT,'tmp/eli5_train_precomputed_dense_docs_15doc_triple_4.json'),
os.path.join(ROOT,'tmp/eli5_train_precomputed_dense_docs_15doc_triple_5.json')]
eli5_precomputed_train_with_triple_refine = os.path.join(ROOT,'eli5_train_precomputed_dense_docs_15doc_triple_refine.json')
eli5_precomputed_valid_with_triple_refine = os.path.join(ROOT,'eli5_valid_precomputed_dense_docs_15doc_triple_refine.json')
eli5_precomputed_test_with_triple_refine = os.path.join(ROOT,'eli5_test_precomputed_dense_docs_15doc_triple_refine.json')
eli5_precomputed_valid_with_triple = os.path.join(ROOT,'eli5_valid_precomputed_dense_docs_15doc_triple.json')
eli5_precomputed_test_with_triple = os.path.join(ROOT,'eli5_test_precomputed_dense_docs_15doc_triple.json')