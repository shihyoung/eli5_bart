import config_2 as config
import json
from lfqa_utils import *
import nlp
import numpy as np
from transformers import BartTokenizer
from Bart_chatbot import BartForChatbot
import get_triple_from_document

# wiki40b_snippets = nlp.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']

eli5 = nlp.load_dataset('eli5')
# faiss_res = faiss.StandardGpuResources()
# wiki40b_passage_reps = np.memmap(
#             'wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat',
#             dtype='float32', mode='r',
#             shape=(wiki40b_snippets.num_rows, 128)
# )

# wiki40b_index_flat = faiss.IndexFlatIP(128)
# wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
# wiki40b_gpu_index.add(wiki40b_passage_reps)
# training loop proper
class ArgumentsS2S():
    def __init__(self):
        self.batch_size = 8
        self.backward_freq = 16
        self.max_length = 768
        self.print_freq = 100
        self.model_save_name = "seq2seq_models/eli5_bart_model_base_max_length_768"
        self.learning_rate = 2e-4
        self.num_epochs = 3

# qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
# qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to('cuda:0')
# _ = qar_model.eval()

s2s_args = ArgumentsS2S()
import time
eli5_train_docs = json.load(open(config.eli5_precomputed_train))
eli5_valid_docs = json.load(open(config.eli5_precomputed_valid))

# eli5_test_docs = json.load(open(config.eli5_precomputed_test))

# eli5_train_docs = json.load(open(config.eli5_precomputed_train_final_remove_dupl))
# eli5_valid_docs = json.load(open(config.eli5_precomputed_valid_final_remove_dupl))
# eli5_test_docs = json.load(open(config.eli5_precomputed_test_final_remove_dupl))


# from stanza.server import CoreNLPClient

#
# new_eli5_valid_docs = []
#
# for k, d, src_ls in tqdm(eli5_valid_docs):
#     document = d.lower().split('<p>')
#     sentences = []
#     all_triples = []
#     i=0
#
#     for doc in document:
#         doc = doc.strip().replace('\n',' ').replace('\r','')
#         if len(doc) == 0:
#             continue
#         i += 1
#         if i == 11:
#             break
#         sentence_openie = client.annotate(doc,simple_format=False)
#         # for sentence in sentence_openie.sentence:
#         # print (sentence_openie)
#         for sen in sentence_openie['sentences']:
#             s = ' '.join(i['originalText'] for i in sen['tokens'])
#             s_split = s.split()
#             sentences.append(s)
#             triples = []
#             check_span = set()
#             for t in sen['openie']:
#                 c_span = str(t['subjectSpan']) + str(t['relationSpan']) + str(t['objectSpan'])
#                 if c_span in check_span:
#                     continue
#                 check_span.add(c_span)
#                 triple = {}
#
#                 triple['subject'] = ' '.join(s_split[t['subjectSpan'][0]:t['subjectSpan'][1]])
#                 triple['relation'] = ' '.join(s_split[t['relationSpan'][0]:t['relationSpan'][1]])
#                 triple['object'] = ' '.join(s_split[t['objectSpan'][0]:t['objectSpan'][1]])
#                 triple['subjectSpan'] = t['subjectSpan']
#                 triple['relationSpan'] = t['relationSpan']
#                 triple['objectSpan'] = t['objectSpan']
#                 triples.append(triple)
#             all_triples.append(triples)
#
#     new_eli5_valid_docs += [(k, d, src_ls, sentences, all_triples)]
#
# new_eli5_test_docs = []
# for k, d, src_ls in tqdm(eli5_test_docs):
#     document = d.lower().split('<p>')
#     sentences = []
#     all_triples = []
#     i=0
#     for doc in document:
#
#         doc = doc.strip().replace('\n',' ').replace('\r','')
#         if len(doc) == 0:
#             continue
#         i += 1
#         if i == 11:
#             break
#         sentence_openie = client.annotate(doc,simple_format=False)
#         # for sentence in sentence_openie.sentence:
#         # print (sentence_openie)
#         for sen in sentence_openie['sentences']:
#             s = ' '.join(i['originalText'] for i in sen['tokens'])
#             s_split = s.split()
#             sentences.append(s)
#             triples = []
#             check_span = set()
#             for t in sen['openie']:
#                 c_span = str(t['subjectSpan']) + str(t['relationSpan']) + str(t['objectSpan'])
#                 if c_span in check_span:
#                     continue
#                 check_span.add(c_span)
#                 triple = {}
#
#                 triple['subject'] = ' '.join(s_split[t['subjectSpan'][0]:t['subjectSpan'][1]])
#                 triple['relation'] = ' '.join(s_split[t['relationSpan'][0]:t['relationSpan'][1]])
#                 triple['object'] = ' '.join(s_split[t['objectSpan'][0]:t['objectSpan'][1]])
#                 triple['subjectSpan'] = t['subjectSpan']
#                 triple['relationSpan'] = t['relationSpan']
#                 triple['objectSpan'] = t['objectSpan']
#                 triples.append(triple)
#             all_triples.append(triples)
#
#     new_eli5_test_docs += [(k, d, src_ls, sentences, all_triples)]
#
# with open(config.eli5_precomputed_valid_with_triple, "w") as json_file:
#     json.dump(new_eli5_valid_docs, json_file)
# with open(config.eli5_precomputed_test_with_triple, "w") as json_file:
#     json.dump(new_eli5_test_docs, json_file)
#
# exit()
# eli5['train_eli5']['labels'] = eli5['train_eli5'].pop('lm_labels')
# eli5['validation_eli5']['labels'] = eli5['validation_eli5'].pop('lm_labels')
# eli5['test_eli5']['labels'] = eli5['test_eli5'].pop('lm_labels')

qa_s2s_tokenizer, pre_model = make_qa_s2s_model(
    model_name="facebook/bart-base",
    from_file=None,
    is_save=True,
    args=s2s_args,
    device="cuda:0"
)

# triple_module = get_triple_from_document.triple_graph_package(chat_s2s_tokenizer)
# s2s_train_dset = ELI5DatasetS2S_with_graph(eli5['train_eli5'],chat_s2s_tokenizer, document_cache=dict([(k, d) for k, d, src_ls in eli5_train_docs]), triple_module=triple_module)
# s2s_valid_dset = ELI5DatasetS2S_with_graph(eli5['validation_eli5'],chat_s2s_tokenizer, document_cache=dict([(k, d) for k, d, src_ls in eli5_valid_docs]), triple_module=triple_module, training=False)



s2s_train_dset = ELI5DatasetS2S(eli5['train_eli5'],qa_s2s_tokenizer, document_cache=dict([(k, d) for k, d, src_ls in eli5_train_docs]))
s2s_valid_dset = ELI5DatasetS2S(eli5['validation_eli5'],qa_s2s_tokenizer, document_cache=dict([(k, d) for k, d, src_ls in eli5_valid_docs]),training=False)

qa_s2s_model = torch.nn.DataParallel(pre_model)
print (eli5['train_eli5'][12345])
print (eli5['test_eli5'][12345])
# exit()
train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args)
exit()
# qa_s2s_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
# qa_s2s_tokenizer = BartTokenizer.from_pretrained('yjernite/bart_eli5')
# qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')
# qa_s2s_model = BartForChatbot.from_pretrained('yjernite/bart_eli5').to('cuda:0')
# qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')

_ = qa_s2s_model.eval()

questions = []
answers = []

for i in [12345] + [j for j in range(4)]:
    # create support document with the dense index
    question = eli5['test_eli5'][i]['title']
    doc, res_list = query_qa_dense_index(
        question, qar_model, qar_tokenizer,
        wiki40b_snippets, wiki40b_gpu_index, device='cuda:0'
    )
    # concatenate question and support document into BART input
    question_doc = "question: {} context: {}".format(question, doc)
    # generate an answer with beam search
    answer = qa_s2s_generate(
            question_doc, qa_s2s_model, qa_s2s_tokenizer,
            num_answers=1,
            num_beams=8,
            min_len=64,
            max_len=256,
            max_input_length=1024,
            device="cuda:0"
    )[0]
    questions += [question]
    answers += [answer]

df = pd.DataFrame({
    'Question': questions,
    'Answer': answers,
})
df.style.set_properties(**{'text-align': 'left'})

predicted = []
reference = []

# Generate answers for the full test set
print (eli5['test_eli5'].num_rows)
for i in range(eli5['test_eli5'].num_rows):
    # create support document with the dense index
    question = eli5['test_eli5'][i]['title']
    doc, res_list = query_qa_dense_index(
        question, qar_model, qar_tokenizer,
        wiki40b_snippets, wiki40b_gpu_index, device='cuda:0'
    )
    # concatenate question and support document into BART input
    question_doc = "question: {} context: {}".format(question, doc)
    if i % 50 == 0:
        print (i)
    # generate an answer with beam search
    answer = qa_s2s_generate(
            question_doc, qa_s2s_model, qa_s2s_tokenizer,
            num_answers=1,
            num_beams=8,
            min_len=96,
            max_len=256,
            max_input_length=1024,
            device="cuda:0"
    )[0]
    predicted += [answer]
    reference += [eli5['test_eli5'][i]['answers']['text'][0]]
# Compare each generation to the fist answer from the dataset
nlp_rouge = nlp.load_metric('rouge')

scores = nlp_rouge.compute(
    predicted, reference,
    rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
    use_agregator=True, use_stemmer=False
)
df = pd.DataFrame({
    'rouge1': [scores['rouge1'].mid.precision, scores['rouge1'].mid.recall, scores['rouge1'].mid.fmeasure],
    'rouge2': [scores['rouge2'].mid.precision, scores['rouge2'].mid.recall, scores['rouge2'].mid.fmeasure],
    'rougeL': [scores['rougeL'].mid.precision, scores['rougeL'].mid.recall, scores['rougeL'].mid.fmeasure],
}, index=[ 'P', 'R', 'F'])
df.style.format({'rouge1': "{:.4f}", 'rouge2': "{:.4f}", 'rougeL': "{:.4f}"})

print (scores)