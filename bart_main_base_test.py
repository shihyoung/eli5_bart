import json
import config_2 as config

from lfqa_utils import *
eli5 = nlp.load_dataset('eli5')
eli5_test_docs = json.load(open(config.eli5_precomputed_test))
class ArgumentsS2S():
    def __init__(self):
        self.batch_size = 4
        self.backward_freq = 32
        self.max_length = 1024
        self.print_freq = 100
        self.model_save_name = "seq2seq_models/eli5_bart_model_base_max_length_768"
        self.from_pretrained_name = os.path.join('seq2seq_models','eli5_bart_model_base_1_3.1640655269638347_0.4481543996763394_0.11885400231723851_0.32311457544428884_0.3718931075854591.pth')
        self.learning_rate = 2e-4
        self.num_epochs = 3

# qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
# qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to('cuda:0')
# _ = qar_model.eval()

qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to('cuda:0')
_ = qar_model.eval()
s2s_args = ArgumentsS2S()
# qa_s2s_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
# qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')
qa_s2s_tokenizer, qa_s2s_model = make_qa_s2s_model(
    model_name="seq2seq_models/eli5_bart_model_base",
    from_file=s2s_args.from_pretrained_name,
    is_save=False,
    args=s2s_args,
    device="cuda:0"
)
_ = qa_s2s_model.eval()
wiki40b_snippets = nlp.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']
# s2s_test_dset = ELI5DatasetS2S_with_graph(eli5['test_eli5'],chat_s2s_tokenizer, document_cache=config.eli5_precomputed_test_final_remove_dupl, document_cache_2=dict([(k, d) for k, d, src_ls in eli5_test_docs]),training=False)
# chat_s2s_tokenizer = BartTokenizer.from_pretrained('yjernite/bart_eli5')
# chat_s2s_model = BartForChatbot.from_pretrained('yjernite/bart_eli5').to('cuda:0')
# _ = chat_s2s_model.eval()
faiss_res = faiss.StandardGpuResources()
wiki40b_passage_reps = np.memmap(
            'wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat',
            dtype='float32', mode='r',
            shape=(wiki40b_snippets.num_rows, 128)
)

wiki40b_index_flat = faiss.IndexFlatIP(128)
wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
wiki40b_gpu_index.add(wiki40b_passage_reps)
questions = []
answers = []

for i in [12345] + [j for j in range(10)]:
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
    # print ("####question")
    # print (question)
    # print("####answer")
    # print (answer)
    # print("####reference")
    # print (eli5['test_eli5'][i]['answers']['text'][0])
    questions += [question]
    answers += [answer]

df = pd.DataFrame({
    'Question': questions,
    'Answer': answers,
})
# df.style.set_properties(**{'text-align': 'left'})
print (df)
# exit()
predicted = []
reference = []

# Generate answers for the full test set
print (eli5['test_eli5'].num_rows)
for i in tqdm(range(eli5['test_eli5'].num_rows)):
    # create support document with the dense index
    question = eli5['test_eli5'][i]['title']
    # in_st, out_st, adj, entity_list = s2s_test_dset.make_example(i)
    doc, res_list = query_qa_dense_index(
        question, qar_model, qar_tokenizer,
        wiki40b_snippets, wiki40b_gpu_index, device='cuda:0'
    )
    # concatenate question and support document into BART input
    question_doc = "question: {} context: {}".format(question, doc)
    if i % 50 == 0:
        print (i)
    # generate an answer with beam search
    # answer = chat_s2s_generate(
    #     in_st, adj, entity_list, chat_s2s_model, chat_s2s_tokenizer,
    #     num_answers=1,
    #     num_beams=8,
    #     min_len=96,
    #     max_len=256,
    #     max_input_length=1024,
    #     device="cuda:0"
    # )[0]
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