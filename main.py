import pyterrier as pt
import pandas as pd
import re
from tira.third_party_integrations import ir_datasets, persist_and_normalize_run, ensure_pyterrier_is_loaded

ensure_pyterrier_is_loaded()
dataset = ir_datasets.load('ir-lab-jena-leipzig-wise-2023/training-20231104-training')

if not pt.started():
    pt.init()

def create_index(documents):
    indexer = pt.IterDictIndexer("./tmp/index", overwrite=True, meta={'docno': 20, 'text': 11000})
    index_ref = indexer.index(documents)
    return pt.IndexFactory.of(index_ref)

def getDFs():
    qrels = {'qid': [], 'docno': [], 'relevance': []}
    topics = {'qid': [], 'query': []}

    for query in dataset.queries_iter():
        topics['qid'].append(query.query_id)
        CleanedText = re.sub(r'[^ \w+]', '', query.default_text())
        topics['query'].append(CleanedText)

    for rel in dataset.qrels_iter():
        qrels['qid'].append(rel.query_id)
        qrels['docno'].append(rel.doc_id)
        qrels['relevance'].append(rel.relevance)

    qrelDF = pd.DataFrame(qrels)
    queryDF = pd.DataFrame(topics)
    return qrelDF, queryDF

Docs = []
for document in list(dataset.docs_iter()):
    CleanedText = document.default_text().replace('\n', ' ')
    Docs.append({'docno': document.doc_id, 'text': CleanedText})
    
index = create_index(Docs)
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
qrels, topics = getDFs()
#print(pt.Experiment([bm25], topics, qrels, eval_metrics=['ndcg_cut_10', 'map_10','ndcg']))
results = bm25.transform(topics)
persist_and_normalize_run(results, output_file='./tmp', system_name='BM25Base')

