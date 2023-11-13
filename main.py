import pyterrier as pt
import pandas as pd
from tira.third_party_integrations import ir_datasets

dataset = ir_datasets.load('ir-lab-jena-leipzig-wise-2023/training-20231104-training')

if not pt.started():
    pt.init()

def create_index(documents):
    indexer = pt.IterDictIndexer("/tmp/index", overwrite=True, meta={'docno': 100, 'text': 20480})
    index_ref = indexer.index(documents)
    return pt.IndexFactory.of(index_ref)

#documents = list(dataset.docs_iter())
#df = pd.DataFrame([{'doc_id': [],'text': []}])
data = {'doc_id': [], 'text': []}


for documents in list(dataset.docs_iter()) [:5]:
    print (documents.doc_id)
    data['doc_id'].append(documents.doc_id)
    data['text'].append(documents.text)

df = pd.DataFrame(data)

# Print the resulting DataFrame
print(df)
#index = create_index(documents)
#bm25 = pt.BatchRetrieve(index, wmodel="BM25")
#bm25.search("cheap car")

