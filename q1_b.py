import csv
import math
from collections import defaultdict, Counter

# ------------ Data Loading ------------ #
def load_documents(file_path):
    documents = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            doc_id, stemmed_content = row[0], row[1].split()
            documents[doc_id] = stemmed_content
    return documents

def load_queries(file_path):
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            query_id, stemmed_query = row[0], row[1].split()
            queries[query_id] = stemmed_query
    return queries

def load_assessments(file_path):
    relevance = defaultdict(set)
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            query_id, doc_id = row[0], row[1]
            relevance[query_id].add(doc_id)
    return relevance

# ------------ IDF Computation ------------ #
def compute_idf(documents):
    df = defaultdict(int)
    N = len(documents)
    for content in documents.values():
        unique_terms = set(content)
        for term in unique_terms:
            df[term] += 1
    idf = {term: math.log((N + 1) / (df_t + 1)) for term, df_t in df.items()}
    return idf

# ------------ Scoring Models ------------ #
def tfidf_score(query, doc, idf):
    tf = Counter(doc)
    doc_len = len(doc)
    return sum((tf[t] / doc_len) * idf.get(t, 0) for t in query)

def max_tf_tfidf_score(query, doc, idf):
    tf = Counter(doc)
    max_tf = max(tf.values()) if tf else 1
    return sum((tf[t] / max_tf) * idf.get(t, 0) for t in query)

def bm25_score(query, doc, idf, avg_doc_len, k1=1.5, b=0.75):
    tf = Counter(doc)
    doc_len = len(doc)
    score = 0.0
    for t in query:
        f = tf[t]
        idf_t = idf.get(t, 0)
        denom = f + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += idf_t * ((f * (k1 + 1)) / (denom + 1e-6))
    return score

def rank_documents(query, documents, idf, model='tfidf', avg_doc_len=None):
    results = []
    for doc_id, doc in documents.items():
        if model == 'tfidf':
            score = tfidf_score(query, doc, idf)
        elif model == 'maxtf':
            score = max_tf_tfidf_score(query, doc, idf)
        elif model == 'bm25':
            score = bm25_score(query, doc, idf, avg_doc_len)
        else:
            raise ValueError(f"Unknown model: {model}")
        results.append((doc_id, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

# ------------ Evaluation ------------ #
def precision_recall_at_k(ranked_docs, relevant_docs, k):
    retrieved = [doc_id for doc_id, _ in ranked_docs[:k]]
    relevant_retrieved = [d for d in retrieved if d in relevant_docs]
    precision = len(relevant_retrieved) / k
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
    return precision, recall

def evaluate(queries, documents, relevance, idf, k, model):
    total_precision = 0.0
    total_recall = 0.0
    num_queries = len(queries)
    avg_doc_len = sum(len(doc) for doc in documents.values()) / len(documents)

    for qid, query in queries.items():
        ranked = rank_documents(query, documents, idf, model, avg_doc_len)
        precision, recall = precision_recall_at_k(ranked, relevance[qid], k)
        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    return avg_precision, avg_recall

# ------------ Main ------------ #
if __name__ == '__main__':
    documents = load_documents('/Users/busa/Downloads/IRDM/Code/queries.csv')
    queries = load_queries('/Users/busa/Downloads/IRDM/Code/queries.csv')
    relevance = load_assessments('/Users/busa/Downloads/IRDM/Code/assessments.csv')
    idf = compute_idf(documents)
    models = ['tfidf', 'maxtf', 'bm25']

    for model in models:
        print(f"\nModel: {model.upper()}")
        for k in [5, 10, 20]:
            avg_p, avg_r = evaluate(queries, documents, relevance, idf, k, model)
            print(f"k={k} | Avg Precision: {avg_p:.4f}, Avg Recall: {avg_r:.4f}")