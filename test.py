import requests
import json
import ipdb
from sentence_transformers import SentenceTransformer, util
import torch
import typing

def get_pair(dataset:str) -> dict():
    '''
    process the dataset
    '''
    f = open(dataset)
    data = json.load(f)
    
    out = dict()
    i = data['questions']
    for pair in i:
        ext_ans = pair["answers"]
        qs_en = pair["question"][0]["string"]
        out[qs_en] = ext_ans

    f.close()
    return out

def getTopSimi(sent: str, qs_pool: list):
    """
    This is a simple application for sentence embeddings: semantic search

    We have a corpus with various sentences. Then, for a given query sentence,
    we want to find the most similar sentence in this corpus.

    This script outputs for various queries the top 5 most similar sentences in the corpus.
    """


    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Corpus with example sentences
    corpus = qs_pool
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Query sentences:
    queries = [sent]


    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(corpus))
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        out = list()
        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx], "(Score: {:.4f})".format(score))
            out.append(corpus[idx])

    return out

def getTriple(sent:str) -> list:
    
    return None

def formatInstruc(sent: str, bg_pair: list, triples: typing.Optional[list] = None) -> str:
    q1, a1, q2, a2 = bg_pair[0][0], bg_pair[0][1], bg_pair[1][0], bg_pair[1][1]
    instruct = f"You'er an expert in logistics and events for manufactoring industry. Answer my question based on the context and your own knowledge: {sent} \n Here is some context knowledge you should know: \n Question: {q1} \n Answer: {a1} \n Questions: {q2} \n Answer: {a2} \n Relevant facts: {triples}"
    return instruct

    
def getANS(prompt:str) -> str:
    '''
    run with llama API 
    '''
    payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "key": "M7ZQL9ELMSDXXE86"}
    response = requests.post(url='https://turbo.skynet.coypu.org', headers={'Content-Type': 'application/json'}, json=payload).text
    ans = response.split("content")[-1].split('finish_reason')[0][3:-4]
    return ans


def main(pool_dataset: str, sent: str)-> str:
    pool_dataset = get_pair(pool_dataset)
    qs_pool = [qs for qs in pool_dataset.keys()]
    simiQs = getTopSimi(sent, qs_pool)[:2]
    simiPair = [[qs, pool_dataset[qs]] for qs in simiQs]
    prompt = formatInstruc(sent, simiPair)
    ans = getANS(prompt)
    return ans


if __name__ == "__main__":
    print(main('/export/home/yan/kgqa/dataset/coypu-questions-all.json', 'Does Tokyo have a sea port?"'))

