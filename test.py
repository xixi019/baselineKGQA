import requests
import json
import ipdb
from sentence_transformers import SentenceTransformer, util
import torch
import typing
from flair.data import Sentence
from flair.nn import Classifier
from SPARQLWrapper import SPARQLWrapper, JSON
import logging

logger = logging.getLogger('my-logger')
logger.propagate = False

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
    if len(corpus) > 0:
        corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

        # Query sentences:
        queries = [sent]


        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(10, len(corpus))
        for query in queries:
            query_embedding = embedder.encode(query, convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

    #        print("\n\n======================\n\n")
    #        print("Query:", query)
    #        print("\nTop 5 most similar sentences in corpus:")

            out = list()
            for score, idx in zip(top_results[0], top_results[1]):
    #            print(corpus[idx], "(Score: {:.4f})".format(score))
                out.append(corpus[idx])
    else:
        out = []
    return out

def getTriple(sent:str) -> list:
    
    '''
    extract the entities with flair NLP
    Then link the entities with string match in sparql query.
    return a list of dictionary, with mention as key and list of ids as value
    '''
    ents = list()
    sentence = Sentence(sent)
    tagger = Classifier.load('ner')
    tagger.predict(sentence)
    for entity in sentence.get_spans('ner'):
        ent_str = entity.text
        ents.append(ent_str)
    # a dictionary with mention as key and list of IDs in coypu graph as value
    linked_ent = entLink(ents)
    return linked_ent

def entLink(ents : list):
    ''' 
    easy entity linking module with SPARQL
    '''
    sparql = SPARQLWrapper(
        "http://skynet.coypu.org/coypu-internal"
    )
    sparql.setCredentials("katherine", "0CyivAlseo")
    sparql.setReturnFormat(JSON)

    # gets the first 3 geological ages
    # from a Geological Timescale database,
    # via a SPARQL endpoint
    out = {}
    for ent in ents:
        out[ent] = []
        query = """
            PREFIX text: <http://jena.apache.org/text#>

            SELECT DISTINCT ?inst WHERE {
            (?inst ?score) text:query ("Macao" 10).
            }
            """.replace("Macao", ent)
        sparql.setQuery(query)

        try:
            ret = sparql.queryAndConvert()

            for r in ret["results"]["bindings"]:
                out[ent].append(r["inst"]["value"])
        except Exception as e:
            print(e)

    for mention in out:
        out[mention] = [i for i in out[mention] if "wikidata" not in i]

    return out

def queryTriple(linked_ent:dict):
    ''' 
    input is output from entLink
    extract the triples based on the entity, top 10 triples 
    '''
    sparql = SPARQLWrapper(
        "http://skynet.coypu.org/coypu-internal"
    )
    sparql.setCredentials("katherine", "0CyivAlseo")
    sparql.setReturnFormat(JSON)


    out = {}
    for mention in linked_ent:
        out[mention] = []
        for id in linked_ent[mention]:
            query = """
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT *
                WHERE {
                <entity> ?y ?z . 
                OPTIONAL{ 
                        ?y rdf:type ?type 
                        FILTER( ?type != "rdf:Property" ) }
                        } 
                LIMIT 10                
                
                """.replace("entity", id)
            sparql.setQuery(query)
            try:
                ret = sparql.queryAndConvert()
            

                for r in ret["results"]["bindings"]:
                    out[mention].append([r["y"]["value"], r["z"]["value"]])
            except Exception as e:
                print(e)

    return out

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


def getLabel(id:str):
    '''
    input is the wikidata relation id, return the label of it
    '''
    return 

def main(pool_dataset: str, sent: str)-> str:
    # get similar question answer pairs
    pool_dataset = get_pair(pool_dataset)
    qs_pool = [qs for qs in pool_dataset.keys()]
    simiQs = getTopSimi(sent, qs_pool)[:2]
    simiPair = [[qs, pool_dataset[qs]] for qs in simiQs]

    # get the triples
    ents = entLink(getTriple(sent))
    if len(ents) >=1:
        triples = queryTriple(ents)
        for ent in triples:
            linTrip = [ ent + " is " + i[0].split("#")[-1] + " "+ i[1]for i in triples[ent] if ent not in i[1]]
            if len(linTrip) >=1 :
                topTrip = getTopSimi(sent, linTrip)
            else:
                topTrip = ""
    else:
        topTrip = ""

    prompt = formatInstruc(sent, simiPair, triples= topTrip)
    ans = getANS(prompt)
    return ans


if __name__ == "__main__":
    pool = get_pair('/export/home/yan/kgqa/dataset/coypu-questions-all.json')
    for qs, ans in {qs:pool[qs] for qs in list(pool.keys())[31:32]}.items():
        print(qs)
        print(main('/export/home/yan/kgqa/dataset/coypu-questions-all.json', qs))
    

