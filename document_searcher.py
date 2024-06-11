from sklearn.feature_extraction.text import TfidfVectorizer
from keys import COHERE_API_KEY
import cohere
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
cohere_key = COHERE_API_KEY
co = cohere.Client(cohere_key)
gt = pd.read_excel('data/ground_truth.xlsx')


def letter_tokenizer(word,n = 3):
    tokens = []
    for i in range(len(word) - n + 1):
        tokens.append(word[i:i+n])
    return tokens

def preprocess_text(text):
    tokens = letter_tokenizer(text)
    return ' '.join(tokens)


def generate_prompt(q,retrieved_documents,group_key):
    if retrieved_documents == None:
        return "Not confident enough to generate prompt"
    else:
        context_string = ""
        for section,context in retrieved_documents.items():
            context_string += '-------' + group_key[section] + '-------'
            context_string += ' '.join(context)
            context_string += '\n'
    
        prompt = """ 
    Based on the context below answer the given question. Given Question: {}
    ----------------------------- Context -----------------------------
    {}
        """.format(q,context_string) 
    
        return prompt

def rerank_documents(q,context_group,frac = 0.7):
    if context_group == None:
        return None
    reranked_context = {}
    for group,context in context_group.items():
        n = round(frac*len(context))
        if n > 0:
            res = co.rerank(query = q, documents = context, top_n = n , model = 'rerank-multilingual-v3.0')
            reranked_list = []
            for doc in res.results:
              doc = context[doc.index]
              reranked_list.append(doc)
            reranked_context[group] = reranked_list
        else:
            reranked_context[group] = []
    return reranked_context

class DocSearcher:
    def __init__(self,docs,embedding_model):
        docs['context_embedding'] = docs['Question'].apply(lambda x: embedding_model.embed_query(x))
        docs['Group_short'] = docs['Group_short'].astype(str)
        group_key = docs[['Group_short','Group']].drop_duplicates()
        group_key = dict(zip(group_key['Group_short'], group_key['Group']))
        self.emb = embedding_model
        self.docs = docs
        self.group_key = group_key

    def _get_context_from_scores(self,section_scores,num_section = 5,k = 15):
        docs = self.docs
        group_key = self.group_key
        is_confident = True
        for section,score in section_scores.items():
            if score < 0:
                section_scores[section] = 0
        section_scores = sorted(section_scores.items(), key=lambda x: x[1],reverse = True)
        total_scores = sum(value for key, value in section_scores[0:num_section])
        if total_scores <= 0.25:
            is_confident = False
            context = None
            return is_confident, context
        else:
            context = {}
            for score in section_scores[0:num_section]:
                section = score[0]
                filtered_docs = docs[docs['Group_short']==section].reset_index(drop = True)
                n_documents = round(k*score[1]/total_scores)
                chosen_row = filtered_docs.iloc[0:n_documents]
                context[section] = chosen_row['q_a'].tolist()
            return is_confident,context     

    def query_documents(self,q,k = 15,method = 'all',num_section = 5):
        docs = self.docs
        docs['q_a'] = '\n Question: ' + docs['Question'] + '\n Answer: ' + docs['Answer']
        if method == 'all':
            scores_tfidf = self.get_section_tfidf_scores(q)
            scores_vector = self.get_section_vector_scores(q)
            scores_combined = {section: (scores_tfidf[section] + scores_vector[section])/2 for section in scores_vector}
            is_confident, context = self._get_context_from_scores(section_scores = scores_combined,num_section = num_section,k = k)
            if is_confident == True:
                return context
            else:
                return None

        elif method == 'tfidf':
            scores_tfidf = self.get_section_tfidf_scores(q)
            is_confident, context = self._get_context_from_scores(section_scores = scores_tfidf,num_section = num_section,k = k)
            if is_confident == True:
                return context
            else:
                return None
            
        elif method == 'vector':
            scores_vector = self.get_section_vector_scores(q)
            is_confident, context = self._get_context_from_scores(section_scores = scores_vector,num_section = num_section,k = k)
            if is_confident == True:
                return context
            else:
                return None

        elif method == 'fuzzy':
            pass
        else:
            raise NameError("Please input the correct method")

    def query_documents_with_prompt(self,q,k = 15,method = 'all'):
        context = self.query_documents(q,k,method)
        if context != None: 
            group_key = self.group_key
            context_string = ""
            for section,doc in context.items():
                context_string += '-------' + group_key[section] + '-------'
                context_string += ' '.join(doc)
                context_string += '\n'
            prompt = """
        Base the context below answer the given question. Given Question: {}
        ------------------------- Context -----------------------------
        {}
            """.format(q,context_string)
            return prompt
        else:
            return "Not confident enough to generate prompt"

    # Whole document scores
    def get_section_tfidf_scores(self,q):
        docs = self.docs
        docs_grouped = docs.groupby('Group_short')['Question'].apply(lambda x: ' '.join(x)).reset_index()
        docs_grouped['docs'] = docs_grouped['Group_short']+' '+docs_grouped['Question']
        docs_grouped['cleaned_document'] = docs_grouped['Question'].apply(preprocess_text)
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs_grouped['cleaned_document'])
        cleaned_query = preprocess_text(q)
        query_vector = tfidf_vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix)
        relevant_docs_indices = similarities.argsort()[0][::-1]
        scores_tfidf = {}
        for i, idx in enumerate(relevant_docs_indices):
            scores_tfidf[docs_grouped.iloc[idx,:]['Group_short']] = similarities[0][idx]     
        return scores_tfidf

    def get_section_vector_scores(self,q):
        emb = self.emb
        docs = self.docs 
        query_embedding = np.array(emb.embed_query(q)).reshape(1,-1)
        context_embeddings = np.array(docs['context_embedding'].tolist())
        scores_vector = cosine_similarity(context_embeddings, query_embedding).flatten()
        docs['similarity_score_vector'] = scores_vector
        docs = docs.sort_values(by = 'similarity_score_vector',ascending = False) 
        self.docs = docs
        scores_vector = docs.groupby(['Group_short'])['similarity_score_vector'].mean().sort_values(ascending = False).to_dict()
        return scores_vector

