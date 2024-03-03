#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import render_template, request, session
from .. import mongo
from . import main
from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from datetime import datetime
import re
import textwrap


openai_api_key="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"
MONGO_URI="mongodb+srv://mediademouser:mediademouserpassword\
@ist-shared.n0kts.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
dbName = "1_media_demo"
collectionName = "business_news"
collection = client[dbName][collectionName]
#embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
#    MONGO_URI,
#    dbName + "." + collectionName,
#    OpenAIEmbeddings(disallowed_special=()),
#    index_name="vector_index")

@main.route('/')
def index():
    query = request.args.get('query')
    if query and query != "":
        pat = re.compile(query, re.I)
        doc = collection.find_one({ "thread.site" : "bnnbreaking.com",
                                "text" : {'$regex': pat}})
        fdate = datetime.fromisoformat(doc["published"]).strftime("%d %b %Y")
        fdoc = textwrap.shorten(doc['text'], 450)
        return render_template('index.html', doc=doc, fdate=fdate, fdoc=fdoc)
    else:
        doc = collection.find_one({ "thread.site" : "bnnbreaking.com" })
        fdate = datetime.fromisoformat(doc["published"]).strftime("%d %b %Y")
        fdoc = textwrap.shorten(doc['text'], 450)
        return render_template('index.html', doc=doc, fdate=fdate, fdoc=fdoc)


@main.route('/post')
def post():
    style = request.args.get('style')
    query = request.args.get('query')
    uuid = request.args.get('uuid')
    if query and query != "":
        pat = re.compile(query, re.I)
        doc = collection.find_one({ "thread.site" : "bnnbreaking.com",
                                    "text" : {'$regex': pat}})
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc=doc['text']
        session['uuid'] = doc['uuid']
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc)
    if style and style == "original":
        doc = collection.find_one({ "uuid" : session['uuid'] })
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc=doc['text']
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc)
    if style and style == "summary":
        doc = collection.find_one({ "uuid" : session['uuid'] })
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc=doc['text']
        lcdocs = [
            Document(page_content=doc['text'], metadata={"source": "local"})
        ]
        # Define prompt
        prompt_template = """Summarize in around 200 words the key facts of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        # Define LLM chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,
                                          document_variable_name="text")
        fdoc = stuff_chain.invoke(lcdocs)['output_text']
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc)
    if style and style == "young":
        doc = collection.find_one({ "uuid" : session['uuid'] })
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc=doc['text']
        lcdocs = [
            Document(page_content=doc['text'], metadata={"source": "local"})
        ]
        # Define prompt
        prompt_template = """Rewrite for a 12 year old person:
        "{text}"
        REFORMATTED:"""
        prompt = PromptTemplate.from_template(prompt_template)
        # Define LLM chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,
                                          document_variable_name="text")
        fdoc = stuff_chain.invoke(lcdocs)['output_text']
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc)
    if style and style == "german":
        doc = collection.find_one({ "uuid" : session['uuid'] })
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc=doc['text']
        lcdocs = [
            Document(page_content=doc['text'], metadata={"source": "local"})
        ]
        # Define prompt
        prompt_template = """Summarize in around 150 words the key facts of the following and translate the summary to German:
        "{text}"
        TRANSLATION:"""
        prompt = PromptTemplate.from_template(prompt_template)
        # Define LLM chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,
                                          document_variable_name="text")
        fdoc = stuff_chain.invoke(lcdocs)['output_text']
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc)
    if style and style == "arabic":
        doc = collection.find_one({ "uuid" : session['uuid'] })
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc=doc['text']
        lcdocs = [
            Document(page_content=doc['text'], metadata={"source": "local"})
        ]
        # Define prompt
        prompt_template = """Summarize in around 250 words the key facts of the following and translate the summary to Arabic:
        "{text}"
        TRANSLATION:"""
        prompt = PromptTemplate.from_template(prompt_template)
        # Define LLM chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,
                                          document_variable_name="text")
        fdoc = stuff_chain.invoke(lcdocs)['output_text']
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc)
    else:
        doc = collection.find_one({ "uuid" : uuid })
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc=doc['text']
        session['uuid'] = doc['uuid']
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc)

    
@main.route('/about')
def about():
    return render_template('about.html')


@main.route('/contact')
def contact():
    return render_template('contact.html')


@main.route('/content')
def yyy():
    topic1 = request.args.get('topic1')
    topic2 = request.args.get('topic2')
    return render_template('content.html', topic1=topic1, topic2=topic2)

@main.route('/about/<topic>')
def xxx(topic):
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.invoke("Tell me news about " + topic + ". Mention just news from January 20th 2024 or newer, including speculations, success, failures, illness, private stories. Write in a style that works well for somebody who wants to get entertained. Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION.")
    content = retriever_output['result'].strip()
    return render_template('about.html', content=content, topic=topic)

