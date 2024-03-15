#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import render_template, redirect, request, session
from mistune import html
from .. import mongo
from . import main
from bson.objectid import ObjectId
from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from datetime import datetime
import re
import textwrap
import string


openai_api_key="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"
MONGO_URI="mongodb+srv://mediademouser:mediademouserpassword\
@ist-shared.n0kts.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
dbName = "1_media_demo"
collectionName = "business_news"
collection = client[dbName][collectionName]

gen_ai_cache_collectionName = "gen_ai_cache"
gen_ai_cache_collection = client[dbName][gen_ai_cache_collectionName]

MAX_DOCS = 8     # used for page views
MAX_DOCS_VS = 30 # used for vector search

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    dbName + "." + collectionName,
    OpenAIEmbeddings(disallowed_special=()),
    index_name="vector_index",
)


def calculate_recommendations(text, history):
    vs_results = vector_search.similarity_search(query=text, k=MAX_DOCS_VS)
    rcom = map(lambda r: r.metadata, vs_results)
    # don't recommend historic items
    rcom = list(filter(lambda r: not r['uuid'] in history, rcom))
    return rcom # TODO: These documents lack the 'text' field - WHY?


def calculate_keywords(text):
    lcdocs = [ Document(page_content=text, metadata={"source": "local"}) ]
    prompt_template = """Given the context of the media article, please provide
    me with 6 keywords that capture the essence of the content and help
    finding the article while searching the web. Consider terms
    that are central to the article's subject and are likely to be imported for
    summarization. Please prioritize names of companies, names of persons,
    names of products, events, technical terms, business terms
    over general words.
    Return as a machine-readable Python list.
    "{text}"
    KEYWORDS:"""
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text")
        keywords_string = stuff_chain.invoke(lcdocs)['output_text']
        print(keywords_string) # audit if OpenAI returns the right format
        keywords = eval(keywords_string) # convert str into list
        keywords = list(filter(lambda keyword: len(keyword) < 30, keywords))
        keywords = keywords[:7] # safety guard - sometimes OpenAI returns too much
    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        keywords = []
    return keywords


def calculate_insights(text):
    lcdocs = [ Document(page_content=text, metadata={"source": "local"}) ]
    prompt_template = """Tell me about the following, writing three paragraphs:
    "{text}"
    INSIGHTS:"""
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text")
        insights = stuff_chain.invoke(lcdocs)['output_text']
    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        insights = ""
    return insights


def calculate_using_rag(question):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    try:
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_search.as_retriever()
        model = ChatOpenAI(model_name="gpt-4-turbo-preview")
        chain = (
            { "context": retriever, "question": RunnablePassthrough() }
            | prompt
            | model
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        gen_ai_cache_collection.insert_one({ "question" : question, "answer" : answer })
    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        answer = ""
    return answer


@main.route('/delete_history', methods=['POST'])
def delete_history():
    session['history'] = []
    return redirect('/backstage')


@main.route('/')
def index():
    query = request.args.get('query')
    if query and query != "":
        pat = re.compile(query, re.I)
        docs = collection.find({ "thread.site" : "bnnbreaking.com",
                                 "text" : {'$regex': pat}}).limit(MAX_DOCS)
    else: # the start page, called without query parameter
        if 'history' in session and len(session['history']) > 0:
            history_docs = list(map(lambda uuid:
                    collection.find_one({ "uuid" : uuid },
                                        { "uuid" : 1, "title" : 1, "_id" : 0 }),
                    session['history']))
            concatenated_titles = ""
            i = 0
            for doc in history_docs[-5:]: # only consider the recent history
                concatenated_titles += ("" if i == 0 else " ") + doc['title']
                i += 1
            print("[DEBUG]: Personalization with doc titles: " + concatenated_titles)
            docs = calculate_recommendations(concatenated_titles, session['history'])
            # for unknown reasons, these docs lack the 'text' field - refetching...
            docs = list(map(lambda doc:
                            collection.find_one({ "uuid" : doc['uuid'] }), docs))
            if len(docs) > MAX_DOCS:
                docs = docs[:MAX_DOCS]
        else: # no personalization possible - shuffle some items to start with
            docs = collection.aggregate([
                { "$match": { "thread.site" : "bnnbreaking.com" } },
                { "$sample": { "size": MAX_DOCS } }
            ])
    # prepare for a nice view
    docs = list(map(lambda doc: doc | {
        'fdate' : datetime.fromisoformat(doc["published"]).strftime("%d %b %Y"),
        'ftext' : textwrap.shorten(doc['text'] if 'text' in doc else 'No content.', 450) }, docs))
    return render_template('index.html', docs=docs)


@main.route('/post')
def post():
    style = request.args.get('style')
    query = request.args.get('query')
    uuid = request.args.get('uuid')

    if query and query != "":
        return index()
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
        try:
            prompt = PromptTemplate.from_template(prompt_template)
            # Define LLM chain
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            # Define StuffDocumentsChain
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="text")
            fdoc = "SUMMARY: " + stuff_chain.invoke(lcdocs)['output_text']
            fdoc = html(fdoc.replace("\n", "\n\n"))
        except Exception as e:
            fdoc = "OpenAI crashed - you should try again (later)"
            print(e) # will be printed in the log file that is residing in /tmp
        rcom = calculate_recommendations(doc['text'], session['history'])
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc, rcom=rcom)
    else:
        if uuid: # highest prio: use uuid page parameter, if provided
            doc = collection.find_one({ "uuid" : uuid })
        elif 'uuid' in session: # session-saved uuid as the second choice
            doc = collection.find_one({ "uuid" : session['uuid'] })
        else: # TODO: use a personalized doc, if history is not empty
            doc = list(collection.aggregate([
                { "$match": { "thread.site" : "bnnbreaking.com" } },
                { "$sample": { "size": 1 } }
            ]))[0]
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc = html(doc['text'].replace("\n", "\n\n"))
        session['uuid'] = doc['uuid']
        if not 'history' in session:
            session['history'] = []
        session['history'].append(doc['uuid']) # YES, allow for dups!
        # to make the floating history work, clicked items ALWAYS must be
        # appended at the end
        if len(session['history']) > 30: # limit the max length of history
            session['history'] = session['history'][-30:]
        keywords = calculate_keywords(doc['text'])
        rcom = calculate_recommendations(doc['text'], session['history'])
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc,
                               rcom=rcom, keywords=keywords)


@main.route('/backstage')
def about():
    if not 'history' in session:
        session['history'] = []
    docs = list(map(lambda uuid:
                    collection.find_one({ "uuid" : uuid },
                                        { "uuid" : 1, "title" : 1, "_id" : 0 }),
                    session['history']))
    try:
        gen_ai_cache = list(gen_ai_cache_collection.find())
    except Exception as e:
        gen_ai_cache = []
        print(e) # will be printed in the log file that is residing in /tmp
    return render_template('about.html', history=docs, gen_ai_cache=gen_ai_cache)


@main.route('/insights')
def insights():
    query = request.args.get('query')
    if query and query != "":
        content = html(calculate_insights(query))
        title = ' '.join([w.title() if w.islower() else w for w in query.split()])
        return render_template('insights.html', html_title="AI-Generated Insights",
                               title=title, content=content)
    else:
        return render_template('insights.html', html_title="AI-Generated Insights",
                               title="No Title", content="No insights.")


@main.route('/rag')
def rag():
    _id = request.args.get('_id')
    query = request.args.get('query')
    if _id and _id != "":
        try:
            cached_entry = gen_ai_cache_collection.find_one({ "_id" : ObjectId(_id) })
        except Exception as e:
            cached_entry = { "question" : "AI-Generated Insights (RAG)" ,
                             "answer" : "Error reading cached Q/A"}
            print(e) # will be printed in the log file that is residing in /tmp
        content = html(cached_entry['answer'])
        title = ' '.join([w.title() if w.islower() else w for w in cached_entry['question'].split()])
        return render_template('rag.html', placeholder="AI-Generated Insights (RAG)",
                               title=title, content=content)
    elif query and query != "":
        cached_entry = None
        try:
            cached_entry = gen_ai_cache_collection.find_one({ "question" : query })
        except Exception as e:
            print(e) # will be printed in the log file that is residing in /tmp
        if cached_entry:
            content = html(cached_entry['answer'])
            title = ' '.join([w.title() if w.islower() else w for w in cached_entry['question'].split()])
        else:
            content = html(calculate_using_rag(query))
            title = ' '.join([w.title() if w.islower() else w for w in query.split()])
        return render_template('rag.html', placeholder="AI-Generated Insights (RAG)",
                               title=title, content=content)
    else:
        return render_template('rag.html', placeholder="AI-Generated Insights (RAG)",
                               title="AI-Generated Insights (RAG)",
                               content="""<p>Please enter your question in the form above.
                               Answer will be provided based only on the documents
                               stored in MongoDB, using Vector Search and OpenAI GPT-4.</p>

                               <p>GPT-4 is currently still a very expensive model,
                               so please use this part of the demo with care. Also,
                               calculation can take some time. You have to be patient, and
                               please avoid refreshing the page or re-entering the question.</p>
                               """)


@main.route('/contact')
def contact():
    return render_template('contact.html')
