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
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from datetime import datetime, timedelta
import re, textwrap, string, os, time


openai_api_key = os.environ['OPENAI_API_KEY']
MONGO_URI = os.environ['MONGODB_IST_MEDIA']

client = MongoClient(MONGO_URI)
dbName = "1_media_demo"
collectionName = "business_news"
collection = client[dbName][collectionName]

gen_ai_cache_collectionName = "gen_ai_cache"
gen_ai_cache_collection = client[dbName][gen_ai_cache_collectionName]

MAX_DOCS_VS = 30  # number of results for vector search
MAX_DOCS = 8      # number of articles on the home page
MAX_RCOM = 3      # number of recommended articles


vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    dbName + "." + collectionName,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    index_name="vector_index",
)


def similarity_search(text: str, history: list[str], count=MAX_DOCS_VS) -> list[dict]:
    results = vector_search.similarity_search(query=text, k=MAX_DOCS_VS)
    docs = map(lambda r: r.metadata, results) # docs lack the 'text' field - WHY?
    # don't recommend history items, and limit to count
    return list(filter(lambda r: not r['uuid'] in history, docs))[:count]


def calculate_recommendations(embedding, history, count=MAX_DOCS_VS):
    results = vector_search._similarity_search_with_score(embedding=embedding, k=MAX_DOCS_VS)
    tuples = map(lambda r: (r[0].metadata, r[1]), results)
    # don't recommend history items, and limit to count
    return list(filter(lambda t: not t[0]['uuid'] in history, tuples))[:count]


def calculate_keywords(text: str) -> list[str]:
    lcdocs = [ Document(page_content=text, metadata={"source": "local"}) ]
    prompt_template = """Return a machine-readable Python list.
    Given the context of the media article, please provide
    me with 6 short keywords that capture the essence of the content and help
    finding the article while searching the web. Consider terms
    that are central to the article's subject and are likely to be imported for
    summarization. Please prioritize names of companies, names of persons,
    names of products, events, technical terms, business terms
    over general words.
    Return a machine-readable Python list.
    "{text}"
    KEYWORDS:"""
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text")
        keywords_string = stuff_chain.invoke(lcdocs)['output_text']
        keywords = eval(keywords_string) # convert str into list
        keywords = list(filter(lambda keyword: len(keyword) < 30, keywords))
        keywords = keywords[:7] # safety guard - sometimes OpenAI returns too much
    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        keywords = []
    return keywords


def calculate_insights(text: str) -> str:
    lcdocs = [ Document(page_content=text, metadata={"source": "local"}) ]
    prompt_template = """Tell me about the following, writing three paragraphs:
    "{text}"
    INSIGHTS:"""
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text")
        insights = stuff_chain.invoke(lcdocs)['output_text']
    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        insights = ""
    return insights


def calculate_using_rag(question: str) -> str:
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    try:
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_search.as_retriever()
        model = ChatOpenAI(model_name="gpt-4-turbo")
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


def calculate_using_rag_and_return_context(question):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    try:
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_search.as_retriever(search_kwargs={ "k" : 3 })
        model = ChatOpenAI(model_name="gpt-3.5-turbo")
        chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | model
            | StrOutputParser()
        )
        chain_with_source = RunnableParallel(
            { "context": retriever, "question": RunnablePassthrough() }
        ).assign(answer=chain)
        result = chain_with_source.invoke(question)
        answer = result['answer']
        cx = result['context']
        context = []
        for c in cx:
            context.append(collection.find_one({ "uuid" : c.metadata['uuid'] }))
        gen_ai_cache_collection.insert_one({ "question" : question, "answer" : answer })
    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        answer = ""
    return answer, context


def capitalize(text: str) -> str:
    return ' '.join([w.title() if w.islower() else w for w in text.split()])


def check_for_quality_read():
    if 'uuid_since' in session:
        delta_seconds = int(time.time()) - session['uuid_since']
        session.pop('uuid_since')
        if delta_seconds > 10 and delta_seconds < 300:
            try:
                collection.update_one({ "uuid" : session['uuid'] },
                                      { "$inc" : { "read_count" : 1 }})
            except Exception as e:
                print(e)


@main.route('/delete_articles_history', methods=['POST'])
def delete_articles_history():
    session['history'] = []
    return redirect('/backstage')


@main.route('/delete_insights_history_item/<id>', methods=['GET'])
def delete_insights_history_item(id):
    try:
        gen_ai_cache_collection.delete_one({ "_id" : ObjectId(id) })
    except Exception as e:
        pass
    return redirect('/backstage')


@main.route('/welcome')
def welcome():
    check_for_quality_read()
    return render_template('welcome.html')


@main.route('/')
def index():
    if not 'was_here_before' in session:
        session['was_here_before'] = '1'
        return render_template('welcome.html')
    check_for_quality_read()
    query = request.args.get('query')
    if query and query != "":
        docs = similarity_search(query.strip(), [], MAX_DOCS)
        # for unknown reasons, these docs lack the 'text' field - refetching...
        docs = list(map(lambda doc: collection.find_one({ "uuid" : doc['uuid'] }), docs))
        infoline = 'Showing best-matching articles for vector search: "' + query.strip() + '"'
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
            docs = similarity_search(concatenated_titles, session['history'], MAX_DOCS)
            # for unknown reasons, these docs lack the 'text' field - refetching...
            docs = list(map(lambda doc: collection.find_one({ "uuid" : doc['uuid'] }), docs))
            infoline = "Showing personalized content, based on browsing history"
        else: # no personalization possible - shuffle some items to start with
            docs = collection.aggregate([
                { "$sample": { "size": MAX_DOCS } }
            ])
            infoline = "Showing random content, no browsing history yet"
    # prepare for a nice view
    docs = list(map(lambda doc: doc | {
        'fdate' : datetime.fromisoformat(doc["published"]).strftime("%d %b %Y"),
        'ftext' : textwrap.shorten(doc['text'] if 'text' in doc else 'No content.', 450) }, docs))
    return render_template('index.html', docs=docs, infoline=infoline)


@main.route('/post')
def post():
    check_for_quality_read()
    style = request.args.get('style')
    query = request.args.get('query')
    uuid = request.args.get('uuid')
    if query and query != "":
        return index()
    if style and style == "summary":
        doc = collection.find_one({ "uuid" : session['uuid'] })
        if not doc:
            return render_template('404.html')
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
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
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
        recommendations = calculate_recommendations(doc['text'], session['history'], MAX_RCOM)
        return render_template('post.html', doc=doc, fdate=fdate,
                               fdoc=fdoc, recommendations=recommendations)
    else:
        if uuid: # highest prio: use uuid page parameter, if provided
            doc = collection.find_one({ "uuid" : uuid })
        elif 'uuid' in session: # session-saved uuid as the second choice
            doc = collection.find_one({ "uuid" : session['uuid'] })
        else: # TODO: use a personalized doc, if history is not empty
            doc = list(collection.aggregate([
                { "$sample": { "size": 1 } }
            ]))[0]
        if not doc:
            return render_template('404.html')
        sdate = doc["published"]
        fdate = datetime.fromisoformat(sdate).strftime("%a %d %b %Y %H:%M")
        fdoc = html(doc['text'].replace("\n", "\n\n"))
        session['uuid'] = doc['uuid']
        session['uuid_since'] = int(time.time())
        if not 'history' in session:
            session['history'] = []
        session['history'].append(doc['uuid']) # YES, allow for dups!
        # to make the floating history work, clicked items ALWAYS must be
        # appended at the end
        if len(session['history']) > 30: # limit the max length of history
            session['history'] = session['history'][-30:]
        if not 'keywords' in doc or len(doc['keywords']) == 0:
            keywords = calculate_keywords(doc['text'])
            if len(keywords) > 0:
                print("INFO: Caching keywords for uuid " + doc['uuid'])
                try:
                    collection.update_one({ "_id" : doc["_id"] },
                                          { "$set" : { "keywords" : keywords }})
                except Exception as e:
                    print(e)
        else:
            keywords = doc['keywords']
        recommendations = calculate_recommendations(doc['embedding'], session['history'], MAX_RCOM)
        collection.update_one({ "_id" : doc["_id"] },
                              { "$inc" : { "visit_count" : 1 }})
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc,
                               recommendations=recommendations, keywords=keywords,
                               visit_count=doc['visit_count']+1 if 'visit_count' in doc else 1,
                               read_count=doc['read_count'] if 'read_count' in doc else 0)


@main.route('/backstage')
def about():
    check_for_quality_read()
    if not 'history' in session:
        session['history'] = []
    docs = list(map(lambda uuid:
                    collection.find_one({ "uuid" : uuid },
                                        { "uuid" : 1, "title" : 1, "_id" : 0 }),
                    reversed(session['history'])))
    try:
        gen_ai_cache = list(gen_ai_cache_collection.find().sort({ "$natural" : -1 }).limit(100))
    except Exception as e:
        gen_ai_cache = []
        print(e) # will be printed in the log file that is residing in /tmp
    return render_template('about.html', history=docs, gen_ai_cache=gen_ai_cache)


@main.route('/insights')
def insights():
    check_for_quality_read()
    keyword = request.args.get('keyword')
    _id = request.args.get('_id')
    query = request.args.get('query')
    try:
        gen_ai_cache = list(gen_ai_cache_collection.find().sort({ "$natural" : -1 }).limit(20))
    except Exception as e:
        gen_ai_cache = []
        print(e) # will be printed in the log file that is residing in /tmp
    if keyword and keyword != "":
        content = html(calculate_insights(keyword))
        title = capitalize(keyword)
        return render_template('insights.html', placeholder="AI-Generated Insights (RAG)",
                               title=title, content=content, gen_ai_cache=gen_ai_cache)
    elif _id and _id != "":
        try:
            cached_entry = gen_ai_cache_collection.find_one({ "_id" : ObjectId(_id) })
        except Exception as e:
            cached_entry = { "question" : "AI-Generated Insights (RAG)" ,
                             "answer" : "Error reading cached Q/A"}
            print(e) # will be printed in the log file that is residing in /tmp
        content = html(cached_entry['answer'])
        title = capitalize(cached_entry['question'])
        return render_template('insights.html', placeholder="AI-Generated Insights (RAG)",
                               title=title, content=content, gen_ai_cache=gen_ai_cache)
    elif query and query != "":
        query = query.strip()
        cached_entry = None
        try:
            cached_entry = gen_ai_cache_collection.find_one({ "question" : query })
        except Exception as e:
            print(e) # will be printed in the log file that is residing in /tmp
        if cached_entry:
            content = html(cached_entry['answer'])
            title = capitalize(cached_entry['question'])
        else:
            content = html(calculate_using_rag(query))
            title = capitalize(query)
        return render_template('insights.html', placeholder="AI-Generated Insights (RAG)",
                               title=title, content=content, gen_ai_cache=gen_ai_cache)
    else:
        most_read_articles = collection.find().sort({ 'read_count' : -1 }).limit(5)
        return render_template('insights.html', placeholder="AI-Generated Insights (RAG)",
                               title="AI-Generated Insights (RAG)",
                               content="""<p>Please enter your question in the form above.
                               Answer will be provided based only on the documents
                               stored in MongoDB, using Vector Search and OpenAI GPT-4.</p>

                               <p>This page is also used to display AI-generated insights
                               when clicking on a keyword in the Single Post page. In
                               that case, general knowledge from GPT-3.5 is being used,
                               without RAG.</p>

                               <p>GPT-4 is currently still a very expensive model,
                               so please use this part of the demo with care. Also,
                               calculation can take some time. You have to be patient, and
                               please avoid refreshing the page or re-entering the question.</p>

                               <p>Most recent insights are cached, and can be accessed
                               from the right column of this page. Consider using these
                               examples when conducting a demo!</p>
                               """,
                               gen_ai_cache=gen_ai_cache,
                               most_read_articles=most_read_articles)


@main.route('/contact')
def contact():
    check_for_quality_read()
    return render_template('contact.html')
