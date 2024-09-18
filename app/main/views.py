#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import render_template, redirect, request, session, url_for
from flask import current_app as app
from mistune import html
from .. import mongo, logger
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
from urllib.parse import urlparse
import re, textwrap, string, os, time, uuid as python_uuid, requests, geocoder, pycountry


MONGO_URI = os.environ['MONGODB_IST_MEDIA']

client = MongoClient(MONGO_URI)
dbName = "1_media_demo"

gen_ai_cache_collection = client[dbName]["gen_ai_cache"]
ip_info_cache_collection = client[dbName]["ip_info_cache"]
access_log_collection = client[dbName]["access_log"]

MAX_DOCS_VS = 30  # number of results for vector search
MAX_DOCS = 8      # number of articles on the home page
MAX_RCOM = 3      # number of recommended articles


def debug(msg: str):
    if app.config['DEBUG']:
        print("[DEBUG]: " + msg)


def read_collection_from_session():
    if not 'news_source' in session:
        session['news_source'] = 'business_news' # that's the default for now
    return session['news_source']


def collection():
    return client[dbName][read_collection_from_session()]


def log(request):
    ip = request.environ.get("X-Real-IP", request.remote_addr)
    loc = ip_info_cache_collection.find_one({ "ip" : ip })
    if not loc:
        debug("Retrieving IP location information for " + ip + " with geocoder web service")
        ws = geocoder.ip(ip)
        loc = { "ip" : ip,
                "city" : ws.city if ws.city else " - ",
                "country" : ws.country if ws.country else " - " }
        ip_info_cache_collection.insert_one(loc)
    logger.info(ip + " (" + loc['city'] + ", " + loc['country'] + "): " + request.base_url +
            (" (" + str(request.content_length) + " bytes)" if request.content_length else ""))
    try:
        if loc['country'] != " - ":
            log_entry = { "timestamp" : datetime.utcnow(),
                          "path" : urlparse(request.base_url).path,
                          "ip" : loc['ip'],
                          "city" : loc['city'],
                          "country" : loc['country']}
        else:
            log_entry = { "timestamp" : datetime.utcnow(),
                          "path" : urlparse(request.base_url).path,
                          "ip" : loc['ip']}
        access_log_collection.insert_one(log_entry)
    except Exception as e:
        print(e)
    return loc


def vector_search():
    return MongoDBAtlasVectorSearch.from_connection_string(
        MONGO_URI,
        dbName + "." + read_collection_from_session(),
        OpenAIEmbeddings(disallowed_special=()),
        index_name="vector_index")


def similarity_search(text: str,
                      history: list[str], count=MAX_DOCS_VS) -> list[dict]:
    results = vector_search().similarity_search(query=text, k=MAX_DOCS_VS)
    docs = map(lambda r: r.metadata, results) # docs lack the 'text' field - WHY?
    # don't recommend history items, and limit to count
    return list(filter(lambda r: not r['uuid'] in history, docs))[:count]


def calculate_recommendations(embedding: list[float],
                              history: list[str], count=MAX_DOCS_VS) -> list[tuple]:
    results = vector_search()._similarity_search_with_score(embedding=embedding, k=MAX_DOCS_VS)
    tuples = map(lambda r: (r[0].metadata, r[1]), results)
    # don't recommend history items, and limit to count
    return list(filter(lambda t: not t[0]['uuid'] in history, tuples))[:count]


def calculate_keywords(doc: dict) -> list[str]:
    service_url = app.config['API_BASE_URL'] + '/keywords'
    response = requests.post(service_url,
                             json = { "text" : doc['text'],
                                      "llm" : app.config['AVAILABLE_LLMS']['OpenAI GPT-3.5'] })
    try:
        keywords = response.json()['keywords']
    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        keywords = []

    if len(keywords) > 0:
        try:
            collection().update_one({ "_id" : doc["_id"] },
                                  { "$set" : { "keywords" : keywords }})
            debug("keywords cached for uuid " + doc['uuid'])
        except Exception as e:
            print(e)
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
    logger.info(question)
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    try:
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_search().as_retriever()
        model = ChatOpenAI(model_name="gpt-4o")
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
        retriever = vector_search().as_retriever(search_kwargs={ "k" : 3 })
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
            context.append(collection().find_one({ "uuid" : c.metadata['uuid'] }))
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
                collection().update_one({ "uuid" : session['uuid'] },
                                      { "$inc" : { "read_count" : 1 }})
            except Exception as e:
                print(e)


@main.route('/select_news_source', methods=['POST'])
def select_news_source():
    selected_option = request.form['news_option']
    session['news_source'] = 'business_news' if selected_option == 'traditional' else 'news'
    # need to reset some stuff when switching source
    delete_articles_history()
    if 'uuid' in session:
        session.pop('uuid')
    if 'uuid_since' in session:
        session.pop('uuid_since')
    return redirect('/')


@main.route('/delete_articles_history', methods=['POST'])
def delete_articles_history():
    log(request)
    session['history'] = []
    return redirect('/backstage')


@main.route('/delete_articles_history_from_homepage', methods=['GET'])
def delete_articles_history_from_homepage():
    log(request)
    session['history'] = []
    return redirect('/')


@main.route('/delete_insights_history_item/<id>', methods=['GET'])
def delete_insights_history_item(id):
    log(request)
    try:
        gen_ai_cache_collection.delete_one({ "_id" : ObjectId(id) })
    except Exception as e:
        print(e)
    return redirect('/backstage')


@main.route('/recalculate_keywords/<uuid>', methods=['GET'])
def recalculate_keywords(uuid):
    log(request)
    calculate_keywords(collection().find_one({ "uuid" : uuid }))
    return redirect('/post?uuid=' + uuid)


@main.route('/welcome')
def welcome():
    log(request)
    check_for_quality_read()
    return render_template('welcome.html')


@main.route('/')
def index():
    if not 'was_here_before' in session:
        session['was_here_before'] = '1'
        return render_template('welcome.html')
    log(request)
    check_for_quality_read()
    query = request.args.get('query')
    if query and query != "":
        docs = similarity_search(query.strip(), [], MAX_DOCS)
        # for unknown reasons, these docs lack the 'text' field - refetching...
        docs = list(map(lambda doc: collection().find_one({ "uuid" : doc['uuid'] }), docs))
        infoline = '"' + query.strip() + '"'
    else: # the start page, called without query parameter
        if 'history' in session and len(session['history']) > 0:
            history_docs = list(map(lambda uuid:
                    collection().find_one({ "uuid" : uuid },
                                        { "uuid" : 1, "title" : 1, "_id" : 0 }),
                    session['history']))
            concatenated_titles = ""
            i = 0
            for doc in history_docs[-5:]: # only consider the recent history
                concatenated_titles += ("" if i == 0 else " ") + doc['title']
                i += 1
            docs = similarity_search(concatenated_titles, session['history'], MAX_DOCS)
            # for unknown reasons, these docs lack the 'text' field - refetching...
            docs = list(map(lambda doc: collection().find_one({ "uuid" : doc['uuid'] }), docs))
            infoline = "Personalized content"
        else: # no personalization possible - shuffle some items to start with
            docs = collection().aggregate([
                { "$sample": { "size": MAX_DOCS } }
            ])
            infoline = "Random content - no history yet"
    # prepare for a nice view
    docs = list(map(lambda doc: doc | {
        'fdate' : datetime.fromisoformat(doc["published"]).strftime("%d %b %Y"),
        'ftext' : textwrap.shorten(doc['text'] if 'text' in doc else 'No content.', 450) }, docs))
    return render_template('index.html', docs=docs, infoline=infoline)


@main.route('/post')
def post():
    log(request)
    check_for_quality_read()
    style = request.args.get('style')
    query = request.args.get('query')
    uuid = request.args.get('uuid')
    if query and query != "":
        return index()
    if style and style == "summary":
        doc = collection().find_one({ "uuid" : session['uuid'] })
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
        if 'embedding' in doc:
            recommendations = calculate_recommendations(doc['embedding'], session['history'], MAX_RCOM)
        else:
            recommendations = []
        return render_template('post.html', doc=doc, fdate=fdate,
                               fdoc=fdoc, recommendations=recommendations)
    else:
        if uuid: # highest prio: use uuid page parameter, if provided
            doc = collection().find_one({ "uuid" : uuid })
        elif 'uuid' in session: # session-saved uuid as the second choice
            doc = collection().find_one({ "uuid" : session['uuid'] })
        else: # TODO: use a personalized doc, if history is not empty
            doc = list(collection().aggregate([
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
        if not doc['uuid'] in session['history']: # don't allow dups
            session['history'].append(doc['uuid'])
        max_hist_len = 10
        if len(session['history']) > max_hist_len: # limit the max length of history
            session['history'] = session['history'][-max_hist_len:]
        if not 'keywords' in doc or len(doc['keywords']) == 0:
            keywords = calculate_keywords(doc)
        else:
            keywords = doc['keywords']
        if 'embedding' in doc:
            recommendations = calculate_recommendations(doc['embedding'], session['history'], MAX_RCOM)
        else:
            recommendations = []
        collection().update_one({ "_id" : doc["_id"] },
                              { "$inc" : { "visit_count" : 1 }})
        return render_template('post.html', doc=doc, fdate=fdate, fdoc=fdoc,
                               recommendations=recommendations, keywords=keywords,
                               visit_count=doc['visit_count']+1 if 'visit_count' in doc else 1,
                               read_count=doc['read_count'] if 'read_count' in doc else 0)


def get_country_name(iso_code):
    country = pycountry.countries.get(alpha_2=iso_code)
    return country.name if country else "Unknown"


@main.route('/backstage')
def about():
    loc = log(request)
    check_for_quality_read()
    if not 'history' in session:
        session['history'] = []
    docs = list(map(lambda uuid:
                    collection().find_one({ "uuid" : uuid },
                                        { "uuid" : 1, "title" : 1, "_id" : 0 }),
                    reversed(session['history'])))
    try:
        gen_ai_cache = list(gen_ai_cache_collection.find().sort({ "$natural" : -1 }).limit(100))
    except Exception as e:
        gen_ai_cache = []
        print(e) # will be printed in the log file that is residing in /tmp
    try:
        pipeline_countries = [
            {
                "$match": {
                    "country": { "$exists": True },
                    "city": { "$exists": True }
                }
            },
            {
                "$group": {
                    "_id": {
                        "country": "$country",
                        "city": "$city"
                    },
                    "city_access_count": { "$sum": 1 }
                }
            },
            {
                "$sort": { "city_access_count": -1 }
            },
            {
                "$group": {
                    "_id": "$_id.country",
                    "access_count": { "$sum": "$city_access_count" },
                    "top_cities": {
                        "$push": {
                            "city": "$_id.city",
                            "access_count": "$city_access_count"
                        }
                    }
                }
            },
            {
                "$sort": { "access_count": -1 }
            },
            {
                "$limit": 12
            },
            {
                "$project": {
                    "_id": 1,
                    "access_count": 1,
                    "top_cities": {
                        "$slice": ["$top_cities", 5]
                    }
                }
            }
        ]
        pipeline_paths = [
            {
                "$group": {
                    "_id": "$path",
                    "access_count": { "$sum": 1 }
                }
            },
            {
                "$sort": { "access_count": -1 }
            },
            {
                "$limit": 10
            },
            {
                "$project": {
                    "_id": 1,
                    "access_count": 1
                }
            }
        ]
        country_stats = list(access_log_collection.aggregate(pipeline_countries))
        path_stats = list(access_log_collection.aggregate(pipeline_paths))
        for entry in country_stats: # add full country names
            entry['country'] = get_country_name(entry['_id'])
    except Exception as e:
        country_stats = []
        path_stats = []
        print(e) # will be printed in the log file that is residing in /tmp
    return render_template('about.html', history=docs, gen_ai_cache=gen_ai_cache,
                           news_source=session['news_source'] if 'news_source' in session else 'business_news',
                           loc=loc, country_stats=country_stats, path_stats=path_stats)


@main.route('/insights')
def insights():
    log(request)
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
        return render_template('insights.html',
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
        return render_template('insights.html',
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
        return render_template('insights.html',
                               title=title, content=content, gen_ai_cache=gen_ai_cache)
    else:
        most_read_articles = collection().find().sort({ 'read_count' : -1 }).limit(10)
        return render_template('insights.html',
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


@main.route('/new')
def new_article():
    if 'title' in session:
        title = session['title']
    else:
        title = ''
    if 'text' in session:
        text = session['text']
    else:
        text = ''
    return render_template('new.html', title=title, text=text)


@main.route('/submit_post', methods=['POST'])
def submit_post():
    session['title'] = request.form.get('title')
    session['text'] = request.form.get('text')
    if 'image' not in request.files:
        return redirect(url_for('.new_article', error="missing_image"))
    image_file = request.files['image']
    if image_file.filename == '':
        return redirect(url_for('.new_article', error="missing_image"))
    uuid = str(python_uuid.uuid4().hex)
    published = datetime.utcnow().isoformat()
    author = 'Benjamin Lorenz'
    image_file.save(os.path.join('/home/bjjl/content/images', uuid + '.jpg'))
    collection().insert_one({ 'uuid' : uuid, 'published' : published, 'author' : author,
                            'title' : session['title'], 'text' : session['text'] })
    session.pop('title')
    session.pop('text')
    return redirect('/post?uuid=' + uuid)


@main.route('/contact')
def contact():
    log(request)
    check_for_quality_read()
    return render_template('contact.html')
