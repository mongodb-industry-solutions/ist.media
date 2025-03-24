#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import g, render_template, redirect, request, session, url_for, send_file
from flask import current_app as app
from mistune import html
from .. import mongo, logger
from . import main
from bson.objectid import ObjectId
from pymongo import MongoClient, ReturnDocument
from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlencode
from requests.auth import HTTPBasicAuth
from json import JSONEncoder
import re, textwrap, string, os, json, time, uuid as python_uuid, requests, geocoder, pycountry, math
import io, qrcode, bcrypt


IST_MEDIA_AUTH = [ os.environ.get('IST_MEDIA_AUTH_USERNAME', ''),
                   os.environ.get('IST_MEDIA_AUTH_PASSWORD', '') ]

MONGO_URI = os.environ['MONGODB_IST_MEDIA']

DB_NAME = "1_media_demo"
DEFAULT_NEWS_COLLECTION = 'news'

client = MongoClient(MONGO_URI)

gen_ai_cache_collection = client[DB_NAME]["gen_ai_cache"]
ip_info_cache_collection = client[DB_NAME]["ip_info_cache"]
access_log_collection = client[DB_NAME]["access_log"]
daily_collection = client[DB_NAME]["daily"]
users_collection = client[DB_NAME]["users"]
solana_collection_tx = client[DB_NAME]["solana_tx"]
solana_collection_tmp = client[DB_NAME]["solana_tmp"]

ai = OpenAI()

MAX_DOCS_VS = 30  # number of results for vector search
MAX_DOCS = 16     # number of articles on the home page
MAX_RCOM = 3      # number of recommended articles
MAX_RAG = 7       # number of articles for RAG context - 128k token limit


def debug(msg: str):
    if app.config['DEBUG']:
        print("[DEBUG]: " + msg)


def read_collection_from_session():
    if not 'news_source' in session:
        session['news_source'] = DEFAULT_NEWS_COLLECTION
    return session['news_source']


def collection():
    return client[DB_NAME][read_collection_from_session()]


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
        DB_NAME + "." + read_collection_from_session(),
        OpenAIEmbeddings(disallowed_special=()),
        index_name="vector_index")


def similarity_search(text: str,
                      history: list[str], count=MAX_DOCS_VS) -> list[dict]:
    results = vector_search().similarity_search(query=text, k=MAX_DOCS_VS)
    docs = map(lambda r: r.metadata, results) # docs lack the 'text' field - WHY?
    # don't recommend history items, and limit to count
    return list(filter(lambda r: not r['uuid'] in history, docs))[:count]


def hybrid_search(query: str, count=MAX_DOCS) -> list[Document]:
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore = vector_search(),
        search_index_name = "fulltext_index",
        top_k = count,
        vector_penalty = 50.0,
        fulltext_penalty = 70.0
    )
    return retriever.invoke(query)


def calculate_recommendations(embedding: list[float],
                              history: list[str], count=MAX_DOCS_VS) -> list[tuple]:
    results = vector_search()._similarity_search_with_score(query_vector=embedding, k=MAX_DOCS_VS)
    tuples = map(lambda r: (r[0].metadata, r[1]), results)
    # don't recommend history items, and limit to count
    return list(filter(lambda t: not t[0]['uuid'] in history, tuples))[:count]


def calculate_keywords(doc: dict) -> list[str]:
    service_url = app.config['API_BASE_URL'] + '/keywords'
    response = requests.post(service_url,
                             auth = HTTPBasicAuth(IST_MEDIA_AUTH[0], IST_MEDIA_AUTH[1]),
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
    """
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
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
        retriever = vector_search().as_retriever(search_kwargs = { "k" : MAX_RAG })
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
        if delta_seconds > 8 and delta_seconds < 500:
            try:
                collection().update_one({ "uuid" : session['uuid'] },
                                        { "$inc" : { "read_count" : 1 }})
                if 'user' in session:
                    users_collection.update_one( { "username" : session['user'] },
                                                 { "$inc" : { "articles_read" : 1 }})
            except Exception as e:
                print(e)


def get_user():
    return users_collection.find_one({ 'username' : session['user'] }) if 'user' in session else {}


@main.before_request
def make_session_permanent():
    session.permanent = True
    g.user = get_user()


@main.context_processor
def inject_user():
    return { 'user' : g.user }


class MongoJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return obj

    def encode(self, obj):
        def process_item(item):
            if isinstance(item, ObjectId):
                return str(item)
            elif isinstance(item, dict):
                return {k: process_item(v) for k, v in item.items()}
            elif isinstance(item, list) and len(item) > 10:
                total_length = len(item)
                return item[:5] + [ "... (shortened, total: {})".format(total_length) ] + item[-5:]
            elif isinstance(item, list):
                return [process_item(x) for x in item]
            elif isinstance(item, str) and len(item) > 100:
                return item[:100] + " [...]"
            return item

        processed_obj = process_item(obj)
        return super().encode(processed_obj)


@main.route('/json/<_id>')
def show_json(_id):
    doc = collection().find_one({ "_id" : ObjectId(_id) })
    json_data = json.dumps(doc, indent=2, ensure_ascii=False, cls=MongoJSONEncoder)
    return render_template('json.html', json_data=json_data)


@main.route('/select_news_source', methods=['POST'])
def select_news_source():
    selected_option = request.form['news_option']
    session['news_source'] = 'business_news' if selected_option == 'traditional' else DEFAULT_NEWS_COLLECTION
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


@main.route('/howto')
def howto():
    log(request)
    check_for_quality_read()
    return render_template('howto.html')


@main.route('/welcome')
def welcome():
    log(request)
    check_for_quality_read()
    return render_template('welcome.html')


@main.route('/profile')
def profile():
    log(request)
    check_for_quality_read()
    if "user" in session:
        return render_template('profile.html')
    else:
        return redirect('/login')


@main.route('/register')
def register():
    log(request)
    check_for_quality_read()
    if "user" in session:
        return redirect('/profile')
    else:
        return render_template('register.html')


@main.route('/do_register', methods=['POST'])
def do_register():
    username = request.form.get('username')
    password = request.form.get('password').encode('utf-8')

    fullname = request.form.get('fullname')
    email = request.form.get('email')

    hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
    users_collection.insert_one( { 'username' : username, 'password' : hashed_password,
                                   'fullname' : fullname, 'email' : email } )
    session['user'] = username
    return redirect('/profile')


@main.route('/login')
def login():
    log(request)
    check_for_quality_read()
    if "user" in session:
        return redirect('/profile')
    else:
        if 'badlogin' in session:
            session.pop('badlogin')
            return render_template('login.html', badlogin=True)
        else:
            return render_template('login.html')


@main.route('/do_login', methods=['POST'])
def do_login():
    username = request.form.get('username')
    password = request.form.get('password').encode('utf-8')

    user = users_collection.find_one( { 'username' : username } )

    if user and bcrypt.checkpw(password, user['password']):
        session['user'] = username
        return redirect('/')
    else:
        session['badlogin'] = True
        return redirect('/login')


def get_sol_price():
    url = "https://min-api.cryptocompare.com/data/price"
    params = {
        "fsym": "SOL",
        "tsyms": "USD"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data["USD"]


def calculate_sol_amount(usd_amount):
    sol_price = get_sol_price()
    sol_amount = usd_amount / sol_price
    return sol_price, sol_amount


def generate_solana_pay_uri(recipient, amount, memo, label):
    params = {
        "amount": f"{amount:.9f}",
        "memo": memo,
        "label": label
        #"spl-token": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" # USDT
    }
    query_string = urlencode(params)
    return f"solana:{recipient}?{query_string}"


def create_qr_code(payment_uri):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=5,
        border=0,
    )
    qr.add_data(payment_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#00684A", back_color="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr


@main.route('/payment')
def payment():
    log(request)
    check_for_quality_read()
    if "user" in session:
        usd_amount = 5.0 # top up of 5 USD hardcoded for now
        sol_price, sol_amount = calculate_sol_amount(usd_amount)
        return render_template("payment.html", sol_price=sol_price, sol_amount=sol_amount,
                               usd_amount=usd_amount)
    else:
        return redirect('/login')


@main.route('/payment-stage-2')
def payment_stage_2():
    log(request)
    check_for_quality_read()
    if "user" in session:
        signature = request.args.get('tx')
        return render_template("payment-stage-2.html", signature=signature)
    else:
        return redirect('/login')


@main.route('/payment-final')
def payment_final():
    log(request)
    check_for_quality_read()
    if "user" in session:
        sol_amount = session["amount_paid"] if "amount_paid" in session else 0
        # 5 USD = 500 Coins - given for any SOL amount paid --
        # this would need to be further hardened for production use
        coins_additional = 500 if sol_amount > 0 else 0
        user = users_collection.find_one_and_update(
            { 'username' : session['user'] },
            { '$inc' : { 'coins_current' : coins_additional,
                         'coins_lifetime' : coins_additional }},
            return_document=ReturnDocument.AFTER)
        session.pop("amount_paid", None)  # avoid cheating by refreshing the html page
        return render_template("payment-final.html", sol_amount=sol_amount,
                               coins_additional=coins_additional, user=user)
    else:
        return redirect('/login')


def get_memo():
    return session['user'] if 'user' in session else "anonymous"


@main.route('/qr_image/<float:amount>')
def qr_image(amount):
    SOLANA_RECIPIENT_ADDRESS = "918Y2TZvy386gXLWxGM9sBVutviT77xJriCDQsZeheEF"
    memo = get_memo()
    label = "IST.Media"
    payment_uri = generate_solana_pay_uri(SOLANA_RECIPIENT_ADDRESS, amount, memo, label)
    qr_image = create_qr_code(payment_uri)
    return send_file(qr_image, mimetype='image/png')


@main.route('/status')
def payment_status():
    tx_tmp = solana_collection_tmp.find_one({ "memo" : get_memo() })
    if tx_tmp:
        session['tx_in_progress'] = tx_tmp["signature"]
        return tx_tmp["signature"]
    else:
        return "waiting"


@main.route('/status-stage-2')
def payment_status_stage_2():
    if 'tx_in_progress' in session:
        tx = solana_collection_tx.find_one({ "signature" : session['tx_in_progress'],
                                             "memo" : get_memo() })
        if tx:
            users_collection.update_one({ 'username' : get_memo() }, { '$push' : { 'txs' : tx } })
            session.pop('tx_in_progress')
            session["amount_paid"] = tx["amount"]
            return "tx_confirmed"
    # no confirmation yet, or no transaction in progress
    return "waiting"


def adjusted_score(original_score, age_in_seconds, half_life=86400*90):
    lambda_ = math.log(2) / half_life
    time_decay = math.exp(-lambda_ * age_in_seconds)
    return original_score * time_decay, time_decay


@main.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


@main.route('/')
def index():
    if not 'was_here_before' in session:
        session['was_here_before'] = '1'
        return render_template('welcome.html')
    log(request)
    check_for_quality_read()
    query = request.args.get('query')
    if query and query != "":
        docs = hybrid_search(query.strip(), MAX_DOCS)
        docs = list(map(lambda doc: doc.dict()['metadata'] | { "text" : doc.page_content }, docs))
        #docs = list(filter(lambda doc: doc['vector_score'] > 0 and doc['fulltext_score'] > 0, docs))
        for doc in docs:
            timestamp_str = doc['published']
            timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
            timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)
            current_time = datetime.now(timezone.utc)
            time_difference = (timestamp_dt - current_time).total_seconds()
            seconds_away = abs(time_difference)
            doc['score'], doc['time_decay'] = adjusted_score(doc['score'], seconds_away)
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
        else:
            docs = collection().find({}).sort({ "published" : -1 }).limit(MAX_DOCS)
            infoline = "Sorted by time - no history yet"
    # prepare for a nice view
    docs = list(map(lambda doc: doc | {
        'ftext' : textwrap.shorten(doc['text'] if 'text' in doc else 'No content.', 450) }, docs))
    return render_template('index.html', docs=docs, infoline=infoline)


@main.route('/delete')
def delete():
    uuid = request.args.get('uuid')
    collection().delete_one({ "uuid" : uuid })
    session['history'] = []
    return redirect('/')


@main.route('/post')
def post():
    log(request)
    check_for_quality_read()
    style = request.args.get('style')
    query = request.args.get('query')
    uuid = request.args.get('uuid')
    lang = request.args.get('lang')
    if query and query != "":
        return index()
    if style and style == "summary":
        if not 'uuid' in session:
            return redirect('/')
        doc = collection().find_one({ "uuid" : session['uuid'] })
        if not doc:
            return render_template('404.html')
        fdoc = doc['text']
        lcdocs = [
            Document(page_content=doc['text'], metadata={"source": "local"})
        ]
        # Define prompt
        prompt_template = """Summarize in around 250 words the key facts of the following:
        "{text}"
        CONCISE SUMMARY:"""
        try:
            prompt = PromptTemplate.from_template(prompt_template)
            # Define LLM chain
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
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
        return render_template('post.html', doc=doc, fdoc=fdoc, recommendations=recommendations)
    if style and style == "translated":
        if not 'uuid' in session:
            return redirect('/')
        target_lang = lang if lang else "German"
        doc = collection().find_one({ "uuid" : session['uuid'] })
        if not doc:
            return render_template('404.html')
        fdoc = doc['text']
        lcdocs = [
            Document(page_content=doc['text'], metadata={"source": "local"})
        ]
        # Define prompt
        prompt_template = f"""Translate the following English news article into {target_lang}.

        Ensure the translation reflects the tone, structure, and sophistication typical of a
        conservative newspaper, such as "Frankfurter Allgemeine Zeitung" or "Die Welt". Use formal
        language, precise vocabulary, and a balanced perspective while maintaining journalistic
        integrity. Ensure cultural nuances are appropriately adapted to {target_lang}.

        Avoid deeply nested sentences. Rather break them up in several independent sentences. Do not
        use appositions such as "Orange, ein Telco-Unternehmen, hat angekündigt."  Instead, rephrase
        such sentences to structures like "Das Telco-Unternehmen Orange hat angekündigt."

        Avoid the overuse of demonstrative pronouns like "dieser/diese/dieses" unless they are
        necessary for clarity or emphasis. In most cases, prefer simpler structures with definite
        articles (e.g., "die Initiative" instead of "diese Initiative") when the context already
        makes the reference clear. This ensures a more natural and fluent translation.

        Paraphrase and summarize in your own words. Don't translate sentence by
        sentence. Make the text 40% shorter.

        "{{text}}"
        TRANSLATION:"""
        try:
            prompt = PromptTemplate.from_template(prompt_template)
            # Define LLM chain
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            # Define StuffDocumentsChain
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="text")
            fdoc = target_lang.upper() + ": " + stuff_chain.invoke(lcdocs)['output_text']
            fdoc = html(fdoc.replace("\n", "\n\n"))
        except Exception as e:
            fdoc = "OpenAI crashed - you should try again (later)"
            print(e) # will be printed in the log file that is residing in /tmp
        if 'embedding' in doc:
            recommendations = calculate_recommendations(doc['embedding'], session['history'], MAX_RCOM)
        else:
            recommendations = []
        return render_template('post.html', doc=doc, fdoc=fdoc, recommendations=recommendations)
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
        collection().update_one({ "_id" : doc["_id"] }, { "$inc" : { "visit_count" : 1 }})
        if 'user' in session:
            result = users_collection.update_one({ 'username' : session['user'],
                                                   'ids': { '$not' : { '$eq' : doc['uuid'] }}},
                                                 { '$inc' : { 'coins_current' : -1 },
                                                   '$push' : { 'ids' : doc['uuid'] }})
            purchased = 'now' if result.modified_count > 0 else 'earlier'
        else:
            purchased = 'not_applicable'
        return render_template('post.html', doc=doc, fdoc=fdoc,
                               recommendations=recommendations, keywords=keywords, purchased=purchased,
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
                           news_source=session['news_source'] if 'news_source' in session else DEFAULT_NEWS_COLLECTION,
                           loc=loc, country_stats=country_stats, path_stats=path_stats)


def get_mongodb_date_filter(natural_language_date):
    today = datetime.utcnow()

    prompt = f"""
    Convert the following time expression into a MongoDB-compatible filter format.

    Example:
    - "last week" → {{ "$gte": "<YYYY-MM-DD>", "$lt": "<YYYY-MM-DD>" }}
    - "in January 2024" → {{ "$gte": "2024-01-01", "$lt": "2024-02-01" }}
    - "yesterday" → {{ "$gte": "<YYYY-MM-DD>", "$lt": "<YYYY-MM-DD>" }}

    Time expression: "{natural_language_date}"

    Never create date expressions that point to the future. Never do!

    If no conversion is possible, return the universal time filter that goes
    from January 2024 to "{today}".

    Return **only** a valid JSON object, without explanations or comments.
    No wrap of ```json.
    Today is "{today}"
    """

    response = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_text = response.choices[0].message.content.strip()

    # Ensure the response is valid JSON
    try:
        json_text = raw_text.strip("`")  # Remove possible markdown code block formatting
        date_filter = json.loads(json_text)
        return date_filter
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {raw_text}")
        return None  # Return None if parsing fails


def get_news_for_today():
    pipeline = [
        { "$match" : { "published" : get_mongodb_date_filter("today") } },
        { "$project" : { "title" : 1, "published" : 1, "text" : 1 } }
    ]
    results = list(collection().aggregate(pipeline))
    return results


@main.route('/feed')
def feed():
    docs = get_news_for_today()
    return render_template('feed.html', docs=docs)


@main.route('/daily')
def daily():
    log(request)
    check_for_quality_read()

    now = datetime.utcnow()
    if now.hour < 14: # aligned with cronjob config "21 14,23 * * 1-6" CE(S)T timezone
        now -= timedelta(days=1) # not enough news yet - fallback to yesterday

    # for Saturday (5) or Sunday (6), go back to the previous Friday
    if now.weekday() >= 5:
        now -= timedelta(days=(now.weekday() - 4))

    formatted_date = now.strftime("%d %B %Y")
    try:
        doc = daily_collection.find_one({ "day" : formatted_date })
        summary = html(doc['summary']) if 'summary' in doc else None
        entities = doc['entities'] if 'entities' in doc else []
        podcast = f"/content/audio/podcast-{now:%d.%m.%Y}.mp3"

        full_path = os.path.join("/usr/local/share", podcast.lstrip('/'))
        if not os.path.exists(full_path):
            podcast = None # no podcast produced for the day

    except Exception as e:
        summary = None
        entities = []
        podcast = None
    return render_template('daily.html', day=formatted_date, summary=summary,
                           entities=entities, podcast=podcast)


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
                               content="""

                               <p>Please enter your question in the form above.
                               Answers are provided based only on the documents
                               stored in MongoDB, using Vector Search and GPT-4o.</p>

                               <p>This page also serves to display AI-generated insights
                               when clicking on a keyword in the Single Post page. In that
                               case, general knowledge of the LLM is used, without RAG.</p>

                               <p>The generation of answers usually takes some time
                               (5-20 seconds). You have to be patient. Please don't
                               refresh the page or re-enter the question.</p>

                               <p>Most recent insights are cached and can be accessed from
                               the right column of this page, e.g. when conducting a demo.</p>

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
