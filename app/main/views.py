#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import (
    g, render_template, redirect, request, session,
    url_for, send_file, jsonify, current_app as app )
from mistune import html
from .. import mongo, mongo_preview, logger
from ..agentic.main import _user_prompt_prefix
from . import main
from .metrics import (
    compute_user_engagement,
    request_stats_pipelines,
    user_consumption_pipeline )
from typing import List, Dict, Any
from bson import Binary
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
import re, textwrap, string, os, json, time, uuid as python_uuid, numpy as np
import requests, geocoder, pycountry, math, io, qrcode, bcrypt, voyageai, threading


IST_MEDIA_AUTH = [ os.environ.get('IST_MEDIA_AUTH_USERNAME', ''),
                   os.environ.get('IST_MEDIA_AUTH_PASSWORD', '') ]

DEFAULT_NEWS_COLLECTION = 'news'
ALLOWED_NEWS_COLLECTIONS = { 'news', 'business_news' }

gen_ai_cache_collection = mongo.db.gen_ai_cache
ip_info_cache_collection = mongo.db.ip_info_cache
access_log_collection = mongo.db.access_log
daily_collection = mongo.db.daily
users_collection = mongo.db.users
solana_collection_tx = mongo.db.solana_tx
solana_collection_tmp = mongo.db.solana_tmp
engagement_events_collection = mongo.db.engagement_events

ai = OpenAI()
voyage_ai = voyageai.Client()

processing_users = set()
processing_lock = threading.Lock()

MAX_DOCS_VS = 30  # number of results for vector search
MAX_DOCS = 16     # number of articles on the home page
MAX_RCOM = 3      # number of recommended articles
MAX_RAG = 7       # number of articles for RAG context - 128k token limit


############################################
###          BEGIN video search          ###
############################################

parent_folder = "/usr/local/share/content/video/frames"
frames = []

print("Building movie search array...", end="", flush=True)
for root, _, files in os.walk(parent_folder):
    for fname in sorted(files):
        if fname.endswith(".json") and fname.startswith("frame_"):
            path = os.path.join(root, fname)
            with open(path, "r") as f:
                data = json.load(f)
                frames.append({
                    "movie": data["movie"],
                    "offset": data["offset"],
                    "embedding": data["embedding"]
                })
print(" done")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

############################################
###           END video search           ###
############################################


def debug(msg: str):
    if app.config['DEBUG']:
        print("[DEBUG]: " + msg)


# the database that is used for preview (from content lab)
def preview_collection():
    return mongo_preview.db["preview"]


def read_collection_from_session():
    name = session.get("news_source", DEFAULT_NEWS_COLLECTION)
    if name not in ALLOWED_NEWS_COLLECTIONS:
        name = DEFAULT_NEWS_COLLECTION
        session["news_source"] = name
    return name


def collection():
    return mongo.db[read_collection_from_session()]


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
    #logger.info(ip + " (" + loc['city'] + ", " + loc['country'] + "): " + request.base_url +
    #        (" (" + str(request.content_length) + " bytes)" if request.content_length else ""))
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
    """Build a VectorStore bound to the currently selected news collection."""
    return MongoDBAtlasVectorSearch(
        collection = mongo.db[read_collection_from_session()],
        embedding = OpenAIEmbeddings(disallowed_special=()),
        index_name = "vector_index"
    )


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

    template = """Answer the question using both of the following knowledge bases.
    Make it one continguous, comprehensive answer with three paragraphs.
    Don't use bullet points and enumerations.

    **Internal Knowledge:**
    {local_context}

    **Web Research:**
    {web_context}

    Question: {question}"""

    try:
        local_retriever = vector_search().as_retriever(search_kwargs = { "k" : MAX_RAG })
        local_docs = local_retriever.invoke(question)
        local_context = "\n".join(doc.page_content for doc in local_docs)

        client = OpenAI()
        web_response = client.responses.create(
            model = "gpt-4o",
            tools = [ { "type" : "web_search_preview" } ],
            input = question
        )
        web_context = web_response.output_text

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model_name="gpt-4o")
        chain = (
            {
                "local_context" : lambda x: local_context,
                "web_context" : lambda x: web_context,
                "question" : RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        gen_ai_cache_collection.insert_one( { "question" : question, "answer" : answer } )

    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        answer = ""

    return answer


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


def get_user_dict(request):
    user_dict = {
        "username" : "",
        "fullname" : "",
        "is_anonymous" : True
    }
    if "user" in session:
        db_user = users_collection.find_one({ "username" : session["user"] })
        if db_user:
            user_dict.update(db_user)
            user_dict["is_anonymous"] = False
    elif anon_id := request.cookies.get("anon_id"):
        db_user = users_collection.find_one_and_update(
            { "username" : anon_id },
            { "$setOnInsert" : { "fullname" : "Anonymous User" } },
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        if db_user:
            user_dict.update(db_user)
    return user_dict


def update_user_engagement(username):
    try:
        engagement = compute_user_engagement(username)

        _p = user_consumption_pipeline(username)
        stats = list(mongo.db.engagement_events.aggregate(_p))[0]

        mongo.db.users.update_one(
            { 'username' : username },
            { "$set" : { "engagement" : engagement, "stats" : stats }}
        )

    except Exception as e:
        logger.error(f"{username}: background error - {e}")

    finally:
        with processing_lock:
            processing_users.discard(username)

        logger.warning(f"{username}: background thread ended: {datetime.utcnow()}")


@main.before_request
def before_request():
    session.permanent = True
    g.user = get_user_dict(request)
    username = g.user['username']

    if username:
        result = mongo.db.users.find_one_and_update(
            { 'username' : username },
            { '$set' : { 'last_active' : datetime.utcnow() } },
            return_document=ReturnDocument.AFTER
        )
        g.engagement = result.get('engagement') if result else None
        g.stats = result.get('stats') if result else None

        if request.path != '/':
            return

        with processing_lock:
            if username not in processing_users:
                processing_users.add(username)
                t = threading.Thread(target=update_user_engagement, args=(username,))
                t.daemon = True
                t.start()
                logger.warning(f"{username}: background thread started: {datetime.utcnow()}")
            else:
                logger.warning(f"{username}: background thread already running, skipping")


@main.context_processor
def inject_user_metrics():
    try:
        engagement = int( math.floor( g.engagement['smoothed_indexes']['ema28'] + 0.5 ))
    except Exception:
        engagement = 0
    return { 'user' : g.user, 'engagement' : engagement }


class MongoJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return obj

    def encode(self, obj):
        seen = set()
        def process_item(item):
            item_id = id(item)
            if item_id in seen:
                return "[circular reference]"
            if isinstance(item, (dict, list)):
                seen.add(item_id)
                try:
                    if isinstance(item, dict):
                        return { k : process_item(v) for k, v in item.items() }
                    elif isinstance(item, list) and len(item) > 10:
                        total_length = len(item)
                        val = (
                            item[:5]
                            + [ "... (shortened, total: {})".format(total_length) ]
                            + item[-5:]
                        )
                        return [ process_item(x) for x in val ]
                    elif isinstance(item, list):
                        return [ process_item(x) for x in item ]
                finally:
                    seen.remove(item_id)
            if isinstance(item, ObjectId):
                return str(item)
            elif isinstance(item, datetime):
                return item.isoformat()
            elif isinstance(item, bytes):
                return item.decode('utf-8', errors='replace')
            elif isinstance(item, str) and len(item) > 115:
                return item[:115] + " [...]"
            return item

        processed_obj = process_item(obj)
        return super().encode(processed_obj)


@main.route('/json/<_id>')
def show_json(_id):
    log(request)
    if _id == "0": # this is a hack - should refactor at some point
        doc = g.engagement
        title = "User Engagement Details"
    else:
        doc = collection().find_one({ "_id" : ObjectId(_id) })
        title = "MongoDB Article Document"
        if not doc:
            doc = users_collection.find_one({ "_id" : ObjectId(_id) })
            title = "MongoDB User Document"
    json_data = json.dumps(doc, indent=2, ensure_ascii=False, cls=MongoJSONEncoder)
    return render_template('json.html', json_data=json_data, title=title)


@main.route("/track", methods=["POST"])
def track_event():
    data = request.get_json(force=True) or {}
    event = {}

    if "user_id" not in data or not data["user_id"]:
        return jsonify({"error": "Missing user_id"}), 400
    event["user_id"] = data["user_id"]

    if "article_uuid" in data and data["article_uuid"]:
        try:
            article_uuid = python_uuid.UUID(data["article_uuid"])
            event["article_uuid"] = Binary.from_uuid(article_uuid)
        except (ValueError, AttributeError):
            pass

    allowed_fields = [
        "event",
        "scroll_depth",
        "offer_type",
        "amount"
    ]
    for f in allowed_fields:
        if f in data and data[f] not in (None, "", []):
            if f == "scroll_depth":
                try:
                    depth = max(0.0, min(float(data[f]), 1.0))
                    event[f] = round(depth, 2)
                except (ValueError, TypeError):
                    continue
            else:
                event[f] = data[f]

    if event.get("event") == "leave" and "scroll_depth" in event:
        event["read_to_end"] = event["scroll_depth"] >= 0.98

    event["ts"] = datetime.utcnow()

    engagement_events_collection.insert_one(event)
    return jsonify({"status": "ok"})


@main.route("/user_engagement/<user_id>")
def user_engagement(user_id):
    include_today = request.args.get("include_today", "true").lower() != "false"
    data = compute_user_engagement(user_id, windows=(3, 7, 28),
                                   fill_gaps=True,
                                   include_today=include_today)
    return jsonify({"user_id": user_id, **data})


@main.route('/select_news_source', methods=['POST'])
def select_news_source():
    log(request)
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
    if 'user' in session:
        users_collection.update_one({ 'username' : session['user'] }, { "$unset" : { "articles_visited" : 1 } })
    return redirect('/backstage')


@main.route('/delete_articles_history_from_homepage', methods=['GET'])
def delete_articles_history_from_homepage():
    log(request)
    session['history'] = []
    if 'user' in session:
        users_collection.update_one({ 'username' : session['user'] }, { "$unset" : { "articles_visited" : 1 } })
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


@main.route('/ai-context-visualizer')
def ai_context():
    return render_template('ai-context-visualizer.html',
                           context = _user_prompt_prefix(g.user['username']))


@main.route('/profile')
def profile():
    log(request)
    check_for_quality_read()
    try:
        summary = mongo.db.users.find_one({ 'username' : g.user['username'] })['summary']['text']
    except Exception:
        summary = None
    return render_template('profile.html', stats=g.stats, summary=summary)


@main.route('/register_from_signup_promo_page', methods=['POST'])
def register_from_signup_promo_page():
    mongo.db.planner.update_one(
        { "username" : g.user['username'] },
        { "$set" : {
            "history" : {
                "register_click_ts" : datetime.utcnow()
            }
        }},
        upsert=True
    )
    return redirect('/register')


@main.route('/register')
def register():
    log(request)
    check_for_quality_read()
    if "user" in session:  # already registered
        return redirect('/profile')
    else:
        # allow for passing an optional article id --
        # can be used to redirect after registration (register wall)
        return render_template('register.html',
                               article_id=request.args.get('article_id'),
                               title=request.args.get('title'))


@main.route('/do_register', methods=['POST'])
def do_register():
    log(request)
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
    if not g.user['is_anonymous']:
        return redirect('/profile')
    else:
        if 'badlogin' in session:
            session.pop('badlogin')
            return render_template('login.html', badlogin=True)
        else:
            return render_template('login.html')


@main.route('/do_login', methods=['POST'])
def do_login():
    log(request)
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
    if not g.user['is_anonymous']:
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
    log(request)
    SOLANA_RECIPIENT_ADDRESS = "918Y2TZvy386gXLWxGM9sBVutviT77xJriCDQsZeheEF"
    memo = get_memo()
    label = "IST.Media"
    payment_uri = generate_solana_pay_uri(SOLANA_RECIPIENT_ADDRESS, amount, memo, label)
    qr_image = create_qr_code(payment_uri)
    return send_file(qr_image, mimetype='image/png')


@main.route('/status')
def payment_status():
    #log(request)
    tx_tmp = solana_collection_tmp.find_one({ "memo" : get_memo() })
    if tx_tmp:
        session['tx_in_progress'] = tx_tmp["signature"]
        return tx_tmp["signature"]
    else:
        return "waiting"


@main.route('/status-stage-2')
def payment_status_stage_2():
    log(request)
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
    log(request)
    session.pop('user', None)
    return redirect('/')


def get_embeddings(articles_visited):
    uuids = [ item["uuid"] for item in articles_visited ]
    cursor = collection().find({ "uuid" : { "$in" : uuids }}, { "embedding" : 1 })
    return [ doc["embedding"] for doc in cursor if "embedding" in doc ]


def add_to_fifo_if_not_exists(username, array_field, doc, max_length=10): # TODO: define `max_length´ constant
    existing = users_collection.find_one({
        "username" : username,
        f"{array_field}.uuid" : doc['uuid']
    })
    if existing:
        return False
    users_collection.update_one(
        { "username" : username },
        {
            "$push" : {
                array_field: {
                    "$each" : [ { "uuid" : doc['uuid'], "title": doc['title'], "added_at" : datetime.utcnow() } ],
                    "$slice": -max_length
                }
            }
        }
    )
    return True


@main.route('/')
def index():
    if not 'was_here_before' in session:
        session['was_here_before'] = '1'
        return render_template('welcome.html')
    log(request)
    check_for_quality_read()
    query = request.args.get('query')
    section = request.args.get('section') or session.get('section') or "_all"
    session['section'] = section
    if query and query != "":
        cleaned_query = query.strip()
        mongo.db.users.update_one(
            { "username" : g.user['username'] },
            { "$push" : {
                "searches" : {
                    "query" : cleaned_query,
                    "submitted_at" : datetime.utcnow()
                }
            }}
        )
        docs = hybrid_search(cleaned_query, MAX_DOCS)
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
    elif section != "_all" and section != "_personalized":
        docs = collection().find({ "sections" : section }).sort({ "published" : -1 }).limit(MAX_DOCS)
        infoline = "Filtered by section"
    else:
        if section == "_all":
            docs = collection().find({}).sort({ "published" : -1 }).limit(MAX_DOCS)
            infoline = "Sorted by time"
        else:
            if not 'user' in session:
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
                    docs = list(map(lambda doc: collection().find_one({ "uuid" : doc['uuid'] }), docs))
                    infoline = "For anonymous user"
                else:
                    docs = collection().find({}).sort({ "published" : -1 }).limit(MAX_DOCS)
                    infoline = "Sorted by time"
                    section = "_all"
            else:
                user = users_collection.find_one({ "username" : session['user']})
                if 'articles_visited' in user:
                    read_vectors = get_embeddings(user["articles_visited"])
                    user_vector = np.mean(read_vectors, axis=0).tolist()
                    pipeline = [
                        {
                            "$vectorSearch" : {
                                "index" : "vector_index",
                                "queryVector" : user_vector,
                                "path" : "embedding",
                                "numCandidates" : 100,
                                "limit" : MAX_DOCS
                            }
                        },
                        {
                            "$match": {
                                "uuid" : {
                                    "$nin" : [ item["uuid"] for item in user["articles_visited"] ]
                                }
                            }
                        }
                    ]
                    docs = list(collection().aggregate(pipeline))
                    infoline = "For " + user['username']
                else:
                    docs = collection().find({}).sort({ "published" : -1 }).limit(MAX_DOCS)
                    infoline = "Sorted by time"
                    section = "_all"

    # prepare for a nice view:
    # 1) shorten the body content -- similar length each
    # 2) popular articles require registration of users
    docs = [
        doc | {
            "ftext": textwrap.shorten(doc.get("text", "No content."), 450),
            "must_register_to_read": int(doc.get("read_count", 0)) >= 3,
        }
        for doc in docs
    ]

    # the following dynamic calculation of all sections currently existing
    # should probably be moved to a place where this is only calculated once
    # per server start, and not with every index page call. Currently this is
    # not a big performance hit yet, but the more content is collected over
    # time, the worse this will get.
    pipeline = [
        { "$project" : { "sections" : 1 }},
        { "$unwind" : "$sections" },
        { "$group" : { "_id" : None , "all_sections" : { "$addToSet" : "$sections" }}}
    ]
    result = list(collection().aggregate(pipeline))

    if result:
        sections = sorted(result[0]['all_sections'])
    else:
        sections = []

    # check for active promotions - show them just in case
    # TODO: store time of last display in planner doc
    #         open up / allow to re-show after some days
    # cookie solution for now, but this is not flexible. One-show only.
    if (promo := mongo.db.planner.find_one({ "username" : g.user['username'], "type" : 1 })):
        if not 'promo_was_presented_before' in session:
            session['promo_was_presented_before'] = '1'
            return redirect('/signup-promo')

    return render_template('index.html', docs=docs, infoline=infoline,
                           sections=sections, selected_section=section)


@main.route('/delete')
def delete():
    log(request)
    uuid = request.args.get('uuid')
    collection().delete_one({ "uuid" : uuid })
    session['history'] = []
    return redirect('/')


@main.route('/howto_videosearch')
def howto_videosearch():
    log(request)
    return render_template('howto_videosearch.html')


@main.route('/video_search')
def video_search():
    log(request)
    query_text = request.args.get('query')
    if not query_text or query_text == "":
        return jsonify({
            "offset": 0,
            "infoline": "Enter your query above"
        })

    query_embedding = np.array(voyage_ai.multimodal_embed(
        [[query_text]], model="voyage-multimodal-3").embeddings[0])

    scores = []
    for frame in frames:
        score = cosine_similarity(query_embedding, frame["embedding"])
        scores.append(score)

    best_idx = np.argmax(scores)

    movie = None
    scene_start = None
    prev_score = scores[best_idx]
    for i in range(best_idx, -1, -1):
        score = scores[i]
        if prev_score - score > 0.01:
            movie = frames[i]["movie"]
            scene_start = frames[i]["offset"]
            break
        prev_score = score

    if scene_start is None:
        movie = frames[0]["movie"] # TODO: might point to the wrong "default" movie
        scene_start = frames[0]["offset"]

    return jsonify({
        "movie" : movie,
        "offset" : scene_start,
        "infoline" : query_text
    })


@main.route('/video')
def video():
    log(request)
    check_for_quality_read()
    return render_template('video.html', offset=0, infoline="Enter your search query above")


@main.route('/signup-promo')
def signup_promo():
    promo = mongo.db.planner.find_one({ "username" : g.user['username'], "type" : 1 })
    if promo:
        text = promo['promo']['text']
        ts = promo['promo']['ts'].strftime('%d.%m.%Y %H:%M')
    else:
        text = "There's no current acquisition promo for this user."
        ts = ""
    return render_template('signup-promo.html', text=text, ts=ts)


@main.route('/post')
def post():
    log(request)
    check_for_quality_read()
    style = request.args.get('style')
    query = request.args.get('query')
    uuid = request.args.get('uuid')
    lang = request.args.get('lang')
    preview = request.args.get('preview')
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
    if preview and preview != "":
        published = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        doc = preview_collection().find_one({ "_id" : ObjectId(preview)})
        fdoc = '<div class="ai-gen">' + html(doc['content'].replace("\n", "\n\n")) + '</div>'
        return render_template('post.html', doc=doc, fdoc=fdoc, preview=True, published=published,
                               recommendations=[], keywords=doc['keywords'])
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
            #result = users_collection.update_one({ 'username' : session['user'],
            #                                       'articles_visited': { '$not' : { '$eq' : doc['uuid'] }}},
            #                                     { '$inc' : { 'coins_current' : -1 },
            #                                       '$push' : { 'articles_visited' : doc['uuid'] }})
            #purchased = 'now' if result.modified_count > 0 else 'earlier'
            result_inserted = add_to_fifo_if_not_exists(session['user'], 'articles_visited', doc)
            if result_inserted:
                users_collection.update_one(
                    { 'username' : session['user'] },
                    { '$inc': { 'coins_current' : -1 }}
                )
                purchased = 'now'
            else:
                purchased = 'earlier'
        else:
            purchased = 'not_applicable'

        # popular articles are register-only
        doc["must_register_to_read"] = int(doc.get("read_count", 0)) >= 3 and not 'user' in session

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
        country_stats = list(access_log_collection.aggregate(request_stats_pipelines['countries']))
        path_stats = list(access_log_collection.aggregate(request_stats_pipelines['paths']))
        for entry in country_stats: # add full country names
            entry['country'] = get_country_name(entry['_id'])
    except Exception as e:
        country_stats = []
        path_stats = []
        print(e) # will be printed in the log file that is residing in /tmp

    try:
        service_url = app.config['MAIN_BASE_URL'] + '/stats/article_engagement'
        response = requests.get(
            service_url,
            auth=HTTPBasicAuth(IST_MEDIA_AUTH[0], IST_MEDIA_AUTH[1]),
            params={"days": 90, "limit": 5}
        )
        response.raise_for_status()
        article_engagement_stats_json = response.json()
    except Exception as e:
        print(f"Error fetching article engagement stats: {e}")
        article_engagement_stats_json = []

    return render_template(
        'about.html',
        history=docs,
        gen_ai_cache=gen_ai_cache,
        article_engagement_stats_json=article_engagement_stats_json,
        news_source=session['news_source'] if 'news_source' in session else DEFAULT_NEWS_COLLECTION,
        loc=loc,
        country_stats=country_stats,
        path_stats=path_stats)


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
    log(request)
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
    log(request)
    return render_template('new.html')


@main.route('/newurl')
def new_url():
    log(request)
    return render_template('newurl.html')
