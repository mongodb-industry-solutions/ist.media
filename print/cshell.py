from pymongo import MongoClient
from voyageai import Client
from os import environ
import readline

client = MongoClient(environ.get('MONGODB_IST_MEDIA', ''))
dbName = "1_media_demo"
collectionName = "charts"
collection = client[dbName][collectionName]

voyage_client = Client()

print("Enter text queries. Type 'exit' to quit.")

while True:
    try:
        query = input("> ").strip()

        if query.lower() == 'exit':
            break

        if not query:
            continue

        try:
            response = voyage_client.multimodal_embed(
                inputs=[[query]],
                model="voyage-multimodal-3",
                input_type="query"
            )
            query_vector = response.embeddings[0]
        except Exception as e:
            print(f"Error embedding query: {e}")
            continue

        pipeline = [
            {
                '$vectorSearch': {
                    'index': 'charts_vector_index',
                    'path': 'embedding',
                    'queryVector': query_vector,
                    'numCandidates': 10,
                    'limit': 1
                }
            },
            {
                '$project': {
                    'description': 1,
                    'score': {'$meta': 'vectorSearchScore'},
                    '_id': 0
                }
            }
        ]

        try:
            results = list(collection.aggregate(pipeline))
            if results:
                print("Top match:")
                for result in results:
                    print(
                        f" - {result['description']}"
                        f" (score: {result['score']:.4f})"
                    )
            else:
                print("No matches found.")
        except Exception as e:
            print(
                f"Vector search error: {e}."
                f" Ensure Atlas Vector Search is enabled and index is ready."
            )

    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break

client.close()
