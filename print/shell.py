from pymongo import MongoClient
from voyageai import Client
from os import environ
import readline

client = MongoClient(environ.get('MONGODB_IST_MEDIA', ''))
dbName = "1_media_demo"
collectionName = "print"
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
                    'index': 'print_vector_index',
                    'path': 'embedding',
                    'queryVector': query_vector,
                    'numCandidates': 10,
                    'limit': 3
                }
            },
            {
                '$project': {
                    'image_filename': 1,
                    'score': {'$meta': 'vectorSearchScore'},
                    'issue_date': 1,
                    'page': 1,
                    '_id': 0 #
                }
            }
        ]

        try:
            results = list(collection.aggregate(pipeline))
            if results:
                print("Top 3 matches:")
                for result in results:
                    print(
                        f" - {result['image_filename']}"
                        f" (score: {result['score']:.4f}"
                        f", date: {result.get('issue_date', 'N/A')}"
                        f", page: {result.get('page', 'N/A')})"
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
