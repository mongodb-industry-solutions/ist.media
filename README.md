# ist.media - News Demo
This demo showcases an AI-powered news website.

## Highlights

- Joint play of Python/Flask, MongoDB Atlas, Vector Search, Langchain, and OpenAI.
- Allows to experiment with Personalization, Content Suggestions, Summarization, Keyword Extraction, and RAG-based News Insights. See [ist.media/welcome](https://ist.media/welcome) for more information about the functionality of the demo.

## Installation on a Mac

First and foremost - the application is installed and ready to run [in the cloud](https://ist.media). You should only aim for installing yourself if there's a strong reason for it.

I am hosting the webapp on a NetBSD/amd64 machine with 4 GB of RAM, 2 CPUs, and a 40 GB NVMe disk, running in Frankfurt at [vultr.com](https://vultr.com), and connecting to MongoDB Atlas, running an M10 at GCP/Frankfurt.

Ok, if you still want to get the demo running locally, here's the instructions.

#### Set environment variables to access your OpenAI token and your MongoDB Atlas cluster

```
OPENAI_API_KEY="<your token>"
MONGODB_IST_MEDIA="<your connection string>"
```

#### Install Python 3.11

```
brew install python@3.11
```

#### Set the path in your .zshrc

```
export PATH="$(brew --prefix)/opt/python@3.11/libexec/bin:$PATH"
```

#### Create and activate a Python virtual environment

```
python3 -m venv <dir>
cd <dir>; source ./bin/activate
```

#### Install Python packages

```
pip install -r requirements.txt
```

#### Populate your MongoDB database with news articles

They have to follow this schema:

![news datamodel](https://github.com/mongodb-industry-solutions/ist.media/blob/main/etc/datamodel.png?raw=true)

The ```embedding``` field is calculated by calling ```ist.media/scripts/vectorize.py```, the ```keywords``` field is calculated with AI on-the-fly, so can be left alone. Finally, ```visit_count``` and ```read_count``` will be calculated in a later version of the demo from within application code, so these also do not need to be existing in your data model.

A vector index should then be calculated from within the Atlas web interface, using ```cosine``` similarity and ```1536``` dimensions.

#### Start the application:

```
cd ist.media
./bin/uwsgi-debug.start
```

You will probably need to adapt some paths in the start script. If all goes well, you can access the app from your browser at localhost:9090.


## Acknowledgements

I want to thank Steve Dalby and Boris Bialek for giving me the opportunity to build this demo - it is so much fun!

[Benjamin Lorenz](https://www.linkedin.com/in/benjaminlorenz/)
