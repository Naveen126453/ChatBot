# from flask import Flask, render_template, request

# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# app = Flask(__name__)

# # Lightweight model
# llm = Ollama(model="mistral")

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db = FAISS.load_local(
#     "faiss_index",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever()
# )

# @app.route("/", methods=["GET", "POST"])
# def index():
#     answer = ""
#     if request.method == "POST":
#         query = request.form["query"]
#         answer = qa.run(query)
#     return render_template("index.html", answer=answer)

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

app = Flask(__name__)

# LLM (can be mistral / phi / tinyllama)
llm = Ollama(model="mistral")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        query = request.form["query"]

        # 🔹 STEP 1: Check if relevant documents exist
        docs = db.similarity_search(query, k=2)

        if docs and len(docs[0].page_content.strip()) > 50:
            # 🔹 STEP 2A: Use PDF-based RAG
            answer = qa.run(query)
        else:
            # 🔹 STEP 2B: Fallback to generic LLM response
            answer = llm.invoke(query)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
