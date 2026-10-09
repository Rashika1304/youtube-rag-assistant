from functools import lru_cache
from urllib.parse import urlparse, parse_qs

import torch
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_video_id(url):
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":")[0]

    if host.endswith("youtube.com") and parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [None])[0]
    elif host == "youtu.be":
        return parsed.path.strip("/").split("/")[0] or None
    elif host.endswith("youtube.com") and parsed.path.startswith(("/embed/", "/shorts/")):
        parts = parsed.path.strip("/").split("/")
        return parts[1] if len(parts) > 1 else None

    return None


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@lru_cache(maxsize=1)
def get_llm_components():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


def process_video(url):
    video_id = get_video_id(url)

    if not video_id:
        raise ValueError("Please enter a valid YouTube URL.")

    transcript_items = YouTubeTranscriptApi().fetch(
        video_id, languages=["en"]
    )
    transcript = " ".join(item.text for item in transcript_items)

    if not transcript.strip():
        raise ValueError("The transcript is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "video_id": video_id,
            "chunk_index": i
        })

    vector_store = FAISS.from_documents(chunks, get_embeddings())

    return {
        "video_id": video_id,
        "transcript": transcript,
        "vector_store": vector_store
    }


def answer_question(vector_store, question):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Answer using only the transcript context below.
If the answer is not present, say you don't know.

Transcript context:
{context}

Question: {question}

Answer:"""

    tokenizer, model = get_llm_components()
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return answer, docs
