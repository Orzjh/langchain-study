import os, io, uuid, time, pickle, json, sys
from http import HTTPStatus
from typing import List, Dict, Any

import fitz
from PIL import Image
import numpy as np
import faiss
from sklearn.preprocessing import normalize
import dashscope
from dotenv import load_dotenv

import anyio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools, convert_mcp_to_langchain_tools

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

DATA_DIR           = "knowledge_base_multimodal"
IMAGE_SAVE_DIR     = os.path.join(DATA_DIR, "extracted_images")
VECTOR_STORE_PATH  = "faiss_index_qwen_api_rag"

TEXT_EMBED_MODEL   = "text-embedding-v1"
QWEN_VL_MODEL      = "qwen-vl-plus"
CHAT_MODEL         = "gpt-4o"

def rewrite_query(original: str) -> str:
    """Rewrite the raw query into a short search‑optimised form."""
    system_msg = {"role": "system", "content": "Rewrite the user's query to a concise, search‑optimised form."}
    user_msg   = {"role": "user",   "content": original}
    try:
        resp = dashscope.Generation.call(model=CHAT_MODEL, messages=[system_msg, user_msg], result_format="message")
        if resp.status_code == HTTPStatus.OK:
            return resp.output["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Rewrite error] {e}")
    return original  # fallback


def get_text_embeddings_api(text_list: List[str]) -> List[List[float] | None]:
    """Batch embed via DashScope text‑embedding API."""
    if not text_list:
        return []
    try:
        resp = dashscope.TextEmbedding.call(model=TEXT_EMBED_MODEL, input=text_list)
        if resp.status_code == HTTPStatus.OK:
            size = len(text_list)
            out: List[List[float] | None] = [None] * size
            for info in resp.output["embeddings"]:
                out[info["text_index"]] = info["embedding"]
            return out
    except Exception as e:
        print(f"[Embedding error] {e}")
    return [None] * len(text_list)


def generate_caption_api(image_path: str) -> str:
    """Describe an image using Qwen‑VL."""
    uri = f"file://{os.path.abspath(image_path)}"
    msg = [{"role": "user", "content": [{"image": uri}, {"text": "Describe this image in detail."}]}]
    try:
        resp = dashscope.MultiModalConversation.call(model=QWEN_VL_MODEL, messages=msg)
        if resp.status_code == HTTPStatus.OK:
            return resp.output["choices"][0]["message"]["content"] or "No description."
    except Exception as e:
        print(f"[Caption error] {e}")
    return "Description error."


def generate_qwen_vl_response_api(query: str, retrieved: List[Dict[str, Any]], max_images: int = 1) -> str:
    """Answer *only* from retrieved multimodal context using Qwen‑VL."""
    system = {"role": "system", "content": [{"text": "You are a helpful assistant. Answer ONLY from provided context."}]}
    user_content: List[Dict[str, str]] = []

    # attach up to `max_images` images
    added = 0
    for item in retrieved:
        if item.get("type") == "image_caption" and added < max_images and os.path.exists(item.get("image_path", "")):
            user_content.append({"image": f"file://{os.path.abspath(item["image_path"])}"})
            added += 1

    # textual context
    texts = [it["content"] for it in retrieved if it.get("type") == "text"]
    if texts:
        user_content.append({"text": "--- Context ---\n" + "\n\n".join(texts) + "\n--- End ---"})

    # user question
    user_content.append({"text": f"Question: {query}"})
    user = {"role": "user", "content": user_content}

    try:
        resp = dashscope.MultiModalConversation.call(model=QWEN_VL_MODEL, messages=[system, user])
        if resp.status_code == HTTPStatus.OK:
            ans = resp.output["choices"][0]["message"]["content"]
            # API may return list or str
            if isinstance(ans, list):
                return "".join(x.get("text", "") for x in ans)
            return ans
    except Exception as e:
        return f"[Generation error] {e}"
    return "Generation failed."

# ---------------------------------------------------------------------------
# Index construction + storage (FAISS)
# ---------------------------------------------------------------------------

def extract_and_index_api(data_dir: str, image_save_dir: str, index_dir: str):
    """Parse PDFs, extract text & images, embed, and build FAISS index."""
    os.makedirs(image_save_dir, exist_ok=True)
    texts_meta, images_meta = [], []

    print("[1] Extracting text & images from PDFs …")
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".pdf"):
            continue
        fp = os.path.join(data_dir, fn)
        try:
            doc = fitz.open(fp)
            for page_num in range(len(doc)):
                pg = doc.load_page(page_num)
                # text blocks
                for blk in pg.get_text("blocks"):
                    txt = blk[4].strip()
                    if len(txt) > 30:
                        texts_meta.append({
                            "type": "text",
                            "content": txt,
                            "source": f"{fn}:page{page_num + 1}"
                        })
                # images
                for img in pg.get_images(full=True):
                    xref = img[0]
                    imgd = doc.extract_image(xref)
                    ext = imgd["ext"].lower()
                    if ext not in ("png", "jpg", "jpeg", "webp"):
                        continue
                    name = f"img_{uuid.uuid4()}.{ext}"
                    savep = os.path.join(image_save_dir, name)
                    Image.open(io.BytesIO(imgd["image"])).save(savep)
                    images_meta.append({
                        "type": "image",
                        "path": savep,
                        "source": f"{fn}:page{page_num + 1}"
                    })
            doc.close()
        except Exception as e:
            print(f"[Extract error] {fn}: {e}")

    # prepare docs for embedding
    print("[2] Generating captions & embeddings …")
    all_meta, docs = [], []
    for t in texts_meta:
        all_meta.append(t)
        docs.append(t["content"])
    for img in images_meta:
        cap = generate_caption_api(img["path"])
        all_meta.append({
            "type": "image_caption",
            "content": cap,
            "image_path": img["path"],
            "source": img["source"]
        })
        docs.append(cap)
        time.sleep(0.2)  # be gentle on API

    # batch embed
    embeddings, final_meta = [], []
    BATCH = 20
    for i in range(0, len(docs), BATCH):
        batch = docs[i : i + BATCH]
        embs = get_text_embeddings_api(batch)
        for m, e in zip(all_meta[i : i + BATCH], embs):
            if e is not None:
                embeddings.append(e)
                final_meta.append(m)
        time.sleep(0.3)

    if not embeddings:
        raise RuntimeError("No embeddings produced – aborting index build.")

    # build FAISS
    arr = np.asarray(embeddings, dtype="float32")
    arr = normalize(arr, axis=1)
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    # persist
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index_to_doc.pkl"), "wb") as f:
        pickle.dump(final_meta, f)
    with open(os.path.join(index_dir, "embeddings.pkl"), "wb") as f:
        pickle.dump(arr, f)

    print(f"[✓] Built FAISS with {index.ntotal} vectors.")
    return index, final_meta, arr

# ---------------------------------------------------------------------------
# Retrieval & re‑ranking
# ---------------------------------------------------------------------------

def retrieve_from_index(query: str, index: faiss.Index, mapping: List[Dict[str, Any]], k: int = 10):
    embs = get_text_embeddings_api([query])
    if not embs or embs[0] is None:
        return []
    qv = normalize(np.asarray([embs[0]], dtype="float32"), axis=1)
    D, I = index.search(qv, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(mapping):
            item = dict(mapping[idx])  # shallow copy
            item["score"] = float(dist)
            item["mapping_idx"] = int(idx)
            results.append(item)
    return results


def rerank_results(query: str, candidates: List[Dict[str, Any]], emb_array: np.ndarray):
    embs = get_text_embeddings_api([query])
    if not embs or embs[0] is None:
        return candidates
    qv = normalize(np.asarray([embs[0]], dtype="float32"), axis=1)[0]
    scored = []
    for c in candidates:
        idx = c.get("mapping_idx")
        vec = emb_array[idx]
        new_score = float(np.dot(qv, vec))
        c["score"] = new_score
        scored.append((new_score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]

# ---------------------------------------------------------------------------
# MCP integration (Tavily / Fetch / Filesystem)
# ---------------------------------------------------------------------------

MCP_CONFIGS = {
    "fetch": {
        "command": "uvx",
        "args": ["mcp-server-fetch"]
    },
    "tavily": {
        "command": "python",
        "args": ["tavily_mcp.py"],
        "transport": "stdio",
    },
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/Users/orzjh/Desktop",
            "/Users/orzjh/Desktop/knowledge-base"
        ]
    }
}

async def run_agent(message: str) -> str:
    """Fire up an ephemeral MCP agent for the given message and return its answer."""
    tools, cleanup = await convert_mcp_to_langchain_tools(MCP_CONFIGS)
    try:
        model = ChatOpenAI(model_name="gpt-4o")
        agent = create_react_agent(model, tools)
        resp = await agent.ainvoke({"messages": message})
        return resp["messages"][-1].content
    finally:
        await cleanup()

# ---------------------------------------------------------------------------
# Main CLI loop (async) – prefix "mcp:" routes to external tools, else local RAG
# ---------------------------------------------------------------------------

async def chat_loop():
    # load or build index
    idx_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    meta_path = os.path.join(VECTOR_STORE_PATH, "index_to_doc.pkl")
    emb_path  = os.path.join(VECTOR_STORE_PATH, "embeddings.pkl")

    if all(os.path.exists(p) for p in (idx_path, meta_path, emb_path)):
        print("[Load] Existing FAISS index …")
        faiss_index = faiss.read_index(idx_path)
        mapping     = pickle.load(open(meta_path, "rb"))
        emb_array   = pickle.load(open(emb_path,  "rb"))
    else:
        faiss_index, mapping, emb_array = extract_and_index_api(DATA_DIR, IMAGE_SAVE_DIR, VECTOR_STORE_PATH)

    print("\n=== Multimodal RAG CLI (type 'quit' to exit) ===")
    while True:
        raw = input("You: ").strip()
        if raw.lower() in {"quit", "exit"}:
            print("Good‑bye!"); break

        if raw.lower().startswith("mcp:"):
            # route to MCP agent
            query = raw[len("mcp:"):].lstrip()
            print("[MCP] running agent …")
            answer = await run_agent(query)
            print("Agent:", answer)
            continue

        # ----- local multimodal RAG -----
        rewritten = rewrite_query(raw)
        top_k = retrieve_from_index(rewritten, faiss_index, mapping, k=10)
        reranked = rerank_results(rewritten, top_k, emb_array)[:5]
        print(f"[Top‑5 scores] {[round(x['score'], 4) for x in reranked]}")
        print("Assistant: ", end="", flush=True)
        ans = generate_qwen_vl_response_api(raw, reranked)
        # crude streaming effect
        for ch in ans:
            print(ch, end="", flush=True)
            time.sleep(0.003)
        print()

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        anyio.run(chat_loop)
    except KeyboardInterrupt:
        print("\nInterrupted – exiting.")
