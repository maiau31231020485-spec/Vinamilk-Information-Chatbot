import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import nltk
import random
import time
import base64
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import io
import contextlib
import sys

# Tải tài nguyên NLTK
nltk.download('punkt')
stemmer = PorterStemmer()

# Định nghĩa dữ liệu intents
intents = {
    "intents": [
        {"tag": "chao_hoi", "patterns": ["Chào", "Hi", "Hello", "Xin chào", "Chào buổi sáng", "Chào buổi chiều", "Chào buổi tối"], 
         "responses": ["Chào bạn! Tôi có thể giúp gì cho bạn về Vinamilk?", "Xin chào! Bạn muốn tìm hiểu điều gì về Vinamilk?", "Chào mừng bạn đến với Vinamilk!", "Rất vui được hỗ trợ bạn. Hãy đặt câu hỏi về Vinamilk nhé!"]},
        {"tag": "gioi_thieu", "patterns": ["Vinamilk là gì?", "Giới thiệu về công ty", "Thông tin Vinamilk"], 
         "responses": ["Vinamilk là công ty sữa hàng đầu Việt Nam."]},
        {"tag": "tam_biet", "patterns": ["Tạm biệt", "Chào tạm biệt", "Bye", "Cảm ơn bạn nha!"], 
         "responses": ["Tạm biệt bạn! Hẹn gặp lại", "Chúc bạn một ngày tốt lành!", "Hẹn gặp lại bạn trong lần trò chuyện tiếp theo!", "Cảm ơn bạn đã ghé thăm Vinamilk!💖", "iu iu 💕", "Hẹn gặp lại bạn nhé! Chúc bạn một ngày tuyệt vời cùng Vinamilk 🥛❤️"]}
    ]
}

def tokenize(sentence):
    return word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

all_words = []
tags = []
xy = []
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
ignore_words = ['?', '.', '!', ',']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.l2(out)
        return out

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
inputs = torch.from_numpy(X_train).float()
targets = torch.from_numpy(y_train).long()
for epoch in range(500):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()

# Ảnh nền base64
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning("Không tìm thấy file hình nền.")
        return None
img_path = "veny/Vinamilk.png"
img_base64 = get_base64_of_bin_file(img_path)

# Giao diện
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;600&display=swap');
    html, body, .stApp {{ font-family: 'Roboto', sans-serif !important; }}
    .stApp {{ background: {'#f0f2f6' if not img_base64 else f'url("data:image/png;base64,{img_base64}")'}; background-size: cover; background-position: center; background-attachment: fixed; position: relative; min-height: 100vh; }}
    .stApp::before {{ content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.4); z-index: -1; }}
    h1 {{ text-align: center; color: white !important; margin-bottom: 30px; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); }}
    .bot-msg {{ background-color: #0071CE !important; color: #ffffff !important; padding: 12px 18px !important; border-radius: 15px !important; margin-bottom: 15px !important; display: inline-block !important; font-size: 1.1em !important; border: 1px solid #ffffff !important; background: linear-gradient(145deg, #0071CE, #005bb5) !important; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3) !important; }}
    .user-msg {{ background-color: #e0f0ff !important; color: #000000 !important; padding: 12px 18px !important; border-radius: 15px !important; margin-bottom: 15px !important; display: inline-block !important; font-size: 1.1em !important; border: 1px solid #b0d4f1 !important; background: linear-gradient(145deg, #e0f0ff, #c3e0ff) !important; box-shadow: none !important; float: right !important; margin-left: auto !important; margin-right: 0 !important; max-width: 80% !important; }}
    .stChatMessage {{ background-color: transparent !important; box-shadow: none !important; padding: 0 !important; margin-bottom: 0 !important; }}
    .stTextInput > div > div > input {{ font-size: 1.1em !important; padding: 12px !important; border: 1px solid #333333 !important; border-radius: 15px !important; background-color: #ffffff !important; }}
    .stChatInputContainer {{ box-shadow: none !important; background: none !important; }}
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstores/db_faiss", embedding_model, allow_dangerous_deserialization=True)

@st.cache_resource
def load_large_llm():
    return CTransformers(model="models/vinallama-7b-chat_q5_0.gguf", model_type="llama", max_new_tokens=1024, temperature=0.01)

db = load_vector_db()
llm = load_large_llm()
template = """Bạn là một trợ lý AI. Dưới đây là một số thông tin nội bộ của công ty:

{context}

Dựa vào thông tin trên, hãy trả lời câu hỏi sau:
Câu hỏi: {question}
Trả lời:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
llm_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 2}), return_source_documents=False, chain_type_kwargs={"prompt": prompt})

df_nv = pd.read_csv("data/tra_cuu_nhan_vien.csv", encoding="utf-8-sig")

def tra_cuu_nhan_vien(question):
    question_lower = question.lower()
    for _, row in df_nv.iterrows():
        try:
            ma_nv = int(row["Mã nhân viên"]) if not pd.isna(row["Mã nhân viên"]) else None
            ten_nv = row["Họ và tên"].lower() if not pd.isna(row["Họ và tên"]) else ""
        except:
            continue
        if (ma_nv is not None and str(ma_nv) in question_lower) or (ten_nv in question_lower):
            return row
    return None

def trich_thong_tin_yeu_cau(question, row):
    question = question.lower()
    response_parts = []
    if "phòng ban" in question and not pd.isna(row.get("Phòng ban")):
        response_parts.append(f"Phòng ban của {row['Họ và tên']} là: {row['Phòng ban']}")
    if "chức vụ" in question and not pd.isna(row.get("Chức vụ")):
        response_parts.append(f"Chức vụ của {row['Họ và tên']} là: {row['Chức vụ']}")
    if "mã nhân viên" in question and not pd.isna(row.get("Mã nhân viên")):
        response_parts.append(f"Mã nhân viên của {row['Họ và tên']} là: {int(row['Mã nhân viên'])}")
    if "ngày vào" in question and not pd.isna(row.get("Ngày vào công ty")):
        response_parts.append(f"Ngày vào công ty của {row['Họ và tên']} là: {row['Ngày vào công ty']}")
    if "nghỉ phép" in question and not pd.isna(row.get("Số ngày nghỉ phép còn lại")):
        response_parts.append(f"{row['Họ và tên']} còn lại {int(row['Số ngày nghỉ phép còn lại'])} ngày nghỉ phép")
    if not response_parts:
        return f'''Thông tin của {row["Họ và tên"]}:
- Mã nhân viên: {int(row["Mã nhân viên"])}
- Phòng ban: {row["Phòng ban"]}
- Chức vụ: {row["Chức vụ"]}
- Ngày vào công ty: {row["Ngày vào công ty"]}
- Số ngày nghỉ phép còn lại: {int(row["Số ngày nghỉ phép còn lại"])}'''
    return "\n".join(response_parts)

def chatbot(question):
    # Kiểm tra intents trước
    tokenized_question = tokenize(question)
    bag = bag_of_words(tokenized_question, all_words)
    bag_tensor = torch.from_numpy(bag).float()
    
    with torch.no_grad():
        output = model(bag_tensor)
        probabilities = torch.softmax(output, dim=0)
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()
    
    # Nếu confidence đủ cao (>0.75), trả về câu trả lời từ intents
    if confidence > 0.75:
        tag = tags[predicted_index]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    row = tra_cuu_nhan_vien(question)
    if row is not None and not row.isnull().all():
        return trich_thong_tin_yeu_cau(question, row)
    if any(x in question.lower() for x in ["nhân viên", "chức vụ", "phòng ban", "nghỉ phép", "mã nhân viên", "ngày vào"]):
        return "Không tìm thấy thông tin bạn cần!"
    with contextlib.redirect_stdout(io.StringIO()):
        response = llm_chain.invoke({"query": question})
    return response["result"]

st.markdown('<h1>Vinamilk Chatbot 🐄</h1>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "bot", "content": "Chào bạn! Hãy hỏi tôi về Vinamilk."}]
if "last_input" not in st.session_state:
    st.session_state.last_input = None

for message in st.session_state.messages:
    if message["role"] == "bot":
        with st.chat_message(message["role"], avatar="veny/Logo Vinamilk.png"):
            st.markdown(f'<div class="bot-msg">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="user-msg">{message["content"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_input = prompt
    st.rerun()

if st.session_state.last_input:
    question = st.session_state.last_input
    st.session_state.last_input = None
    st.session_state.messages.append({"role": "bot", "content": "Đang tìm kiếm thông tin..."})
    st.rerun()

if st.session_state.messages[-1]["content"] == "Đang tìm kiếm thông tin...":
    question = next(m["content"] for m in reversed(st.session_state.messages) if m["role"]=="user")
    response = chatbot(question)
    st.session_state.messages[-1] = {"role": "bot", "content": response}
    st.rerun()