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

# Tải tokenizer
stemmer = PorterStemmer()
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Dữ liệu intents
intents = {
    "intents": [
        {
            "tag": "chao_hoi",
            "patterns": ["Chào", "Hi👋", "Hello👋", "Xin chào👋", "Tôi muốn hỏi về Vinamilk", "Chào buổi sáng", "Chào buổi chiều", "Chào buổi tối"],
            "responses": ["Chào bạn👋! Tôi có thể giúp gì cho bạn về Vinamilk?","Xin chào! Bạn muốn tìm hiểu điều gì về Vinamilk?", "Chào mừng bạn đến với Vinamilk!","Rất vui được hỗ trợ bạn. Hãy đặt câu hỏi về Vinamilk nhé!"]
        },
        {
            "tag": "gioi_thieu",
            "patterns": ["Vinamilk là gì?", "Giới thiệu về công ty", "Thông tin Vinamilk"],
            "responses": ["🥛Vinamilk là công ty sữa hàng đầu Việt Nam."]
        },
        {
            "tag": "san_pham",
            "patterns": ["Vinamilk bán sản phẩm gì?", "Các dòng sữa", "Vinamilk có sữa gì?"],
            "responses": ["Vinamilk cung cấp các loại sữa tươi, sữa chua, sữa đặc..."]
        },
        {
            "tag": "doi_tac",
            "patterns": ["Đối tác của Vinamilk là ai?", "Vinamilk hợp tác với ai?"],
            "responses": ["Vinamilk có nhiều đối tác trong và ngoài nước, như Miraka (New Zealand)..."]
        },
        {
            "tag": "tam_biet",
            "patterns": ["Tạm biệt 👋", "Chào tạm biệt👋", "Bye", "Cảm ơn bạn nha!"],
            "responses": ["Tạm biệt bạn! Hẹn gặp lại", "Chúc bạn một ngày tốt lành!","Hẹn gặp lại bạn trong lần trò chuyện tiếp theo!",    "Cảm ơn bạn đã ghé thăm Vinamilk!💖",  "iu iu 💕","Hẹn gặp lại bạn nhé! Chúc bạn một ngày tuyệt vời cùng Vinamilk 🥛❤️"]
        }
        
    ]
}

# Xử lý văn bản
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

# Chuẩn bị dữ liệu
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

# Mô hình đơn giản
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

for epoch in range(100):
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

img_path = "Vinamilk.png"
img_base64 = get_base64_of_bin_file(img_path)

# Giao diện
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;600&display=swap');

    html, body, .stApp {{
        font-family: 'Roboto', sans-serif !important;
    }}

    
    .stApp {{
        background: {'#f0f2f6' if not img_base64 else f'url("data:image/png;base64,{img_base64}")'};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        position: relative;
        min-height: 100vh;
    }}
    .stApp::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.4);
        z-index: -1;
    }}
    h1 {{
        text-align: center;
        color: white !important;
        margin-bottom: 30px;
        font-size: 2.5em;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }}
    .bot-msg {{
        background-color: #0071CE !important;
        color: #ffffff !important;
        padding: 12px 18px !important;
        border-radius: 15px !important;
        margin-bottom: 15px !important;
        display: inline-block !important;
        font-size: 1.1em !important;
        border: 1px solid #ffffff !important;
        background: linear-gradient(145deg, #0071CE, #005bb5) !important;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3) !important;
    }}
    .user-msg {{
        background-color: #e0f0ff !important;
        color: #000000 !important;
        padding: 12px 18px !important;
        border-radius: 15px !important;
        margin-bottom: 15px !important;
        display: inline-block !important;
        font-size: 1.1em !important;
        border: 1px solid #b0d4f1 !important;
        background: linear-gradient(145deg, #e0f0ff, #c3e0ff) !important;
        box-shadow: none !important;
        float: right !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        max-width: 80% !important;
    }}
    .stChatMessage {{
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
    }}
    .stTextInput > div > div > input {{
        font-size: 1.1em !important;
        padding: 12px !important;
        border: 1px solid #333333 !important;
        border-radius: 15px !important;
        background-color: #ffffff !important;
    }}
    .stChatInputContainer {{
        box-shadow: none !important;
        background: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Tiêu đề
st.markdown('<h1>Vinamilk Chatbot 🐄</h1>', unsafe_allow_html=True)

# Lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "bot", "content": "Chào bạn! Hãy hỏi tôi về Vinamilk."}]

# Hiển thị lịch sử
#for message in st.session_state.messages:
    #with st.chat_message(message["role"]):
        #st.markdown(f'<div class="{message["role"]}-msg">{message["content"]}</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "bot":
        with st.chat_message(message["role"], avatar="Logo Vinamilk.png"):
            st.markdown(f'<div class="bot-msg">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="user-msg">{message["content"]}</div>', unsafe_allow_html=True)
# Nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)

    sentence_tokens = tokenize(prompt)
    X = bag_of_words(sentence_tokens, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    with torch.no_grad():
        output = model(X)
        _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    response = "Tôi không hiểu câu hỏi. Hãy thử lại!"
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            break

    # Hiệu ứng gõ – nhưng không lặp lại phản hồi
    with st.chat_message("bot",avatar="Logo Vinamilk.png"):
        placeholder = st.empty()
        typed_text = ""
        for char in response:
            typed_text += char
            placeholder.markdown(f'<div class="bot-msg">{typed_text}</div>', unsafe_allow_html=True)
            time.sleep(0.03)

    st.session_state.messages.append({"role": "bot", "content": response})
    #with st.chat_message("bot"):
        #st.markdown(f'<div class="bot-msg">{response}</div>', unsafe_allow_html=True)
    #with st.chat_message("bot", avatar="avttt.png"):
        #st.markdown(f'<div class="bot-msg">{response}</div>', unsafe_allow_html=True)

