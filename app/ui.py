"""
Streamlit chat page. Start the API first, then:

  streamlit run app/ui.py

Set API_URL if the API is not on http://localhost:8000.
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Arcvale Knowledge Base", page_icon="📘")
st.title("📘 Arcvale Knowledge Base AI")
st.caption("Ask about leave, attendance, working hours, travel, reimbursement, benefits, IT policy, "
           "onboarding or the code of conduct.")


@st.cache_data(ttl=60)
def get_info():
    try:
        return requests.get(API_URL + "/info", timeout=5).json()
    except requests.RequestException:
        return None


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model info")
    info = get_info()
    if info is None:
        st.error("API not reachable at {}.\nStart it with:\n`uvicorn app.api:app --port 8000`".format(API_URL))
    else:
        st.metric("Parameters", "{:.1f}M".format(info["parameters"] / 1e6))
        st.write("**Layers:** {} · **d_model:** {} · **heads:** {}".format(info["layers"], info["d_model"], info["heads"]))
        st.write("**Context:** {} tokens · **Vocab:** {} (own BPE)".format(info["context"], info["vocab"]))
        st.write("**Training stages:**")
        losses = info.get("stage_val_losses", {})
        for s in info["training_stages"]:
            key = s.split()[1]
            st.write("- {}{}".format(s, " — val loss {:.3f}".format(losses[key]) if key in losses else ""))
        if info.get("final_val_loss") is not None:
            st.write("**Final val loss:** {:.3f}".format(info["final_val_loss"]))
        st.write("**Checkpoint:** `{}` on `{}`".format(info["checkpoint"], info["device"]))
    st.divider()
    st.warning("Disclaimer: answers come from a small experimental language model trained from scratch. "
               "It can be wrong. Always confirm important details with HR or the official policy document.")

# -------------------------------------------------------------------- chat
if "history" not in st.session_state:
    st.session_state.history = []


def show(msg):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("source"):
            st.caption("📄 Source: {}".format(msg["source"]))
        elif msg["role"] == "assistant" and msg.get("latency_ms") is not None:
            st.caption("No policy source")
        if msg.get("latency_ms") is not None:
            st.caption("⏱ {:.0f} ms".format(msg["latency_ms"]))


for m in st.session_state.history:
    show(m)

question = st.chat_input("e.g. How many casual leaves do I get?")
if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.history.append(user_msg)
    show(user_msg)
    try:
        r = requests.post(API_URL + "/ask", json={"question": question}, timeout=60)
        r.raise_for_status()
        data = r.json()
        bot_msg = {"role": "assistant", "content": data["answer"] or "(empty answer)",
                   "source": data["source"], "latency_ms": data["latency_ms"]}
    except requests.RequestException as e:
        bot_msg = {"role": "assistant", "content": "Could not reach the API: {}".format(e)}
    st.session_state.history.append(bot_msg)
    show(bot_msg)
