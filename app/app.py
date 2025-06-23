import streamlit as st
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
from utils import (
    get_concepts, load_transcript, embed_sentences,
    generate_candidates, train_classifier, save_concept, load_concept_model,
    predict_concept_matches
)
import os

st.set_page_config(page_title="🔍 Concept Tracker", layout="wide")
st.title("🎯 Smart Concept Tracker")

# Sidebar: Concept management
concepts = get_concepts()
concept_names = [c["name"] for c in concepts]

st.sidebar.header("🧠 Concept Settings")
mode = st.sidebar.radio("Choose Mode", ["Create New Concept", "Use Existing Concept"])

if mode == "Create New Concept":
    new_concept_name = st.sidebar.text_input("Concept Name")
    example_1 = st.sidebar.text_input("Example 1")
    example_2 = st.sidebar.text_input("Example 2")
    example_3 = st.sidebar.text_input("Example 3")
    example_4 = st.sidebar.text_input("Example 4")
    example_5 = st.sidebar.text_input("Example 5")

    examples = [e for e in [example_1, example_2, example_3, example_4, example_5] if e.strip()]
    if len(examples) < 5:
        st.warning("Please enter 5 example sentences.")
else:
    selected_concept = st.sidebar.selectbox("Pick a Concept", concept_names)
    selected_concept_data = next((c for c in concepts if c["name"] == selected_concept), None)
    if not selected_concept_data:
        st.error("Selected concept file not found. Try refreshing or re-creating it.")
        st.stop()


# Transcript Upload
st.header("📄 Upload Transcript")
uploaded_file = st.file_uploader("Upload a plain .txt file", type=["txt"])

if uploaded_file:
    transcript_sentences = load_transcript(uploaded_file)
    st.success(f"Transcript loaded: {len(transcript_sentences)} sentences.")

    if mode == "Create New Concept" and len(examples) == 5:
        st.header("🔍 Tag Similar Sentences")
        candidates = generate_candidates(transcript_sentences, examples)
        labels = []

        for i, sent in enumerate(candidates):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i+1}.** {sent}")
            with col2:
                choice = st.radio("Match?", ["Yes", "No"], key=i)
                labels.append(1 if choice == "Yes" else 0)

        if st.button("✅ Train Concept"):
            model_path = f"models/{new_concept_name.lower().replace(' ', '_')}_model.pkl"
            train_classifier(candidates, labels, examples, model_path, model)
            save_concept(new_concept_name, examples, model_path)
            st.success("🎉 Concept saved and model trained.")
            concept_model = load_concept_model(model_path)
            matched_sentences = predict_concept_matches(
                transcript_sentences,              # the same transcript
                concept_model,                     # freshly saved model
                examples                           # concept examples
            )
            
            # ✅ Show full transcript with highlights
            import re

            def normalize(text):
                return re.sub(r'[^\w\s]', '', text.lower().strip())

            normalized_matches = set(normalize(s) for s in matched_sentences)

            st.markdown("### 📄 Full Transcript with Highlights")
            st.info(f"🔍 Found **{len(matched_sentences)}** instances of the concept.")

            for line in transcript_sentences:
                norm_line = normalize(line)
                if norm_line in normalized_matches:
                    st.markdown(f"<span style='background-color: #ffff00'>{line}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(line)

            # st.markdown("### 📌 Matches in this transcript")
            # st.info(f"🔍 Found **{len(matched_sentences)}** instances of the concept.")
            # for sent in matched_sentences:
            #     st.write(f"- {sent}")

    elif mode == "Use Existing Concept":
        st.header("📄 Full Transcript with Highlights")

        model = load_concept_model(selected_concept_data["model_path"])
        examples = selected_concept_data["examples"]
        matched_sentences = predict_concept_matches(transcript_sentences, model, examples)

        import re

        def normalize(text):
            return re.sub(r'[^\w\s]', '', text.lower().strip())

        normalized_matches = set(normalize(s) for s in matched_sentences)

        st.metric("Mentions Found", len(matched_sentences))

        # Join full transcript
        full_text = " ".join(transcript_sentences)

        # Highlight matched sentences
        highlighted_text = full_text
        for sentence in matched_sentences:
            escaped = re.escape(sentence)
            highlighted_text = re.sub(
                escaped,
                f"<span style='background-color: #006400; color: white'>{sentence}</span>",
                highlighted_text,
                flags=re.IGNORECASE
            )

        st.markdown("### 📄 Full Transcript with Highlights")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # for line in transcript_sentences:
        #     norm_line = normalize(line)
        #     if norm_line in normalized_matches:
        #         st.markdown(f"<div style='background-color:#006400;padding:5px;border-radius:5px'><b>{line}</b></div>", unsafe_allow_html=True)
        #     else:
        #         st.markdown(f"<div style='padding:5px'>{line}</div>", unsafe_allow_html=True)

    # elif mode == "Use Existing Concept":
    #     st.header("📌 Predicted Concept Matches")
    #     model = load_concept_model(selected_concept_data["model_path"])
    #     examples = selected_concept_data["examples"]
    #     matches = predict_concept_matches(transcript_sentences, model, examples)
    #     st.metric("Mentions Found", len(matches))

    #     with st.expander("🔍 See Matching Sentences"):
    #         for sent in matches:
    #             st.markdown(f"- {sent}")
