"""Streamlit interface for the trained PromptCraft adapter."""

from pathlib import Path

import streamlit as st

from src.inference import generate_optimized_prompt
from src.model_loader import DEFAULT_MODEL, load_model_and_tokenizer

st.set_page_config(page_title="PromptCraft AI", page_icon="✨", layout="wide")
st.title("PromptCraft AI")
st.caption("A custom-dataset fine-tuned prompt optimizer")
st.write("Turn an unclear request into a clearer, more useful prompt while preserving the original intent.")

with st.sidebar:
    st.header("Model")
    model_name = st.text_input("Base model", DEFAULT_MODEL)
    adapter_path = st.text_input("LoRA adapter path", "models/promptcraft-lora")
    st.info("The app requires a trained adapter. It never silently substitutes the base model.")

@st.cache_resource(show_spinner=False)
def get_model(name: str, adapter: str):
    return load_model_and_tokenizer(name, adapter)

with st.form("optimizer"):
    original = st.text_area("Original prompt", height=160, placeholder="Explain machine learning.")
    category = st.selectbox("Category", ["Education", "Coding", "Research", "Writing", "Presentations",
                                         "Image Generation", "Data Analysis", "Career", "Security", "General"])
    col1, col2 = st.columns(2)
    with col1:
        audience = st.text_input("Target audience")
        output_format = st.text_input("Desired output format")
    with col2:
        tone = st.text_input("Tone")
        requirements = st.text_area("Additional requirements", height=80)
    submitted = st.form_submit_button("Optimize Prompt", type="primary")

if submitted:
    try:
        if not original.strip():
            st.error("Enter an original prompt first.")
        elif not Path(adapter_path).exists():
            st.error(f"Trained adapter not found at `{adapter_path}`. Run the training command from the README first.")
        else:
            with st.spinner("Loading the fine-tuned adapter and generating..."):
                model, tokenizer, mode = get_model(model_name, adapter_path)
                result, seconds = generate_optimized_prompt(
                    model, tokenizer, original, category, audience, output_format, tone, requirements
                )
            st.success(f"Generated with {mode} in {seconds:.2f}s")
            left, right = st.columns(2)
            with left:
                st.subheader("Original")
                st.write(original)
            with right:
                st.subheader("Optimized")
                st.write(result)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(str(exc))

st.divider()
st.subheader("How it works")
st.markdown("1. The supplied CSV/JSONL is validated and split without overwriting the source.  \n"
            "2. The Qwen instruction model is fine-tuned with a LoRA adapter.  \n"
            "3. This page loads that adapter on the base model and generates an optimized prompt.")
st.caption("Subjective quality claims require human ratings in outputs/evaluation_results.csv; the starter dataset is AI-assisted and pending human review.")
