import streamlit as st
from transformers import pipeline

# No caching
def create_simple_llm():
    # Fixed model name
    generator = pipeline("text-generation", model="distilgpt2", pad_token_id=50256)
    return generator

def generate_text(generator, prompt, max_length=100, num_return_sequences=1):
    output = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return output

# Streamlit UI
def main():
    st.title("📝 Text Generation with distilgpt2")
    st.markdown("Enter a prompt, and the AI will generate text based on it.")

    # This will reload the model on every run (slower, but uncached)
    generator = create_simple_llm()

    prompt = st.text_area("Enter a prompt", value="Once upon a time")
    max_length = st.slider("Max length", min_value=20, max_value=300, value=100)
    num_sequences = st.number_input("Number of sequences", min_value=1, max_value=5, value=1)

    if st.button("Generate Text"):
        with st.spinner("Generating..."):
            results = generate_text(generator, prompt, max_length=max_length, num_return_sequences=num_sequences)
            for idx, result in enumerate(results):
                st.subheader(f"Generated #{idx + 1}")
                st.write(result['generated_text'])

if __name__ == "__main__":
    main()
