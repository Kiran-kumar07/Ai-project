from transformers import pipeline

def summarize_text(text):
    # Load summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Split long text into chunks if it's longer than max_token_limit
    max_chunk_size = 1024  # BART model limit
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    summary = ""
    for chunk in chunks:
        summary_part = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += summary_part[0]['summary_text'] + " "

    return summary.strip()

# Sample lengthy article input
input_text = """
Artificial Intelligence (AI) has been transforming industries through automation, prediction, and intelligent decision-making.
From self-driving cars to voice assistants and healthcare diagnostics, AI technologies are embedded into our daily lives.
Natural Language Processing (NLP), a subfield of AI, enables machines to understand, interpret, and respond to human language.
Recent breakthroughs like OpenAI's GPT and BERT from Google have drastically improved machine comprehension.
This has paved the way for better translation systems, smarter chatbots, and even tools that write code or summarize articles automatically.
However, concerns around AI ethics, data privacy, and misuse continue to rise as AI becomes more capable.
Despite the challenges, AI continues to grow rapidly, impacting education, business, and governance.
"""

# Run the summarizer
summary = summarize_text(input_text)

# Output
print("\n--- Input Text ---\n")
print(input_text)
print("\n--- Generated Summary ---\n")
print(summary)
