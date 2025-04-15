from transformers import pipeline


def text_summarizer(text, max_length=150):
    # Explicitly use PyTorch implementation
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        framework="pt"  # Force PyTorch usage
    )
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=30,
        do_sample=False,
        truncation=True
    )
    return summary[0]['summary_text']

if __name__ == "__main__":
    input_text = """
   Machine learning is a branch of artificial intelligence that focuses on developing 
   algorithms and statistical models that enable computers to perform tasks without 
   explicit programming. These algorithms use historical data as input to predict 
   new output values. Common applications include recommendation systems,
   image recognition, and natural language processing. Deep learning, a subset 
   of machine learning, utilizes neural networks with multiple layers to analyze 
   complex patterns in data. The field continues to evolve rapidly with new 
   architectures and techniques emerging regularly."""
    
    print("Original Text Length:", len(input_text))
    result = text_summarizer(input_text)
    print("\nSummary:", result)
    print("Summary Length:", len(result))