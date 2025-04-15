# CODTECH 01 TEXT_SUMMARIZATION_TOOL
# Requirements: pip install nltk sumy

import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Download necessary NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Add this missing resource
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

def summarize_text(text, sentences_count=3):
    """
    Summarizes input text using Latent Semantic Analysis
    Returns concise summary with specified number of sentences
    """
    # Parse text using plaintext parser
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # Initialize LSA summarizer
    summarizer = LsaSummarizer()
    
    # Generate summary
    summary = summarizer(parser.document, sentences_count)
    
    # Return joined summary sentences
    return ' '.join([str(sentence) for sentence in summary])

if __name__ == "__main__":
    # Example usage with sample text
    sample_text = """
    Machine learning is a branch of artificial intelligence that focuses on developing 
    algorithms and statistical models that enable computers to perform tasks without 
    explicit programming. These algorithms use historical data as input to predict 
    new output values. Common applications include recommendation systems,
    image recognition, and natural language processing. Deep learning, a subset 
    of machine learning, utilizes neural networks with multiple layers to analyze 
    complex patterns in data. The field continues to evolve rapidly with new 
    architectures and techniques emerging regularly.
    """
    
    print("Original Text:")
    print(sample_text)
    print("\nSummary:")
    print(summarize_text(sample_text))