#%%
from pathlib import Path
import pickle
import numpy as np
import os



# Load the TF-IDF scores and stopwords set@7
# 2. Model/Artifact loading at startup
BASE_DIR = Path(__file__).parent.parent

HATE_TFIDF = BASE_DIR / "artifacts/hate_tfidf_scores.pkl"
STOP_WORDS    = BASE_DIR /  "artifacts/stopwords_set.pkl"
# Hyperparameters needed for model instantiation


# Load the preprocessed data
def load_tfidf_data(tfidf_scores_path=HATE_TFIDF, 
                    stopwords_path=STOP_WORDS):
    """Load the TF-IDF scores and stopwords set."""
    
    # Check if files exist
    if not os.path.exists(tfidf_scores_path):
        raise FileNotFoundError(f"TF-IDF scores file not found: {tfidf_scores_path}")
    if not os.path.exists(stopwords_path):
        raise FileNotFoundError(f"Stopwords file not found: {stopwords_path}")
    
    # Load TF-IDF scores
    with open(tfidf_scores_path, "rb") as f:
        tfidf_scores = pickle.load(f)
    
    # Load stopwords set
    with open(stopwords_path, "rb") as f:
        stopwords_set = pickle.load(f)
    
    print(f"Loaded {len(tfidf_scores)} TF-IDF scores and {len(stopwords_set)} stopwords")
    return tfidf_scores, stopwords_set

# Load the data
tfidf_scores, stopwords_set = load_tfidf_data()

def enhance_attributions(tokens, attribution_scores, tfidf_scores=tfidf_scores, 
                         stopwords_set=stopwords_set, tfidf_threshold=0.01, 
                         attribution_threshold=0.1):
    """
    Enhance model attributions using TF-IDF scores.
    
    Args:
        tokens: List of tokens from the input text
        attribution_scores: Model attribution scores (from Integrated Gradients)
        tfidf_scores: Dictionary of word/phrase to TF-IDF score
        stopwords_set: Set of stopwords to filter out (excluding offensive words)
        tfidf_threshold: Minimum TF-IDF score to consider
        attribution_threshold: Minimum attribution score to consider
    
    Returns:
        enhanced_scores: Enhanced attribution scores
        highlight_flags: Boolean flags for words to highlight
    """
    enhanced_scores = []
    highlight_flags = []
    
    # Handle empty input or edge cases
    if not tokens or not attribution_scores:
        return [], []
    
    # Normalize attribution scores to [0,1] range
    attr_abs = [abs(score) for score in attribution_scores]
    max_attr = max(attr_abs) if max(attr_abs) > 0 else 1.0
    norm_attributions = [abs(score)/max_attr for score in attribution_scores]
    
    # Check for bigrams that might be in our TF-IDF vocabulary
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append((tokens[i] + " " + tokens[i+1]).lower())
    
    # First pass: check which tokens are parts of high-scoring bigrams
    in_high_scoring_bigram = [False] * len(tokens)
    for i, bigram in enumerate(bigrams):
        if bigram in tfidf_scores and tfidf_scores[bigram] >= tfidf_threshold:
            in_high_scoring_bigram[i] = True
            in_high_scoring_bigram[i+1] = True
    
    # Second pass: calculate scores for each token
    for i, (token, attr_score) in enumerate(zip(tokens, norm_attributions)):
        token_lower = token.lower()
        
        # Skip stopwords and common words (unless part of a high-scoring bigram)
        if token_lower in stopwords_set and not in_high_scoring_bigram[i]:
            enhanced_scores.append(0.0)
            highlight_flags.append(False)
            continue
        
        # Get TF-IDF score (default to 0 if not in our vocabulary)
        tfidf_score = tfidf_scores.get(token_lower, 0.0)
        
        # Look for bigrams this token is part of
        bigram_boost = 0.0
        if i < len(tokens) - 1:
            bigram = token_lower + " " + tokens[i+1].lower()
            bigram_boost = max(bigram_boost, tfidf_scores.get(bigram, 0.0))
        if i > 0:
            bigram = tokens[i-1].lower() + " " + token_lower
            bigram_boost = max(bigram_boost, tfidf_scores.get(bigram, 0.0))
        
        # Use the higher of unigram or bigram TF-IDF
        effective_tfidf = max(tfidf_score, bigram_boost)
        
        # Combine attribution and TF-IDF
        # Formula: boost attribution by TF-IDF factor
        combined_score = attr_score * (1.0 + 2.0 * effective_tfidf)
        
        # Determine if this word should be highlighted
        highlight = (attr_score >= attribution_threshold and 
                    (effective_tfidf >= tfidf_threshold or in_high_scoring_bigram[i]) and
                    (token_lower not in stopwords_set or in_high_scoring_bigram[i]))
        
        enhanced_scores.append(combined_score)
        highlight_flags.append(highlight)
    
    return enhanced_scores, highlight_flags

# Example function to get the detailed explanation with enhanced attributions
def get_explanation(tokens, attribution_scores, tfidf_scores=tfidf_scores, 
                   stopwords_set=stopwords_set):
    """
    Get a detailed explanation of token attributions.
    
    Args:
        tokens: List of tokens from the input text
        attribution_scores: Raw attribution scores from the model
        
    Returns:
        List of dictionaries with token information
    """
    enhanced_scores, highlight_flags = enhance_attributions(
        tokens, attribution_scores, tfidf_scores, stopwords_set
    )
    
    explanation = []
    for token, raw_score, enhanced, highlight in zip(
            tokens, attribution_scores, enhanced_scores, highlight_flags):
        explanation.append({
            "token": token,
            "raw_attribution": float(raw_score),
            "enhanced_attribution": float(enhanced),
            "highlight": highlight
        })
    
    return explanation

# Example of using the function with dummy data (for testing)
if __name__ == "__main__":
    # This is a test example - replace with your actual model's output
    test_tokens = ["Muslims", "should", "be", "banned", "from", "entering", "this", "country"]
    # Dummy attribution scores - in real use, these would come from your IG implementation
    test_attributions = [0.8, 0.1, 0.05, 0.7, 0.05, 0.15, 0.1, 0.3]
    
    explanation = get_explanation(test_tokens, test_attributions)
    
    print("\nExample explanation:")
    for item in explanation:
        highlight_marker = "⚠️ " if item["highlight"] else ""
        print(f"{highlight_marker}{item['token']}: raw={item['raw_attribution']:.3f}, "
              f"enhanced={item['enhanced_attribution']:.3f}")
