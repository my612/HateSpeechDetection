// This is a mock service that simulates the API call for text analysis
// In a real application, this would be replaced with actual API calls
export const analyzeMockText = text => {
  // Simulate API call delay
  return new Promise(resolve => {
    setTimeout(() => {
      // List of words that might be considered problematic in our mock analysis
      const potentialHateWords = ['hate', 'terrible', 'awful', 'stupid', 'idiot', 'dumb', 'disgusting', 'horrible', 'worst', 'pathetic', 'useless'];
      // Find all occurrences of potentially problematic words
      const wordAttributions = [];
      let overallScore = 0;
      let matchCount = 0;
      potentialHateWords.forEach(word => {
        const regex = new RegExp(`\\b${word}\\b`, 'gi');
        let match;
        while ((match = regex.exec(text)) !== null) {
          const startIndex = match.index;
          const endIndex = startIndex + match[0].length;
          // Generate a score between 0.3 and 0.9 for this word
          // More "severe" words get higher scores
          const baseScore = 0.3 + potentialHateWords.indexOf(word.toLowerCase()) / potentialHateWords.length * 0.6;
          wordAttributions.push({
            word: match[0],
            startIndex,
            endIndex,
            score: baseScore
          });
          overallScore += baseScore;
          matchCount++;
        }
      });
      // Calculate overall confidence score (0-100)
      const confidenceScore = matchCount > 0 ? Math.min(Math.round(overallScore / matchCount * 100), 99) : 5;
      // Determine if text is classified as hate speech based on confidence
      const isHateSpeech = confidenceScore > 30;
      resolve({
        text,
        isHateSpeech,
        confidenceScore,
        wordAttributions
      });
    }, 1500); // 1.5 second delay to simulate processing time
  });
};