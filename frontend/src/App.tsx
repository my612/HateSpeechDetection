import React, { useState } from 'react';
import { InputPanel } from './components/InputPanel';
import { OutputPanel } from './components/OutputPanel';
import { predictText, submitFeedback, skipFeedback } from './apis'; // Update import to match your file structure

export function App() {
  const [inputText, setInputText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const handleTextChange = text => {
    setInputText(text);
    if (prediction) {
      setPrediction(null);
    }
    if (error) {
      setError('');
    }
    // Reset feedback state when text changes
    setFeedbackSubmitted(false);
  };

  const getPredictionResult = (pred) => {
    if (!pred) return 'not-hate';
    if (pred.probability > 0.7) return 'likely-hate';
    if (pred.probability > 0.4) return 'possible-hate';
    return 'not-hate';
  };

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }
    
    try {
      setIsAnalyzing(true);
      setError('');
      setFeedbackSubmitted(false);
      
      // Use the real API
      const result = await predictText(inputText);
      setPrediction(result);
    } catch (err) {
      setError("We're having trouble analyzing your text right now. Please try again.");
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setInputText('');
    setPrediction(null);
    setError('');
    setFeedbackSubmitted(false);
  };

  // Handle providing feedback (Yes/No)
  const handleProvideFeedback = async (isCorrect) => {
    if (!prediction?.prediction_id) return;
    
    try {
      // Call feedback API
      await submitFeedback(prediction.prediction_id, isCorrect);
      setFeedbackSubmitted(true);
    } catch (err) {
      console.error("Error submitting feedback:", err);
      setError("Unable to submit feedback. Please try again later.");
    }
  };

  // Handle skipping feedback
  const handleSkipFeedback = async () => {
    if (!prediction?.prediction_id) return;
    
    try {
      // Call skip feedback API
      await skipFeedback(prediction.prediction_id);
      
      // Mark feedback as submitted to hide feedback buttons
      setFeedbackSubmitted(true);
    } catch (err) {
      console.error("Error skipping feedback:", err);
      setError("Unable to skip feedback. Please try again later.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm py-4">
        <div className="container mx-auto px-4">
          <h1 className="text-2xl font-bold text-gray-800">
            Hate Speech Detector
          </h1>
          <p className="text-sm text-gray-600">
            Analyze text for potential hate speech content
          </p>
        </div>
      </header>
      <main className="container mx-auto px-4 py-8 max-w-5xl">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="w-full lg:w-1/2">
            <InputPanel 
              inputText={inputText} 
              onTextChange={handleTextChange} 
              onAnalyze={handleAnalyze} 
              isAnalyzing={isAnalyzing} 
              error={error} 
              onReset={handleReset} 
            />
          </div>
          <div className="w-full lg:w-1/2">
            <OutputPanel 
              prediction={prediction} 
              predictionResult={getPredictionResult(prediction)}
              isAnalyzing={isAnalyzing}
              onProvideFeedback={handleProvideFeedback}
              onSkipFeedback={handleSkipFeedback}
              feedbackSubmitted={feedbackSubmitted}
            />
          </div>
        </div>
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>No texts are stored—your privacy is protected.</p>
        </div>
      </main>
    </div>
  );
}
