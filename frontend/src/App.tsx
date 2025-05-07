import React, { useState } from 'react';
import { InputPanel } from './components/InputPanel';
import { OutputPanel } from './components/OutputPanel';
import { analyzeMockText } from './utils/mockAnalysisService';
export function App() {
  const [inputText, setInputText] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const handleTextChange = text => {
    setInputText(text);
    if (analysisResult) {
      setAnalysisResult(null);
    }
    if (error) {
      setError('');
    }
  };
  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }
    try {
      setIsAnalyzing(true);
      setError('');
      const result = await analyzeMockText(inputText);
      setAnalysisResult(result);
    } catch (err) {
      setError("We're having trouble analyzing your text right now. Please try again.");
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };
  const handleReset = () => {
    setInputText('');
    setAnalysisResult(null);
    setError('');
  };
  return <div className="min-h-screen bg-gray-50">
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
            <InputPanel inputText={inputText} onTextChange={handleTextChange} onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} error={error} onReset={handleReset} />
          </div>
          <div className="w-full lg:w-1/2">
            <OutputPanel analysisResult={analysisResult} isAnalyzing={isAnalyzing} />
          </div>
        </div>
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>No texts are stored—your privacy is protected.</p>
        </div>
      </main>
    </div>;
}