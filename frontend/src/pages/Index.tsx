
import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { predictText, submitFeedback, skipFeedback } from '@/services/api';
import InputPanel from '@/components/InputPanel';
import OutputPanel from '@/components/OutputPanel';
import FeedbackQueue from '@/components/FeedbackQueue';
import ExportResults from '@/components/ExportResults';
import HighlightLegend from '@/components/HighlightLegend';
import ApiStatusIndicator from '@/components/ApiStatusIndicator';
import { PredictionResponse, PredictionResult } from '@/types';
import { useToast } from '@/hooks/use-toast';
import { RefreshCw, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Index = () => {
  const [activeTab, setActiveTab] = useState<string>('analyze');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult>(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();
  
  const handleAnalyze = async (text: string) => {
    setIsAnalyzing(true);
    setError(null);
    setFeedbackSubmitted(false);
    
    try {
      const result = await predictText(text);
      setPrediction(result);
      
      // Determine the prediction result type based on probability
      if (result.label) { // If classified as hate speech
        if (result.probability > 0.7) {
          setPredictionResult('likely-hate');
        } else {
          setPredictionResult('possible-hate');
        }
      } else {
        setPredictionResult('not-hate');
      }
      
    } catch (error) {
      console.error('Error analyzing text:', error);
      setError('Failed to analyze the text. Please try again.');
      toast({
        title: 'Error',
        description: 'Failed to analyze the text. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const handleReset = () => {
    setPrediction(null);
    setPredictionResult(null);
    setFeedbackSubmitted(false);
    setError(null);
  };
  
  const handleProvideFeedback = async (isCorrect: boolean) => {
    if (!prediction) return;
    
    try {
      await submitFeedback(prediction.prediction_id, isCorrect);
      setFeedbackSubmitted(true);
      toast({
        title: 'Feedback Submitted',
        description: 'Thank you for helping improve our model!',
      });
    } catch (error) {
      console.error('Error submitting feedback:', error);
      toast({
        title: 'Error',
        description: 'Failed to submit feedback. Please try again.',
        variant: 'destructive',
      });
    }
  };
  
  const handleSkipFeedback = async () => {
    if (!prediction) return;
    
    try {
      await skipFeedback(prediction.prediction_id);
      setFeedbackSubmitted(true);
      toast({
        description: 'Feedback skipped',
      });
    } catch (error) {
      console.error('Error skipping feedback:', error);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-5xl mx-auto px-4 py-6">
        <header className="mb-6 text-center">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Hate Speech Detector</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Analyze text for potential hate speech with AI-powered detection and explanations.
            Your feedback helps our system continue to learn and improve.
          </p>
          <div className="mt-2">
            <ApiStatusIndicator />
          </div>
        </header>
        
        <Tabs defaultValue="analyze" value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid grid-cols-2 mb-4">
            <TabsTrigger value="analyze">Analyze Text</TabsTrigger>
            <TabsTrigger value="feedback-queue">Feedback Queue</TabsTrigger>
          </TabsList>
          
          <TabsContent value="analyze" className="space-y-4">
            <InputPanel onSubmit={handleAnalyze} isLoading={isAnalyzing} />
            
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md flex items-start">
                <AlertTriangle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium">Analysis Error</p>
                  <p className="text-sm">{error}</p>
                </div>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="ml-auto"
                  onClick={() => setError(null)}
                >
                  <RefreshCw className="w-4 h-4 mr-1" /> Try Again
                </Button>
              </div>
            )}
            
            {prediction && predictionResult && (
              <div className="space-y-4">
                <div className="flex flex-wrap justify-between items-center">
                  <HighlightLegend />
                  {prediction && <ExportResults prediction={prediction} />}
                </div>
                
                <OutputPanel
                  prediction={prediction}
                  predictionResult={predictionResult}
                  onProvideFeedback={handleProvideFeedback}
                  onSkipFeedback={handleSkipFeedback}
                  feedbackSubmitted={feedbackSubmitted}
                />
                
                <div className="flex justify-center">
                  <Button
                    variant="ghost"
                    onClick={handleReset}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    New Analysis
                  </Button>
                </div>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="feedback-queue">
            <FeedbackQueue />
          </TabsContent>
        </Tabs>
        
        <footer className="mt-8 pt-4 border-t border-gray-200 text-center text-xs text-gray-500">
          <p className="mb-1">Your text is analyzed privately and not permanently stored.</p>
          <p>With your permission, anonymized feedback helps improve the model.</p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
