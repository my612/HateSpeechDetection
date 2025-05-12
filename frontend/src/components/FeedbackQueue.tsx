
import React, { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Check, X, AlertTriangle, Clock, RefreshCw } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { FeedbackQueueItem, FeedbackStats } from '@/types';
import { getNextFeedbackItem, getFeedbackStats, submitFeedback, skipFeedback } from '@/services/api';

const FeedbackQueue: React.FC = () => {
  const [currentItem, setCurrentItem] = useState<FeedbackQueueItem | null>(null);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  
  const loadNextItem = async () => {
    setIsLoading(true);
    setFeedbackSubmitted(false);
    
    try {
      const item = await getNextFeedbackItem();
      setCurrentItem(item);
      const newStats = await getFeedbackStats();
      setStats(newStats);
    } catch (error) {
      console.error('Error loading feedback item:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleProvideFeedback = async (isCorrect: boolean) => {
    if (!currentItem) return;
    
    setIsLoading(true);
    try {
      await submitFeedback(currentItem.prediction_id, isCorrect);
      setFeedbackSubmitted(true);
      
      // Update stats
      const newStats = await getFeedbackStats();
      setStats(newStats);
      
      // After a delay, load the next item
      setTimeout(() => {
        loadNextItem();
      }, 1500);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleSkipFeedback = async () => {
    if (!currentItem) return;
    
    setIsLoading(true);
    try {
      await skipFeedback(currentItem.prediction_id);
      loadNextItem();
    } catch (error) {
      console.error('Error skipping feedback:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  useEffect(() => {
    loadNextItem();
  }, []);
  
  const renderFeedbackStats = () => {
    if (!stats) return null;
    
    const progressPercentage = stats.total > 0 ? 
      Math.round((stats.completed / stats.total) * 100) : 0;
    
    return (
      <div className="mb-4">
        <div className="flex justify-between mb-1 text-sm text-gray-600">
          <span>Feedback Progress</span>
          <span>{stats.completed} of {stats.total} items</span>
        </div>
        <Progress value={progressPercentage} className="h-2" />
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>{stats.completed} completed</span>
          <span>{stats.skipped} skipped</span>
          <span>{stats.pending} pending</span>
        </div>
      </div>
    );
  };
  
  const renderQueueItem = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center p-8 text-gray-500">
          <Clock className="w-8 h-8 mb-2 animate-pulse" />
          <p>Loading next feedback item...</p>
        </div>
      );
    }
    
    if (!currentItem) {
      return (
        <div className="flex flex-col items-center justify-center p-8 text-gray-500">
          <Check className="w-8 h-8 mb-2 text-green-500" />
          <p>No pending feedback items!</p>
          <Button
            variant="outline"
            className="mt-4"
            onClick={loadNextItem}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Check Again
          </Button>
        </div>
      );
    }
    
    const truncatedText = currentItem.text.length > 200 
      ? currentItem.text.substring(0, 200) + '...' 
      : currentItem.text;
    
    return (
      <div className={`p-4 ${feedbackSubmitted ? 'opacity-50' : ''}`}>
        <div className="flex items-start gap-3 mb-3">
          <div className={`p-1.5 rounded-full ${currentItem.predicted_label ? 'bg-red-100' : 'bg-green-100'}`}>
            {currentItem.predicted_label ? (
              <AlertTriangle className="w-5 h-5 text-red-600" />
            ) : (
              <Check className="w-5 h-5 text-green-600" />
            )}
          </div>
          <div>
            <div className="font-medium">
              {currentItem.predicted_label ? 'Detected as Hate Speech' : 'Detected as Not Hate Speech'}
              <span className="ml-2 text-sm text-gray-500">
                ({Math.round(currentItem.probability * 100)}% confidence)
              </span>
            </div>
            <div className="text-xs text-gray-500">ID: {currentItem.prediction_id}</div>
          </div>
        </div>
        
        <div className="p-3 bg-gray-50 rounded-md mb-4 text-gray-800">
          {truncatedText}
        </div>
        
        {feedbackSubmitted ? (
          <div className="text-center text-green-600 font-medium">
            <Check className="inline-block w-5 h-5 mr-1" />
            Thank you for your feedback!
          </div>
        ) : (
          <div className="flex flex-col">
            <h3 className="text-sm font-medium text-gray-700 mb-2 text-center">Was this prediction correct?</h3>
            <div className="flex justify-center gap-3">
              <Button 
                variant="outline" 
                className="border-green-500 text-green-700 hover:bg-green-50"
                onClick={() => handleProvideFeedback(true)}
                disabled={isLoading}
              >
                <Check className="mr-2 h-4 w-4" />
                Yes
              </Button>
              <Button 
                variant="outline" 
                className="border-red-500 text-red-700 hover:bg-red-50"
                onClick={() => handleProvideFeedback(false)}
                disabled={isLoading}
              >
                <X className="mr-2 h-4 w-4" />
                No
              </Button>
              <Button 
                variant="ghost" 
                className="text-gray-500"
                onClick={handleSkipFeedback}
                disabled={isLoading}
              >
                Skip
              </Button>
            </div>
          </div>
        )}
      </div>
    );
  };
  
  return (
    <div className="animate-fade-in">
      {renderFeedbackStats()}
      <Card>
        {renderQueueItem()}
      </Card>
      <p className="mt-3 text-xs text-gray-500 text-center">
        Your feedback helps our model learn and improve over time. 
        Thank you for contributing to making this tool more accurate!
      </p>
    </div>
  );
};

export default FeedbackQueue;
