import React, { useEffect } from 'react';
import { Check, AlertTriangle, Info, X } from 'lucide-react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Button } from '@/components/ui/button';
import { AttributionToken, PredictionResponse, PredictionResult } from '@/types';

interface OutputPanelProps {
  prediction: PredictionResponse | null;
  predictionResult: PredictionResult;
  onProvideFeedback: (isCorrect: boolean) => void;
  onSkipFeedback: () => void;
  feedbackSubmitted: boolean;
}

const OutputPanel: React.FC<OutputPanelProps> = ({
  prediction,
  predictionResult,
  onProvideFeedback,
  onSkipFeedback,
  feedbackSubmitted
}) => {
  useEffect(() => {
    if (prediction) {
      console.log("Prediction data:", prediction);
      console.log("Highlighted segments:", prediction.highlighted_segments?.length || 0);
      console.log("Explanation tokens with highlight=true:", 
        prediction.explanation.filter(t => t.highlight).length);
    }
  }, [prediction]);

  if (!prediction) return null;
  
  const renderResultBanner = () => {
    switch (predictionResult) {
      case 'not-hate':
        return (
          <div className="flex items-center gap-2 bg-hatedetector-success/20 text-hatedetector-success p-3 rounded-md font-medium">
            <Check className="w-5 h-5" />
            <span>Not Hate Speech</span>
            <span className="ml-auto text-sm">Confidence: {Math.round((1 - prediction.probability) * 100)}%</span>
          </div>
        );
      case 'possible-hate':
        return (
          <div className="flex items-center gap-2 bg-hatedetector-warning/20 text-hatedetector-warning p-3 rounded-md font-medium">
            <AlertTriangle className="w-5 h-5" />
            <span>Possibly Hate Speech</span>
            <span className="ml-auto text-sm">Confidence: {Math.round(prediction.probability * 100)}%</span>
          </div>
        );
      case 'likely-hate':
        return (
          <div className="flex items-center gap-2 bg-hatedetector-error/20 text-hatedetector-error p-3 rounded-md font-medium">
            <X className="w-5 h-5" />
            <span>Likely Hate Speech</span>
            <span className="ml-auto text-sm">Confidence: {Math.round(prediction.probability * 100)}%</span>
          </div>
        );
      default:
        return null;
    }
  };
  
  const renderHighlightedText = () => {
    // Use position-based highlighting with highlighted_segments
    if (prediction.highlighted_segments && prediction.highlighted_segments.length > 0) {
      console.log("Using position-based highlighting");
      const segments = prediction.highlighted_segments;
      let result = [];
      let currentPosition = 0;
      
      // Process each highlighted segment
      for (const segment of segments) {
        // Add any text before this segment
        if (segment.start > currentPosition) {
          result.push(
            <span key={`text-${currentPosition}`}>
              {prediction.text.substring(currentPosition, segment.start)}
            </span>
          );
        }
        
        // Determine highlight color based on attribution score
        const score = segment.attribution;
        let bgColor;
        
        if (score > 0.7) {
          bgColor = "bg-hatedetector-highlight-high/30";
        } else if (score > 0.4) {
          bgColor = "bg-hatedetector-highlight-medium/30";
        } else {
          bgColor = "bg-hatedetector-highlight-low/30";
        }
        
        // Add the highlighted segment
        result.push(
          <TooltipProvider key={`highlight-${segment.start}`}>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className={`${bgColor} px-0.5 rounded`}>
                  {segment.original_text}
                </span>
              </TooltipTrigger>
              <TooltipContent>
                <div className="text-sm">
                  <p className="font-semibold">Attribution Score: {score.toFixed(2)}</p>
                  <p>Original Text: "{segment.original_text}"</p>
                  <p>Processed As: "{segment.processed_token}"</p>
                  <p className="text-xs text-gray-500 mt-1">
                    {score > 0.7 ? 'High' : score > 0.4 ? 'Medium' : 'Low'} contribution to prediction
                  </p>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
        
        currentPosition = segment.end;
      }
      
      // Add any remaining text
      if (currentPosition < prediction.text.length) {
        result.push(
          <span key={`text-${currentPosition}`}>
            {prediction.text.substring(currentPosition)}
          </span>
        );
      }
      
      return <div className="text-gray-800 leading-relaxed">{result}</div>;
    }
    
    // Fallback to token-based highlighting if position data isn't available
    console.log("Falling back to token-based highlighting");
    const words = prediction.text.split(/(\s+)/);
    const textWithHighlights = words.map((word, index) => {
      if (word.trim() === '') return word;
      
      // Find matching token (now checking for any token, not just ones with highlight=true)
      const token = prediction.explanation.find(t => t.token.toLowerCase() === word.toLowerCase());
      
      if (!token) {
        return <span key={index}>{word}</span>;
      }
      
      const score = token.enhanced_attribution;
      // Only highlight if score is significant
      if (score < 0.3) {
        return <span key={index}>{word}</span>;
      }
      
      let bgColor;
      if (score > 0.7) {
        bgColor = "bg-hatedetector-highlight-high/30";
      } else if (score > 0.4) {
        bgColor = "bg-hatedetector-highlight-medium/30";
      } else {
        bgColor = "bg-hatedetector-highlight-low/30";
      }
      
      return (
        <TooltipProvider key={index}>
          <Tooltip>
            <TooltipTrigger asChild>
              <span className={`${bgColor} px-0.5 rounded`}>{word}</span>
            </TooltipTrigger>
            <TooltipContent>
              <div className="text-sm">
                <p className="font-semibold">Attribution Score: {score.toFixed(2)}</p>
                <p>Raw Score: {token.raw_attribution.toFixed(2)}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {score > 0.7 ? 'High' : score > 0.4 ? 'Medium' : 'Low'} contribution to prediction
                </p>
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    });
    
    return <div className="text-gray-800 leading-relaxed">{textWithHighlights}</div>;
  };
  
  const renderTopContributors = () => {
    // Use highlighted_segments if available, otherwise use explanation
    if (prediction.highlighted_segments && prediction.highlighted_segments.length > 0) {
      console.log("Using highlighted_segments for top contributors");
      // Sort by attribution score
      const sortedSegments = [...prediction.highlighted_segments]
        .sort((a, b) => b.attribution - a.attribution)
        .slice(0, 10);
      
      return (
        <div className="mt-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Top Contributing Factors:</h3>
          <div className="grid grid-cols-1 gap-2">
            {sortedSegments.map((segment, index) => (
              <div key={index} className="flex items-center">
                <span className="text-sm mr-2">{segment.original_text}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-2">
                  <div 
                    className={`h-full rounded-full ${
                      segment.attribution > 0.7 
                        ? "bg-hatedetector-highlight-high" 
                        : segment.attribution > 0.4 
                          ? "bg-hatedetector-highlight-medium"
                          : "bg-hatedetector-highlight-low"
                    }`}
                    style={{ width: `${Math.min(100, segment.attribution * 100)}%` }}
                  ></div>
                </div>
                <span className="text-xs ml-2 w-12 text-right">{(segment.attribution * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      );
    } else {
      console.log("Using explanation for top contributors");
      // Instead of checking for highlight:true, filter by significant scores
      const sortedTokens = [...prediction.explanation]
        .filter(token => token.enhanced_attribution > 0.3) // Filter by score instead of highlight flag
        .sort((a, b) => b.enhanced_attribution - a.enhanced_attribution)
        .slice(0, 10);
      
      if (sortedTokens.length === 0) {
        return (
          <div className="text-gray-500 italic mt-2">
            No significant contributing factors detected.
          </div>
        );
      }
      
      return (
        <div className="mt-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Top Contributing Factors:</h3>
          <div className="grid grid-cols-1 gap-2">
            {sortedTokens.map((token, index) => (
              <div key={index} className="flex items-center">
                <span className="text-sm mr-2">{token.token}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-2">
                  <div 
                    className={`h-full rounded-full ${
                      token.enhanced_attribution > 0.7 
                        ? "bg-hatedetector-highlight-high" 
                        : token.enhanced_attribution > 0.4 
                          ? "bg-hatedetector-highlight-medium"
                          : "bg-hatedetector-highlight-low"
                    }`}
                    style={{ width: `${Math.min(100, token.enhanced_attribution * 100)}%` }}
                  ></div>
                </div>
                <span className="text-xs ml-2 w-12 text-right">{(token.enhanced_attribution * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      );
    }
  };
  
  const renderFeedbackSection = () => {
    if (feedbackSubmitted) {
      return (
        <div className="mt-4 p-3 bg-green-50 text-green-700 rounded-md text-center animate-fade-in">
          <Check className="inline-block w-5 h-5 mr-2" />
          Thank you for your feedback! It helps improve our model.
        </div>
      );
    }
    
    return (
      <div className="mt-4 border-t pt-4">
        <div className="flex flex-col">
          <h3 className="text-sm font-semibold text-center mb-3">Was this prediction correct?</h3>
          <div className="flex justify-center gap-3">
            <Button 
              variant="outline" 
              className="border-green-500 text-green-700 hover:bg-green-50"
              onClick={() => onProvideFeedback(true)}
            >
              <Check className="mr-2 h-4 w-4" />
              Yes
            </Button>
            <Button 
              variant="outline" 
              className="border-red-500 text-red-700 hover:bg-red-50"
              onClick={() => onProvideFeedback(false)}
            >
              <X className="mr-2 h-4 w-4" />
              No
            </Button>
            <Button 
              variant="ghost" 
              className="text-gray-500"
              onClick={onSkipFeedback}
            >
              Skip
            </Button>
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <Card className="w-full animate-fade-in">
      <CardHeader className="px-4 py-3">
        {renderResultBanner()}
        <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
          <div className="flex items-center">
            <Info className="w-3 h-3 mr-1" />
            <span>ID: {prediction.prediction_id}</span>
          </div>
          <div>
            Analysis completed {new Date().toLocaleTimeString()}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
            Highlighted Text
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="w-4 h-4 ml-1 text-gray-400" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-xs max-w-xs">
                    Highlighted words show their contribution to the hate speech prediction.
                    Colors range from yellow (low) to red (high impact).
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </h3>
          <div className="bg-gray-50 p-3 rounded-md border border-gray-100">
            {renderHighlightedText()}
          </div>
        </div>
        
        {renderTopContributors()}
        
        <div className="bg-blue-50 p-3 rounded-md text-sm text-gray-700">
          <p className="font-semibold text-blue-800">About this analysis:</p>
          <p className="mt-1">
            This tool uses machine learning to identify potential hate speech. 
            The highlighted words show which parts of the text contributed most to the prediction.
          </p>
          <p className="mt-1 text-xs text-gray-500">
            Note: This is an automated analysis and may not always be accurate.
            Your feedback helps improve the model.
          </p>
        </div>
        
        {renderFeedbackSection()}
      </CardContent>
    </Card>
  );
};

export default OutputPanel;
