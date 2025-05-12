
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuSeparator,
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu';
import { Check, Copy, Download } from 'lucide-react';
import { PredictionResponse } from '@/types';
import { toast } from 'sonner';

interface ExportResultsProps {
  prediction: PredictionResponse;
}

const ExportResults: React.FC<ExportResultsProps> = ({ prediction }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopyToClipboard = () => {
    // Generate a simple text format of the results
    const resultText = `
Hate Speech Analysis Results:
---------------------------
Prediction ID: ${prediction.prediction_id}
Text: ${prediction.text}
Result: ${prediction.label ? 'Potentially contains hate speech' : 'Does not contain hate speech'}
Confidence: ${Math.round(prediction.probability * 100)}%

Top contributing factors:
${prediction.explanation
  .filter(token => token.highlight)
  .sort((a, b) => b.enhanced_attribution - a.enhanced_attribution)
  .slice(0, 5)
  .map(token => `- "${token.token}": ${Math.round(token.enhanced_attribution * 100)}% contribution`)
  .join('\n')}
    `.trim();
    
    navigator.clipboard.writeText(resultText).then(() => {
      setCopied(true);
      toast.success('Results copied to clipboard');
      
      setTimeout(() => {
        setCopied(false);
      }, 2000);
    });
  };
  
  const handleDownloadJSON = () => {
    // Create a JSON blob
    const jsonData = JSON.stringify(prediction, null, 2);
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    // Create a link and trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = `hate-speech-analysis-${prediction.prediction_id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    // Clean up
    URL.revokeObjectURL(url);
    toast.success('JSON file downloaded');
  };
  
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <Download className="w-4 h-4 mr-2" />
          Export Results
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={handleCopyToClipboard}>
          {copied ? (
            <>
              <Check className="w-4 h-4 mr-2" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="w-4 h-4 mr-2" />
              Copy to Clipboard
            </>
          )}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleDownloadJSON}>
          <Download className="w-4 h-4 mr-2" />
          Download JSON
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default ExportResults;
