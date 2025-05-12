
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2 } from 'lucide-react';

interface InputPanelProps {
  onSubmit: (text: string) => void;
  isLoading: boolean;
}

const EXAMPLE_TEXTS = [
  "This is an example of neutral text that does not contain hate speech.",
  "I think that group is really stupid and shouldn't have any rights.",
  "I disagree with your opinion, but I respect your right to express it."
];

const InputPanel: React.FC<InputPanelProps> = ({ onSubmit, isLoading }) => {
  const [text, setText] = useState('');
  
  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
  };
  
  const handleSubmit = () => {
    if (text.trim().length > 0) {
      onSubmit(text);
    }
  };
  
  const loadExample = (index: number) => {
    setText(EXAMPLE_TEXTS[index]);
  };
  
  const characterCount = text.length;
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const MAX_CHARS = 10000;
  const isOverLimit = characterCount > MAX_CHARS;
  const isValidInput = text.trim().length > 10 && !isOverLimit;
  
  return (
    <div className="w-full bg-white rounded-lg shadow-sm p-4 border border-gray-200">
      <div className="mb-2 flex justify-between items-center">
        <h2 className="text-lg font-semibold text-gray-800">Enter Text to Analyze</h2>
        <div className="text-sm text-gray-500">
          {`${characterCount}/${MAX_CHARS} characters · ${wordCount} words`}
        </div>
      </div>
      
      <Textarea
        placeholder="Enter your text here..."
        value={text}
        onChange={handleTextChange}
        className="min-h-[200px] mb-3 resize-y"
        maxLength={MAX_CHARS}
      />
      
      <div className="flex flex-col sm:flex-row justify-between items-center gap-3">
        <div className="flex flex-wrap gap-2">
          <span className="text-sm text-gray-600">Examples:</span>
          {EXAMPLE_TEXTS.map((_, index) => (
            <button
              key={index}
              onClick={() => loadExample(index)}
              className="text-sm text-hatedetector-primary hover:underline"
            >
              Example {index + 1}
            </button>
          ))}
        </div>
        
        <Button
          onClick={handleSubmit}
          disabled={!isValidInput || isLoading}
          className="w-full sm:w-auto bg-hatedetector-primary hover:bg-hatedetector-primary/90"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> 
              Analyzing...
            </>
          ) : (
            'Detect Hate Speech'
          )}
        </Button>
      </div>
      
      {isOverLimit && (
        <p className="mt-2 text-sm text-hatedetector-error">
          Text exceeds the maximum character limit. Please shorten your text.
        </p>
      )}
      {text.trim().length > 0 && text.trim().length <= 10 && (
        <p className="mt-2 text-sm text-hatedetector-warning">
          Please enter at least 10 words for more accurate analysis.
        </p>
      )}
    </div>
  );
};

export default InputPanel;
