import React from 'react';
import { Loader2Icon, RefreshCwIcon } from 'lucide-react';
export const InputPanel = ({
  inputText,
  onTextChange,
  onAnalyze,
  isAnalyzing,
  error,
  onReset
}) => {
  const characterCount = inputText.length;
  const maxCharacters = 10000;
  const isOverLimit = characterCount > maxCharacters;
  const isDisabled = !inputText.trim() || isAnalyzing || isOverLimit;
  return <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-gray-800 mb-2">Input Text</h2>
        <p className="text-sm text-gray-600">
          Enter or paste text to analyze for potential hate speech.
        </p>
      </div>
      <div className="relative">
        <textarea className={`w-full h-64 p-4 border ${error ? 'border-red-300' : 'border-gray-300'} rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none`} placeholder="Enter your text here..." value={inputText} onChange={e => onTextChange(e.target.value)} disabled={isAnalyzing}></textarea>
        <div className="absolute bottom-3 right-3 text-sm text-gray-500">
          <span className={isOverLimit ? 'text-red-500' : ''}>
            {characterCount}
          </span>
          /{maxCharacters}
        </div>
      </div>
      {error && <div className="mt-2 text-red-600 text-sm">{error}</div>}
      {isOverLimit && <div className="mt-2 text-red-600 text-sm">
          Text exceeds the maximum character limit.
        </div>}
      <div className="mt-4 flex gap-3">
        <button className={`px-4 py-2 rounded-lg flex items-center gap-2 ${isDisabled ? 'bg-gray-300 text-gray-500 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`} onClick={onAnalyze} disabled={isDisabled}>
          {isAnalyzing ? <>
              <Loader2Icon className="h-4 w-4 animate-spin" />
              Analyzing...
            </> : 'Detect Hate Speech'}
        </button>
        {inputText && <button className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-100 flex items-center gap-2" onClick={onReset} disabled={isAnalyzing}>
            <RefreshCwIcon className="h-4 w-4" />
            Clear
          </button>}
      </div>
      <div className="mt-6">
        <button className="text-sm text-blue-600 hover:underline" onClick={() => onTextChange('This is an example text with some potentially problematic language. Some people are terrible and I hate them all. This kind of language might be flagged by our system.')} disabled={isAnalyzing}>
          Load example text
        </button>
      </div>
    </div>;
};