import React from 'react';
import { PredictionSummary } from './PredictionSummary';
import { TextHighlighter } from './TextHighlighter';
import { AttributionBreakdown } from './AttributionBreakdown';
import { Loader2Icon } from 'lucide-react';
export const OutputPanel = ({
  analysisResult,
  isAnalyzing
}) => {
  if (isAnalyzing) {
    return <div className="bg-white p-6 rounded-lg shadow-md h-full flex flex-col items-center justify-center">
        <Loader2Icon className="h-12 w-12 text-blue-500 animate-spin mb-4" />
        <p className="text-gray-600">Analyzing text for hate speech...</p>
        <p className="text-sm text-gray-500 mt-2">
          This may take a few moments
        </p>
      </div>;
  }
  if (!analysisResult) {
    return <div className="bg-white p-6 rounded-lg shadow-md h-full flex flex-col items-center justify-center text-center">
        <div className="text-gray-400 mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h3 className="text-xl font-medium text-gray-700">
          No Analysis Results
        </h3>
        <p className="mt-2 text-gray-500">
          Enter text and click "Detect Hate Speech" to analyze.
        </p>
      </div>;
  }
  return <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-2">
          Analysis Results
        </h2>
        <p className="text-sm text-gray-600">
          Our system has analyzed your text and identified the following:
        </p>
      </div>
      <PredictionSummary isHateSpeech={analysisResult.isHateSpeech} confidenceScore={analysisResult.confidenceScore} />
      <div className="mt-6">
        <h3 className="text-lg font-medium text-gray-700 mb-2">
          Text Analysis
        </h3>
        <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
          <TextHighlighter text={analysisResult.text} attributions={analysisResult.wordAttributions} />
        </div>
      </div>
      <div className="mt-6">
        <AttributionBreakdown attributions={analysisResult.wordAttributions} />
      </div>
      <div className="mt-6 text-sm text-gray-500 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-700 mb-2">About This Analysis</h4>
        <p className="mb-2">
          This tool uses an AI model to detect potential hate speech. The
          highlighted words show which parts of the text contributed most to the
          detection.
        </p>
        <p>
          <strong>Disclaimer:</strong> AI models have limitations and may not
          always be accurate. This analysis should be used as a guide, not as
          definitive judgment.
        </p>
      </div>
    </div>;
};