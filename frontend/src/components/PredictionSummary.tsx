import React from 'react';
import { AlertTriangleIcon, CheckCircleIcon, AlertCircleIcon } from 'lucide-react';
export const PredictionSummary = ({
  isHateSpeech,
  confidenceScore
}) => {
  // Determine the appropriate styling and content based on the result
  let bgColor = 'bg-green-50';
  let textColor = 'text-green-800';
  let borderColor = 'border-green-200';
  let icon = <CheckCircleIcon className="h-6 w-6 text-green-500" />;
  let status = 'Not Hate Speech';
  if (isHateSpeech) {
    if (confidenceScore > 75) {
      bgColor = 'bg-red-50';
      textColor = 'text-red-800';
      borderColor = 'border-red-200';
      icon = <AlertCircleIcon className="h-6 w-6 text-red-500" />;
      status = 'Likely Hate Speech';
    } else {
      bgColor = 'bg-yellow-50';
      textColor = 'text-yellow-800';
      borderColor = 'border-yellow-200';
      icon = <AlertTriangleIcon className="h-6 w-6 text-yellow-500" />;
      status = 'Possibly Hate Speech';
    }
  }
  return <div className={`${bgColor} ${borderColor} border rounded-lg p-4`}>
      <div className="flex items-center">
        <div className="flex-shrink-0">{icon}</div>
        <div className="ml-3">
          <h3 className={`text-lg font-medium ${textColor}`}>{status}</h3>
          <div className="mt-2">
            <p className={`text-sm ${textColor}`}>
              Confidence: {confidenceScore}%
            </p>
          </div>
        </div>
      </div>
    </div>;
};