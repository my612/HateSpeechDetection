import React from 'react';
export const AttributionBreakdown = ({
  attributions
}) => {
  if (!attributions || attributions.length === 0) {
    return null;
  }
  // Sort attributions by score (highest to lowest)
  const sortedAttributions = [...attributions].sort((a, b) => b.score - a.score).slice(0, 5); // Show only top 5 contributing words/phrases
  return <div>
      <h3 className="text-lg font-medium text-gray-700 mb-3">
        Top Contributing Factors
      </h3>
      <div className="space-y-3">
        {sortedAttributions.map((attribution, index) => {
        const word = attribution.word || attribution.text;
        const score = attribution.score * 100; // Convert to percentage
        // Determine the bar color based on score
        let barColor = 'bg-yellow-100';
        if (score > 75) barColor = 'bg-red-500';else if (score > 50) barColor = 'bg-orange-500';else if (score > 25) barColor = 'bg-yellow-500';else barColor = 'bg-yellow-300';
        return <div key={index} className="flex items-center">
              <div className="w-1/3 pr-4">
                <div className="font-medium text-gray-700 truncate">{word}</div>
              </div>
              <div className="w-2/3">
                <div className="flex items-center">
                  <div className="flex-grow h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div className={`h-full ${barColor} rounded-full`} style={{
                  width: `${score}%`
                }}></div>
                  </div>
                  <div className="ml-3 text-sm text-gray-600 w-12 text-right">
                    {score.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>;
      })}
      </div>
    </div>;
};