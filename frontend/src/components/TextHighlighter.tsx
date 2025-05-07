import React, { useState } from 'react';
export const TextHighlighter = ({
  text,
  attributions
}) => {
  const [hoveredWord, setHoveredWord] = useState(null);
  // This function splits the text and applies highlighting based on attributions
  const renderHighlightedText = () => {
    if (!text || !attributions || attributions.length === 0) {
      return <p>{text}</p>;
    }
    // Sort attributions by startIndex to process in order
    const sortedAttributions = [...attributions].sort((a, b) => a.startIndex - b.startIndex);
    const textParts = [];
    let lastIndex = 0;
    sortedAttributions.forEach((attribution, idx) => {
      // Add text before the current attribution
      if (attribution.startIndex > lastIndex) {
        textParts.push(<span key={`text-${idx}`}>
            {text.substring(lastIndex, attribution.startIndex)}
          </span>);
      }
      // Add the highlighted word/phrase
      const word = text.substring(attribution.startIndex, attribution.endIndex);
      const score = attribution.score * 100; // Convert to percentage
      // Determine highlight color based on score
      let bgColor = 'bg-transparent';
      let textColor = 'text-gray-800';
      if (score > 75) {
        bgColor = 'bg-red-200';
        textColor = 'text-red-900';
      } else if (score > 50) {
        bgColor = 'bg-orange-200';
        textColor = 'text-orange-900';
      } else if (score > 25) {
        bgColor = 'bg-yellow-200';
        textColor = 'text-yellow-900';
      } else if (score > 0) {
        bgColor = 'bg-yellow-100';
        textColor = 'text-yellow-800';
      }
      textParts.push(<span key={`highlight-${idx}`} className={`${bgColor} ${textColor} px-0.5 rounded cursor-pointer relative`} onMouseEnter={() => setHoveredWord(attribution)} onMouseLeave={() => setHoveredWord(null)}>
          {word}
          {hoveredWord === attribution && <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-48 p-2 bg-gray-800 text-white text-xs rounded shadow-lg z-10">
              <div className="text-center mb-1 font-medium">
                Contribution Score: {score.toFixed(1)}%
              </div>
              <div className="text-center text-gray-300">
                {score > 75 ? 'High' : score > 50 ? 'Medium' : 'Low'}{' '}
                contribution to hate speech prediction
              </div>
              <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-gray-800"></div>
            </div>}
        </span>);
      lastIndex = attribution.endIndex;
    });
    // Add any remaining text after the last attribution
    if (lastIndex < text.length) {
      textParts.push(<span key="text-end">{text.substring(lastIndex)}</span>);
    }
    return <p className="leading-relaxed">{textParts}</p>;
  };
  return <div className="relative">
      {renderHighlightedText()}
      <div className="mt-4 flex items-center">
        <div className="text-xs text-gray-500">Contribution Level:</div>
        <div className="ml-2 flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-yellow-100 rounded"></span>
          <span className="text-xs">Low</span>
          <span className="inline-block w-3 h-3 bg-yellow-200 rounded"></span>
          <span className="inline-block w-3 h-3 bg-orange-200 rounded"></span>
          <span className="text-xs">Medium</span>
          <span className="inline-block w-3 h-3 bg-red-200 rounded"></span>
          <span className="text-xs">High</span>
        </div>
      </div>
    </div>;
};