
import React from 'react';
import { Info } from 'lucide-react';
import { 
  Popover, 
  PopoverContent, 
  PopoverTrigger
} from '@/components/ui/popover';

const HighlightLegend: React.FC = () => {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button className="flex items-center text-xs text-gray-500 hover:text-gray-700">
          <Info className="w-3 h-3 mr-1" />
          How to read highlights
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-80">
        <div className="space-y-3">
          <h3 className="font-semibold">Understanding Text Highlights</h3>
          <p className="text-sm text-gray-600">
            Words are highlighted based on their contribution to the hate speech prediction:
          </p>
          
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="flex items-center mb-1">
                <span className="w-4 h-4 bg-hatedetector-highlight-low/30 mr-2"></span>
                <span className="text-sm">Low Impact</span>
              </div>
              <div className="flex items-center mb-1">
                <span className="w-4 h-4 bg-hatedetector-highlight-medium/30 mr-2"></span>
                <span className="text-sm">Medium Impact</span>
              </div>
              <div className="flex items-center">
                <span className="w-4 h-4 bg-hatedetector-highlight-high/30 mr-2"></span>
                <span className="text-sm">High Impact</span>
              </div>
            </div>
            <div className="pl-2 border-l text-xs text-gray-600">
              <p>Hover over highlighted words to see detailed attribution scores.</p>
            </div>
          </div>
          
          <div className="text-xs text-gray-500">
            <h4 className="font-medium mb-1">About Attribution Scores</h4>
            <p>
              Each word has a raw attribution score based on its impact on the prediction.
              Enhanced scores use TF-IDF weighting to account for word frequency patterns.
            </p>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default HighlightLegend;
