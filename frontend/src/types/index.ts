
export interface AttributionToken {
  token: string;
  raw_attribution: number;
  enhanced_attribution: number;
  highlight: boolean;
}

export interface PredictionResponse {
  prediction_id: string;
  text: string;
  label: boolean;
  probability: number;
  tokens: string[];
  explanation: AttributionToken[];
}

export interface FeedbackResponse {
  status: string;
  message: string;
}

export interface FeedbackQueueItem {
  prediction_id: string;
  text: string;
  predicted_label: boolean;
  probability: number;
}

export interface FeedbackStats {
  total: number;
  completed: number;
  skipped: number;
  pending: number;
}

export type PredictionResult = "not-hate" | "possible-hate" | "likely-hate" | null;
