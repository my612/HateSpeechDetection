
import { PredictionResponse, FeedbackResponse, FeedbackQueueItem, FeedbackStats } from "../types";
import config from "../config/api.config";

const API_BASE_URL = config.baseUrl;

// Prediction API
export const predictText = async (text: string): Promise<PredictionResponse> => {
  console.log("Analyzing text:", text);
  
  try {
    const response = await fetch(`${API_BASE_URL}${config.endpoints.predict}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new Error(errorData?.detail || `API error: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error predicting text:", error);
    throw error;
  }
};

// Feedback submission API
export const submitFeedback = async (
  predictionId: string,
  isCorrect: boolean
): Promise<FeedbackResponse> => {
  console.log(`Submitting feedback for prediction ${predictionId}: ${isCorrect ? 'correct' : 'incorrect'}`);
  
  try {
    const response = await fetch(`${API_BASE_URL}${config.endpoints.submitFeedback}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        prediction_id: predictionId,
        true_label: isCorrect ? 1 : 0
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new Error(errorData?.detail || `API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error submitting feedback:", error);
    throw error;
  }
};

export const skipFeedback = async (
  predictionId: string
): Promise<FeedbackResponse> => {
  console.log(`Skipping feedback for prediction ${predictionId}`);
  
  try {
    const response = await fetch(`${API_BASE_URL}${config.endpoints.skipFeedback}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prediction_id: predictionId }),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new Error(errorData?.detail || `API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error skipping feedback:", error);
    throw error;
  }
};
// Get next feedback item API
export const getNextFeedbackItem = async (): Promise<FeedbackQueueItem | null> => {
  console.log("Requesting next feedback item");
  
  try {
    const response = await fetch(`${API_BASE_URL}${config.endpoints.nextFeedbackItem}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new Error(errorData?.detail || `API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.status === "empty") {
      return null;
    }
    
    return {
      prediction_id: data.prediction_id,
      text: data.text,
      predicted_label: data.predicted_label,
      probability: data.probability
    };
  } catch (error) {
    console.error("Error getting next feedback item:", error);
    throw error;
  }
};

// Get feedback statistics API
export const getFeedbackStats = async (): Promise<FeedbackStats> => {
  console.log("Requesting feedback stats");
  
  try {
    const response = await fetch(`${API_BASE_URL}${config.endpoints.feedbackStats}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new Error(errorData?.detail || `API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error getting feedback stats:", error);
    throw error;
  }
};

// Check API health
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}${config.endpoints.health}`);
    return response.ok;
  } catch (error) {
    console.error("API health check failed:", error);
    return false;
  }
};
