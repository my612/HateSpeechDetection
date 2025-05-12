
// API Configuration
const config = {
  // Base URL for the API
  baseUrl: "http://localhost:8000",
  
  // API endpoints
  endpoints: {
    predict: "/predict",
    submitFeedback: "/feedback",
    skipFeedback: "/feedback/skip",
    nextFeedbackItem: "/feedback/next",
    feedbackStats: "/feedback/stats",
    health: "/health"
  },
  
  // Request timeouts (in milliseconds)
  timeouts: {
    predict: 10000,
    feedback: 5000
  }
};

export default config;
