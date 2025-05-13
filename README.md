# Hate Speech Detection

A machine learning application that detects hate speech in text with explanations of which words contributed to the classification. Features online learning from user feedback, explainable AI, and a React-based UI.

## 📋 Overview

This project consists of:

- **Backend**: FastAPI-based Python service with a neural network model for hate speech detection
- **Frontend**: React TypeScript application with real-time analysis and visualization
- **Features**: Interactive UI, word contribution highlighting, explainable predictions, and continuous learning via user feedback

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Git LFS (for large model files)

### Clone the Repository

```bash
# Make sure Git LFS is installed first
git lfs install

# Clone the repository with LFS files
git clone https://github.com/my612/HateSpeechDetection.git
cd HateSpeechDetection
```

To setup backend:
```bash
# Navigate to backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The backend API will be available at !(http://localhost:8000)[]
To setup frontend:
```bash
# Open a new terminal window
# Navigate to frontend directory from project root
cd frontend

# Install dependencies
npm install
# or if you use yarn
# yarn install

# Start the development server
npm run dev
# or with yarn
# yarn dev
```

The frontend application will be available at http://localhost:3000
🧠 How It Works

    Text Analysis: Enter text in the input panel and click "Analyze"
    Word Highlighting: The system highlights words that contributed to the hate speech prediction
    Explainability: Hover over highlighted words to see their contribution scores
    Feedback: Provide feedback on predictions to help the model improve through online learning

🛠️ API Endpoints

    POST /predict: Analyze text for hate speech
    POST /feedback: Submit feedback on predictions
    POST /feedback/skip: Skip providing feedback
    GET /feedback/stats: Get statistics on collected feedback
    GET /health: Check API health

📊 Project Structure

HateSpeechDetection/

├── backend/

│   ├── app/

│   │   ├── main.py         # FastAPI application

│   │   ├── model.py        # Neural network model

│   │   └── ...

│   ├── artifacts/          # Model weights and vocabulary

│   └── requirements.txt    # Python dependencies

├── frontend/

│   ├── src/

│   │   ├── components/     # React components

│   │   ├── apis.tsx        # API integration

│   │   └── App.tsx         # Main application

│   ├── package.json        # Frontend dependencies

│   └── ...

└── README.md

🔒 Privacy

    No user data or analyzed text is stored permanently
    Feedback is used only to improve the model's accuracy
    All processing happens locally within the application

📝 License

MIT License
🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
📧 Contact

For questions or support, please open an issue on the GitHub repository.

Happy detecting! 🕵️‍♀️


