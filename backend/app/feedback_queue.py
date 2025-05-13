# feedback_queue.py
import time
import uuid
from collections import deque


class FeedbackQueue:
    def __init__(self, max_size=100):
        """
        Initialize a queue for collecting feedback on predictions.
        
        Args:
            max_size: Maximum number of items to keep in the queue
        """
        self.queue = deque(maxlen=max_size)
        self.seen_ids = set()  # To avoid duplicates
    
    def add_prediction(self, prediction_id, text, predicted_label, probability):
        """
        Add a new prediction to the feedback queue.
        
        Args:
            prediction_id: Unique ID for the prediction
            text: Original text that was classified
            predicted_label: Model's prediction (True/False)
            probability: Confidence score
        """
        # Avoid duplicates
        if prediction_id in self.seen_ids:
            return False
        
        # Add to queue
        self.queue.append({
            "prediction_id": prediction_id,
            "text": text,
            "predicted_label": predicted_label,
            "probability": probability,
            "timestamp": time.time(),
            "status": "pending"  # pending, skipped, or completed
        })
        self.seen_ids.add(prediction_id)
        return True
    
    def get_next_feedback_item(self):
        """
        Get the next prediction needing feedback.
        
        Returns:
            Next item or None if queue is empty
        """
        # Find the oldest pending item
        for item in self.queue:
            if item["status"] == "pending":
                return item
        return None
    
    def mark_feedback_complete(self, prediction_id, feedback_label):
        """
        Mark an item as having received feedback.
        
        Args:
            prediction_id: ID of the prediction
            feedback_label: The feedback provided (True/False)
            
        Returns:
            Success status
        """
        for item in self.queue:
            if item["prediction_id"] == prediction_id:
                item["status"] = "completed"
                item["feedback_label"] = feedback_label
                return True
        return False
    
    def mark_feedback_skipped(self, prediction_id):
        """
        Mark an item as skipped (user declined to provide feedback).
        
        Args:
            prediction_id: ID of the prediction
            
        Returns:
            Success status
        """
        for item in self.queue:
            if item["prediction_id"] == prediction_id:
                item["status"] = "skipped"
                return True
        return False
    
    def get_feedback_stats(self):
        """Get statistics about feedback collection."""
        total = len(self.queue)
        completed = sum(1 for item in self.queue if item["status"] == "completed")
        skipped = sum(1 for item in self.queue if item["status"] == "skipped")
        pending = total - completed - skipped
        
        return {
            "total": total,
            "completed": completed,
            "skipped": skipped,
            "pending": pending
        }
