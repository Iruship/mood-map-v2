import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import bodyLanguageService from '../services/bodyLanguageService';
import './BodyLanguageTraining.css';

const BodyLanguageTraining = () => {
  const navigate = useNavigate();
  const [isRecording, setIsRecording] = useState(false);
  const [className, setClassName] = useState('');
  const [trainingStatus, setTrainingStatus] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [availableClasses, setAvailableClasses] = useState([]);
  const [captureCount, setCaptureCount] = useState(0);
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [accuracy, setAccuracy] = useState(null);
  const [isCapturing, setIsCapturing] = useState(true);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [trainingTips, setTrainingTips] = useState([
    "Hold each pose steady for better detection",
    "Include slight variations of the same pose",
    "Train with different lighting conditions",
    "Maintain good distance from camera (show full upper body)",
    "Collect at least 15-20 samples per class for better accuracy"
  ]);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  
  useEffect(() => {
    setupCamera();
    fetchAvailableClasses();
    
    return () => {
      // Cleanup camera on unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
        });
      }
    };
  }, []);
  
  const setupCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
      
      setIsCapturing(true);
    } catch (err) {
      console.error("Error accessing camera:", err);
      setIsCapturing(false);
      setFeedbackMessage("Error accessing camera. Please make sure your camera is connected and permissions are granted.");
    }
  };
  
  const fetchAvailableClasses = async () => {
    try {
      const response = await bodyLanguageService.getAvailableClasses();
      if (response.success) {
        setAvailableClasses(response.classes);
      }
    } catch (error) {
      console.error("Error fetching available classes:", error);
    }
  };
  
  const startRecording = () => {
    if (!className.trim()) {
      setFeedbackMessage("Please enter a class name first.");
      return;
    }
    
    setIsRecording(true);
    setFeedbackMessage(`Recording samples for class "${className}". Make the body pose/gesture and click Capture Sample.`);
    setCaptureCount(0);
  };
  
  const stopRecording = () => {
    setIsRecording(false);
    setFeedbackMessage(`Recorded ${captureCount} samples for class "${className}".`);
    
    // Refresh the page to ensure available classes are updated
    window.location.reload();
  };
  
  const captureSample = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    try {
      // Draw the current video frame on the canvas
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      
      // Convert canvas to blob
      const blob = await new Promise(resolve => {
        canvas.toBlob(resolve, 'image/jpeg');
      });
      
      // Send to API
      const response = await bodyLanguageService.recordTrainingData(blob, className);
      
      if (response.success) {
        setCaptureCount(prevCount => prevCount + 1);
        setFeedbackMessage(`Captured sample ${captureCount + 1} for class "${className}"`);
      } else {
        setFeedbackMessage(`Failed to capture sample: ${response.message || 'No face or body detected'}`);
      }
    } catch (error) {
      console.error("Error capturing sample:", error);
      setFeedbackMessage(`Error capturing sample: ${error.message}`);
    }
  };
  
  const trainModel = async () => {
    setIsTraining(true);
    setTrainingStatus('Training model...');
    
    try {
      const response = await bodyLanguageService.trainModel();
      
      if (response.success) {
        setTrainingComplete(true);
        setAccuracy(response.accuracy);
        setTrainingStatus(`Training complete! Model accuracy: ${(response.accuracy * 100).toFixed(2)}%`);
        
        // Refresh available classes
        fetchAvailableClasses();
      } else {
        setTrainingStatus(`Training failed: ${response.error}`);
      }
    } catch (error) {
      console.error("Error training model:", error);
      setTrainingStatus(`Error training model: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };
  
  const downloadModel = () => {
    bodyLanguageService.downloadModel();
  };
  
  const deleteAllData = async () => {
    if (!showDeleteConfirm) {
      setShowDeleteConfirm(true);
      return;
    }
    
    setIsDeleting(true);
    try {
      const response = await bodyLanguageService.deleteAllData();
      
      if (response.success) {
        setFeedbackMessage("All training data and models deleted successfully");
        setTrainingComplete(false);
        setAccuracy(null);
        setTrainingStatus('');
        setAvailableClasses([]);
      } else {
        setFeedbackMessage(`Failed to delete data: ${response.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error("Error deleting data:", error);
      setFeedbackMessage(`Error deleting data: ${error.message}`);
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
      // Refresh the page to show updated state
      window.location.reload();
    }
  };
  
  const cancelDelete = () => {
    setShowDeleteConfirm(false);
  };
  
  return (
    <div className="body-language-training">
      <div className="header">
        <button className="back-button" onClick={() => navigate("/dashboard")}>
          ← Back to Dashboard
        </button>
        <h1 className="title">Body Language Training</h1>
      </div>
      
      <div className="container">
        <div className="training-section">
          <div className="camera-container">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline
              className={isCapturing ? "active" : "hidden"}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
          
          <div className="controls">
            <div className="class-input">
              <label htmlFor="class-name">Class Name:</label>
              <input 
                type="text" 
                id="class-name" 
                value={className} 
                onChange={(e) => setClassName(e.target.value)}
                disabled={isRecording}
                placeholder="e.g., Happy, Sad, Victory, etc."
              />
            </div>
            
            {!isRecording ? (
              <button 
                className="start-recording" 
                onClick={startRecording}
                disabled={!isCapturing || !className.trim()}
              >
                Start Training Class
              </button>
            ) : (
              <>
                <button 
                  className="capture-sample" 
                  onClick={captureSample}
                >
                  Capture Sample
                </button>
                <button 
                  className="stop-recording" 
                  onClick={stopRecording}
                >
                  Finish Class Training
                </button>
                <div className="sample-count">
                  Samples captured: {captureCount}
                </div>
              </>
            )}
          </div>
          
          {feedbackMessage && (
            <div className="feedback-message">
              {feedbackMessage}
            </div>
          )}
          
          {isRecording && (
            <div className="training-tips">
              <h4>Tips for better training:</h4>
              <ul>
                {trainingTips.map((tip, index) => (
                  <li key={index}>{tip}</li>
                ))}
              </ul>
              <p className="training-warning">
                <strong>Important:</strong> For best results, collect at least 15-20 samples per class and try to maintain consistent lighting conditions.
              </p>
            </div>
          )}
        </div>
        
        <div className="model-section">
          <h2>Available Classes</h2>
          {availableClasses.length > 0 ? (
            <div className="class-list">
              {availableClasses.map((cls, index) => (
                <div key={index} className="class-item">
                  {cls}
                </div>
              ))}
            </div>
          ) : (
            <p>No classes trained yet. Start by capturing samples for a class.</p>
          )}
          
          <div className="training-controls">
            <button 
              className="train-model"
              onClick={trainModel}
              disabled={isTraining || availableClasses.length === 0}
            >
              {isTraining ? 'Training...' : 'Train Model'}
            </button>
            
            {trainingStatus && (
              <div className="training-status">
                {trainingStatus}
              </div>
            )}
            
            {trainingComplete && (
              <button 
                className="download-model"
                onClick={downloadModel}
              >
                Download Trained Model
              </button>
            )}
            
            <div className="delete-section">
              {showDeleteConfirm ? (
                <div className="delete-confirm">
                  <p className="warning-text">Are you sure? This will delete ALL training data and models.</p>
                  <div className="confirm-buttons">
                    <button 
                      className="confirm-delete"
                      onClick={deleteAllData}
                      disabled={isDeleting}
                    >
                      {isDeleting ? 'Deleting...' : 'Yes, Delete Everything'}
                    </button>
                    <button 
                      className="cancel-delete"
                      onClick={cancelDelete}
                      disabled={isDeleting}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <button 
                  className="delete-data"
                  onClick={deleteAllData}
                  disabled={isDeleting}
                >
                  Delete All Training Data
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BodyLanguageTraining; 