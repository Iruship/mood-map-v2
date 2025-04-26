import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import bodyLanguageService from '../services/bodyLanguageService';
import './BodyLanguageDetection.css';

// Helper function for stabilizing predictions
const stabilizePredictions = (prevResult, newResult, confidenceThreshold = 0.65) => {
  // If no previous prediction, use new one if confident enough
  if (!prevResult || !prevResult.body_language_class) {
    return newResult.confidence >= confidenceThreshold ? newResult : null;
  }
  
  // If new prediction matches previous with high confidence, use it
  if (newResult.body_language_class === prevResult.body_language_class && 
      newResult.confidence >= confidenceThreshold) {
    return newResult;
  }
  
  // If new prediction differs but has very high confidence, use it
  if (newResult.confidence >= 0.85) {
    return newResult;
  }
  
  // Otherwise stick with previous prediction to reduce flickering
  return prevResult;
};

const BodyLanguageDetection = () => {
  const navigate = useNavigate();
  const [isCapturing, setIsCapturing] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [detectionActive, setDetectionActive] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [error, setError] = useState(null);
  const [availableClasses, setAvailableClasses] = useState([]);
  const [showModelStatus, setShowModelStatus] = useState(false);
  
  // Add these new state variables:
  const [consecutiveFailures, setConsecutiveFailures] = useState(0);
  const [isModelReady, setIsModelReady] = useState(false);
  const [stabilizedResult, setStabilizedResult] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [currentModel, setCurrentModel] = useState('default');
  const [isChangingModel, setIsChangingModel] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  
  useEffect(() => {
    setupCamera();
    checkModelStatus();
    fetchAvailableModels();
    
    return () => {
      // Cleanup camera and detection interval on unmount
      stopRealTimeDetection();
      
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
      setError("Error accessing camera. Please make sure your camera is connected and permissions are granted.");
    }
  };
  
  const checkModelStatus = async () => {
    try {
      const response = await bodyLanguageService.getAvailableClasses();
      if (response.success) {
        setAvailableClasses(response.classes);
        
        if (response.classes.length === 0) {
          setShowModelStatus(true);
          setError("No trained model available. Please train a model first.");
        } else {
          setIsModelReady(true);
          setError(null);
        }
      }
    } catch (error) {
      console.error("Error checking model status:", error);
      setShowModelStatus(true);
      setError("Error checking model status. Please try again later.");
    }
  };
  
  const fetchAvailableModels = async () => {
    try {
      const response = await bodyLanguageService.getAvailableModels();
      if (response.success) {
        setAvailableModels(response.available_models);
        setCurrentModel(response.current_model);
      }
    } catch (error) {
      console.error("Error fetching available models:", error);
      setError("Error fetching available models. Please try again later.");
    }
  };
  
  const handleModelChange = async (modelType) => {
    if (modelType === currentModel) return;
    
    setIsChangingModel(true);
    setError(null);
    
    // Stop detection if active
    if (detectionActive) {
      stopRealTimeDetection();
    }
    
    try {
      const response = await bodyLanguageService.switchModel(modelType);
      if (response.success) {
        setCurrentModel(response.current_model);
        setDetectionResult(null);
        setStabilizedResult(null);
      } else {
        setError(response.error || "Failed to switch model. Please try again.");
      }
    } catch (error) {
      console.error("Error switching model:", error);
      setError("Error switching model. Please try again later.");
    } finally {
      setIsChangingModel(false);
    }
  };
  
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    const imageUrl = canvas.toDataURL('image/jpeg');
    setCapturedImage(imageUrl);
  };
  
  const analyzeBodyLanguage = async () => {
    if (!capturedImage) return;
    
    setIsAnalyzing(true);
    try {
      // Convert base64 to blob
      const response = await fetch(capturedImage);
      const blob = await response.blob();
      
      const result = await bodyLanguageService.detectBodyLanguage(blob);
      
      if (result.success) {
        setDetectionResult(result);
        setStabilizedResult(result); // For single image, use result directly
        setConsecutiveFailures(0);
      } else {
        setError(result.error || "Failed to detect body language. Please try again.");
      }
    } catch (err) {
      console.error("Error analyzing body language:", err);
      setError("Error analyzing image. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const retake = () => {
    setCapturedImage(null);
    setDetectionResult(null);
    setError(null);
  };
  
  const startRealTimeDetection = () => {
    if (!isModelReady) {
      setError("Model is not ready. Please check if you have trained the model properly.");
      return;
    }
    
    setDetectionActive(true);
    setError(null);
    
    // Start detection interval (every 1000ms for more reliability and less CPU usage)
    detectionIntervalRef.current = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;
      
      try {
        // Capture frame
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        
        // Convert to blob
        const blob = await new Promise(resolve => {
          canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });
        
        // Analyze frame
        const result = await bodyLanguageService.detectBodyLanguage(blob);
        
        if (result.success) {
          // Stabilize predictions to reduce flickering
          const stabilized = stabilizePredictions(stabilizedResult, result);
          if (stabilized) {
            setStabilizedResult(stabilized);
            setDetectionResult(stabilized);
          }
          setError(null);
          setConsecutiveFailures(0);
        } else {
          // Count consecutive failures
          setConsecutiveFailures(prev => prev + 1);
          
          // Only show error after multiple consecutive failures
          if (consecutiveFailures >= 3) {
            setError("Having trouble detecting body language. Please ensure you're visible in the frame.");
          }
        }
      } catch (err) {
        console.error("Error in real-time detection:", err);
        setConsecutiveFailures(prev => prev + 1);
        
        if (consecutiveFailures >= 5) {
          stopRealTimeDetection();
          setError("Too many errors during detection. Real-time detection has been stopped.");
        }
      }
    }, 1000);
  };
  
  const stopRealTimeDetection = () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    setDetectionActive(false);
  };
  
  const renderDetectionResult = () => {
    const result = detectionActive ? stabilizedResult : detectionResult;
    
    if (!result) return null;
    
    const { body_language_class, confidence } = result;
    
    return (
      <div className="detection-result">
        <div className="result-header">
          <h3>Detection Result</h3>
        </div>
        
        <div className="result-body">
          <div className="result-item">
            <span className="label">Body Language:</span>
            <span className="value">{body_language_class}</span>
          </div>
          
          <div className="result-item">
            <span className="label">Confidence:</span>
            <span className="value">{(confidence * 100).toFixed(2)}%</span>
          </div>
          
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ 
                width: `${confidence * 100}%`,
                backgroundColor: confidence > 0.7 ? '#27ae60' : confidence > 0.5 ? '#f39c12' : '#e74c3c'
              }}
            ></div>
          </div>
          
          {detectionActive && (
            <div className="detection-tips">
              <p>For better detection:</p>
              <ul>
                <li>Ensure good lighting on your face and body</li>
                <li>Make your pose clear and distinct</li>
                <li>Make sure your full upper body is visible</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };
  
  // Update the render function to show real-time prediction
  const renderVideoWithPrediction = () => {
    return (
      <>
        <div className="camera-container real-time">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline
            className="active"
          />
          <div className="realtime-badge">Real-time Detection Active</div>
          
          {detectionResult && (
            <div className="realtime-prediction">
              <span className="prediction-label">{detectionResult.body_language_class}</span>
              <span className="prediction-confidence">
                {(detectionResult.confidence * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
        
        <div className="realtime-controls">
          <button 
            className="stop-realtime-button"
            onClick={stopRealTimeDetection}
          >
            Stop Real-time Detection
          </button>
        </div>
      </>
    );
  };
  
  // Add this new component for model selector
  const renderModelSelector = () => {
    return (
      <div className="model-selector">
        <label htmlFor="model-select">Detection Model: </label>
        <select 
          id="model-select" 
          value={currentModel}
          onChange={(e) => handleModelChange(e.target.value)}
          disabled={isChangingModel || detectionActive}
        >
          {availableModels.map((model) => (
            <option key={model} value={model}>
              {model === 'default' ? 'Default Model' : 
               model === 'neural_network' ? 'Neural Network' : 
               'RNN Model'}
            </option>
          ))}
        </select>
        {isChangingModel && <span className="loading-indicator">Switching model...</span>}
      </div>
    );
  };
  
  // Return the component UI
  return (
    <div className="body-language-detection">
      <div className="header">
        <button className="back-button" onClick={() => navigate("/dashboard")}>
          ← Back to Dashboard
        </button>
        <h1 className="title">Body Language Detection</h1>
      </div>
      
      <div className="container">
        {/* Add the model selector here */}
        {renderModelSelector()}
        
        {showModelStatus && availableClasses.length === 0 ? (
          <div className="no-model-message">
            <p>{error}</p>
            <button 
              className="train-model-button"
              onClick={() => navigate("/body-language-training")}
            >
              Go to Training Page
            </button>
          </div>
        ) : (
          <div className="detection-container">
            <div className="video-section">
              {!capturedImage && !detectionActive ? (
                <>
                  <div className="camera-container">
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline
                      className={isCapturing ? "active" : "hidden"}
                    />
                  </div>
                  
                  <div className="capture-controls">
                    <button 
                      className="capture-button"
                      onClick={captureImage}
                      disabled={!isCapturing}
                    >
                      Capture Image
                    </button>
                    
                    <button 
                      className="realtime-button"
                      onClick={startRealTimeDetection}
                      disabled={!isCapturing}
                    >
                      Start Real-time Detection
                    </button>
                  </div>
                </>
              ) : detectionActive ? (
                renderVideoWithPrediction()
              ) : (
                <>
                  <div className="captured-container">
                    <img 
                      src={capturedImage} 
                      alt="Captured" 
                      className="captured-image" 
                    />
                  </div>
                  
                  <div className="analyze-controls">
                    <button 
                      className="analyze-button"
                      onClick={analyzeBodyLanguage}
                      disabled={isAnalyzing}
                    >
                      {isAnalyzing ? "Analyzing..." : "Analyze Body Language"}
                    </button>
                    
                    <button 
                      className="retake-button"
                      onClick={retake}
                      disabled={isAnalyzing}
                    >
                      Retake Photo
                    </button>
                  </div>
                </>
              )}
            </div>
            
            <div className="result-section">
              {error ? (
                <div className="error-message">{error}</div>
              ) : (
                renderDetectionResult()
              )}
              
              {availableClasses.length > 0 && (
                <div className="available-classes">
                  <h3>Available Classes</h3>
                  <div className="class-list">
                    {availableClasses.map((cls, index) => (
                      <div key={index} className="class-item">
                        {cls}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default BodyLanguageDetection; 