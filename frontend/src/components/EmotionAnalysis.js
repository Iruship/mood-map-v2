import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./EmotionAnalysis.css";

const EmotionAnalysis = () => {
  const navigate = useNavigate();
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [emotion, setEmotion] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // Emotion information and advice
  const EMOTION_INFO = {
    "Happy": {
      description: "You are experiencing joy and positive emotions. Regular experiences of happiness can help protect against depression and build emotional resilience. Maintaining positive emotions is key to mental well-being.",
      advice: [
        "Share your happiness with others around you",
        "Use this positive energy for creative activities",
        "Take a moment to appreciate what makes you happy",
        "Document this moment in a gratitude journal",
        "Build on this positive state to strengthen your mental health"
      ]
    },
    "Sad": {
      description: "You appear to be feeling down or melancholic. While sadness is a normal emotion, persistent sadness can be a sign of depression. If these feelings persist for more than two weeks or interfere with daily life, consider speaking with a mental health professional.",
      advice: [
        "Reach out to friends or family for support",
        "Practice self-care activities you enjoy",
        "Consider going for a walk in nature",
        "Listen to uplifting music or watch a comedy",
        "Remember that it's okay to not be okay",
        "If feelings persist, consider talking to a counselor or therapist"
      ]
    },
    "Angry": {
      description: "You seem to be experiencing frustration or anger. Anger can sometimes mask underlying depression, and frequent irritability can be a sign of depression, especially in men. Understanding and managing anger is important for mental health.",
      advice: [
        "Take deep breaths to calm your mind",
        "Try counting to ten slowly",
        "Consider physical exercise to release tension",
        "Write down what's bothering you",
        "Take a break from the situation if possible",
        "Consider if there are deeper feelings beneath the anger"
      ]
    },
    "Fearful": {
      description: "You appear to be experiencing anxiety or fear. Anxiety often co-occurs with depression, and persistent fear can contribute to depressive symptoms. Managing anxiety is crucial for overall mental well-being.",
      advice: [
        "Practice grounding techniques (5-4-3-2-1 method)",
        "Focus on your breathing",
        "Remind yourself that you are safe",
        "Talk to someone you trust about your concerns",
        "Consider meditation or calming exercises",
        "If anxiety is frequent, consider professional support"
      ]
    },
    "Disgusted": {
      description: "You seem to be experiencing aversion or disgust. Strong feelings of disgust, especially towards oneself, can be associated with depression and negative self-image. Understanding these feelings is important for emotional health.",
      advice: [
        "Remove yourself from the triggering situation if possible",
        "Focus on pleasant sensory experiences",
        "Practice mindfulness to center yourself",
        "Consider what might be causing this reaction",
        "Challenge negative self-thoughts if present",
        "Practice self-compassion exercises"
      ]
    },
    "Neutral": {
      description: "You appear to be in a balanced emotional state. While emotional neutrality is normal, persistent emotional numbness or lack of feeling can sometimes be a sign of depression. It's important to maintain emotional awareness.",
      advice: [
        "Use this calm state for productive activities",
        "Practice mindfulness to maintain balance",
        "Consider setting goals or planning ahead",
        "Engage in activities you enjoy",
        "Monitor your emotional patterns over time",
        "Stay connected with others even when feeling neutral"
      ]
    },
    "Surprised": {
      description: "You seem to be experiencing unexpected emotions. Sudden emotional changes can be normal, but frequent emotional instability might be related to mood disorders including depression. Being aware of emotional patterns is helpful.",
      advice: [
        "Take a moment to process what surprised you",
        "Consider if this reveals something new about yourself",
        "Use this energy for creative thinking",
        "Share your experience with others if appropriate",
        "Notice if certain situations regularly trigger strong emotions",
        "Keep track of sudden mood changes"
      ]
    }
  };

  useEffect(() => {
    setupCamera();
    return cleanup;
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
    }
  };

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext("2d");
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0);
      
      const imageDataURL = canvasRef.current.toDataURL("image/jpeg");
      setCapturedImage(imageDataURL);
    }
  };

  const analyzeEmotion = async () => {
    if (!capturedImage) return;

    setIsAnalyzing(true);
    try {
      // Convert base64 to blob
      const response = await fetch(capturedImage);
      const blob = await response.blob();
      
      const formData = new FormData();
      formData.append("image", blob, "capture.jpg");

      const result = await fetch("http://127.0.0.1:5001/api/emotion-detection/detect", {
        method: "POST",
        body: formData,
        headers: {
          "Accept": "application/json",
        },
        mode: "cors"
      });

      const data = await result.json();
      
      if (data.success && data.faces.length > 0) {
        setEmotion(data.faces[0].emotion);
      } else {
        setEmotion(null);
        alert("No face detected. Please try again.");
      }
    } catch (err) {
      console.error("Error analyzing emotion:", err);
      alert("Error analyzing emotion. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const retake = async () => {
    // Clean up existing stream
    cleanup();
    
    // Reset states
    setCapturedImage(null);
    setEmotion(null);
    setIsCapturing(false);
    
    // Reinitialize camera
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
      console.error("Error restarting camera:", err);
      setIsCapturing(false);
    }
  };

  return (
    <div className="emotion-analysis">
      <div className="header">
        <button className="back-button" onClick={() => navigate("/dashboard")}>
          ← Back to Dashboard
        </button>
        <h1 className="heading">Emotion Analysis</h1>
      </div>

      <div className="container">
        <div className="camera-section">
          {!capturedImage ? (
            <>
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline
                className={isCapturing ? "active" : "hidden"}
              />
              <button 
                className="capture-button"
                onClick={captureImage}
                disabled={!isCapturing}
              >
                Capture Image
              </button>
            </>
          ) : (
            <>
              <img src={capturedImage} alt="Captured" className="captured-image" />
              <div className="button-group">
                <button className="analyze-button" onClick={analyzeEmotion} disabled={isAnalyzing}>
                  {isAnalyzing ? "Analyzing..." : "Analyze Emotion"}
                </button>
                <button className="retake-button" onClick={retake}>
                  Retake Photo
                </button>
              </div>
            </>
          )}
          <canvas ref={canvasRef} style={{ display: "none" }} />
        </div>

        {emotion && EMOTION_INFO[emotion] && (
          <div className="emotion-info">
            <h2>Your Emotion: {emotion}</h2>
            <p className="description">{EMOTION_INFO[emotion].description}</p>
            <div className="advice-section">
              <h3>Suggestions:</h3>
              <ul>
                {EMOTION_INFO[emotion].advice.map((tip, index) => (
                  <li key={index}>{tip}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmotionAnalysis; 