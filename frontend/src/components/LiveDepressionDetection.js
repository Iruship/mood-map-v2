import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  LineElement,
  PointElement
} from 'chart.js';
import { Bar, Doughnut, Line } from 'react-chartjs-2';
import './LiveDepressionDetection.css';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  LineElement,
  PointElement
);

const LiveDepressionDetection = () => {
  const navigate = useNavigate();
  const [depressionLevel, setDepressionLevel] = useState('Normal');
  const [depressionScore, setDepressionScore] = useState(0);
  const [analysisTime, setAnalysisTime] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [emotionHistory, setEmotionHistory] = useState([]);
  const [showSteps, setShowSteps] = useState(true);
  const [analysisSteps, setAnalysisSteps] = useState([
    { id: 1, text: 'Video Recorded Successfully', completed: false },
    { id: 2, text: 'Uploading Video Data', completed: false },
    { id: 3, text: 'Analyzing Facial Movements', completed: false },
    { id: 4, text: 'Identifying Emotions', completed: false },
    { id: 5, text: 'Processing Emotion Patterns', completed: false },
    { id: 6, text: 'Calculating Stability Metrics', completed: false },
    { id: 7, text: 'Analyzing Depression Indicators', completed: false },
    { id: 8, text: 'Generating Final Report', completed: false }
  ]);
  const [analysisReport, setAnalysisReport] = useState(null);
  
  // Emotion weights for depression scoring
  const EMOTION_WEIGHTS = {
    'Angry': 0.7,    // High contribution to depression
    'Disgusted': 0.6,
    'Fearful': 0.8,
    'Happy': -1.0,   // Reduces depression score
    'Neutral': 0.2,
    'Sad': 1.0,      // Highest contribution to depression
    'Surprised': 0.3
  };

  // Depression level thresholds
  const DEPRESSION_LEVELS = {
    NORMAL: { max: 0.3, label: 'Normal' },
    MILD: { max: 0.5, label: 'Mild Depression' },
    MODERATE: { max: 0.7, label: 'Moderate Depression' },
    SEVERE: { max: 1.0, label: 'Severe Depression' }
  };

  // Function to calculate depression level based on score
  const calculateDepressionLevel = (score) => {
    if (score <= DEPRESSION_LEVELS.NORMAL.max) return DEPRESSION_LEVELS.NORMAL.label;
    if (score <= DEPRESSION_LEVELS.MILD.max) return DEPRESSION_LEVELS.MILD.label;
    if (score <= DEPRESSION_LEVELS.MODERATE.max) return DEPRESSION_LEVELS.MODERATE.label;
    return DEPRESSION_LEVELS.SEVERE.label;
  };

  // Format seconds to mm:ss
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Start analysis animation
  const startAnalysisAnimation = () => {
    // Immediately mark first step as completed
    setAnalysisSteps(prev => 
      prev.map(step => 
        step.id === 1 
          ? { ...step, completed: true }
          : step
      )
    );

    // Then start the interval for remaining steps
    let step = 2;
    const interval = setInterval(() => {
      if (step <= analysisSteps.length) {
        setAnalysisSteps(prev => 
          prev.map(s => 
            s.id === step 
              ? { ...s, completed: true }
              : s
          )
        );
        step++;
      } else {
        clearInterval(interval);
        // Hide steps after all are complete
        setTimeout(() => {
          setAnalysisSteps([]);
        }, 1000);
      }
    }, 500);
  };

  // Add new analysis functions
  const analyzeEmotionVariance = (emotionHistory) => {
    if (!emotionHistory || emotionHistory.length === 0) {
      return [];
    }
    
    const emotionCounts = {};
    emotionHistory.forEach(entry => {
      emotionCounts[entry.emotion] = (emotionCounts[entry.emotion] || 0) + 1;
    });
    
    const totalEntries = emotionHistory.length;
    const variance = Object.entries(emotionCounts).map(([emotion, count]) => ({
      emotion,
      percentage: (count / totalEntries) * 100,
      count
    }));
    
    return variance.sort((a, b) => b.percentage - a.percentage);
  };

  const analyzeEmotionTransitions = (emotionHistory) => {
    if (!emotionHistory || emotionHistory.length < 2) {
      return [];
    }
    
    const transitions = [];
    for (let i = 1; i < emotionHistory.length; i++) {
      const prev = emotionHistory[i - 1].emotion;
      const current = emotionHistory[i].emotion;
      if (prev !== current) {
        transitions.push({ from: prev, to: current });
      }
    }
    return transitions;
  };

  const analyzeEmotionStability = (emotionHistory) => {
    if (!emotionHistory || emotionHistory.length < 2) {
      return [];
    }
    
    const emotionDurations = {};
    let currentEmotion = emotionHistory[0].emotion;
    let startTime = new Date(emotionHistory[0].timestamp);
    
    for (let i = 1; i < emotionHistory.length; i++) {
      if (emotionHistory[i].emotion !== currentEmotion) {
        const endTime = new Date(emotionHistory[i].timestamp);
        const duration = (endTime - startTime) / 1000; // in seconds
        
        if (!emotionDurations[currentEmotion]) {
          emotionDurations[currentEmotion] = { total: 0, count: 0 };
        }
        emotionDurations[currentEmotion].total += duration;
        emotionDurations[currentEmotion].count += 1;
        
        currentEmotion = emotionHistory[i].emotion;
        startTime = endTime;
      }
    }
    
    return Object.entries(emotionDurations).map(([emotion, data]) => ({
      emotion,
      averageDuration: data.total / data.count
    })).sort((a, b) => b.averageDuration - a.averageDuration);
  };

  const generateAnalysisReport = (emotionHistory) => {
    if (!emotionHistory || emotionHistory.length === 0) {
      return {
        dominantEmotion: { emotion: 'No Data', percentage: 0 },
        emotionDistribution: [],
        emotionStability: [],
        totalTransitions: 0,
        mostCommonTransition: null,
        recordingDuration: analysisTime,
        depressionLevel: depressionLevel,
        depressionScore: depressionScore
      };
    }
    
    const variance = analyzeEmotionVariance(emotionHistory);
    const transitions = analyzeEmotionTransitions(emotionHistory);
    const stability = analyzeEmotionStability(emotionHistory);
    
    // Calculate a more comprehensive depression score based on emotion distribution
    const calculatedDepressionScore = calculateWeightedDepressionScore(variance);
    // Use the calculated score if it's available, otherwise fall back to the current score
    const finalDepressionScore = calculatedDepressionScore !== null ? calculatedDepressionScore : depressionScore;
    // Calculate depression level based on the score
    const finalDepressionLevel = calculateDepressionLevel(finalDepressionScore);
    
    return {
      dominantEmotion: variance[0] || { emotion: 'No Data', percentage: 0 },
      emotionDistribution: variance,
      emotionStability: stability,
      totalTransitions: transitions.length,
      mostCommonTransition: findMostCommonTransition(transitions),
      recordingDuration: analysisTime,
      depressionLevel: finalDepressionLevel,
      depressionScore: finalDepressionScore
    };
  };

  // New function to calculate depression score from emotion distribution
  const calculateWeightedDepressionScore = (emotionDistribution) => {
    if (!emotionDistribution || emotionDistribution.length === 0) {
      return null;
    }
    
    let totalScore = 0;
    
    // Calculate weighted average based on emotion distribution
    emotionDistribution.forEach(item => {
      const weight = EMOTION_WEIGHTS[item.emotion] || 0;
      // Use percentage as weight for each emotion
      totalScore += weight * (item.percentage / 100);
    });
    
    // Normalize to 0-1 range (same as done in the original function)
    return Math.max(0, Math.min(1, (totalScore + 1) / 2));
  };

  const findMostCommonTransition = (transitions) => {
    const transitionCounts = {};
    transitions.forEach(transition => {
      const key = `${transition.from}→${transition.to}`;
      transitionCounts[key] = (transitionCounts[key] || 0) + 1;
    });
    
    const mostCommon = Object.entries(transitionCounts)
      .sort((a, b) => b[1] - a[1])[0];
    
    return mostCommon ? {
      transition: mostCommon[0],
      count: mostCommon[1]
    } : null;
  };

  // Modify the stop recording handler
  const handleStopRecording = () => {
    setIsRecording(false);
    setShowAnalysis(true);
    setShowSteps(true);
    setAnalysisReport(null);
    
    // Reset steps
    setAnalysisSteps(steps => steps.map(step => ({ ...step, completed: false })));

    // Stop video stream
    const videoEl = document.getElementById('video');
    if (videoEl && videoEl.srcObject) {
      const tracks = videoEl.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }

    // Mark first step as completed immediately
    setAnalysisSteps(steps => 
      steps.map(step => 
        step.id === 1
          ? { ...step, completed: true } 
          : step
      )
    );

    // Function to get random interval between 500ms and 1500ms
    const getRandomInterval = () => Math.floor(Math.random() * 1000) + 500;

    // Start loop from step 2 with random intervals
    let currentStep = 1;
    const processNextStep = () => {
      if (currentStep <= 8) {  // Updated to match new number of steps
        setAnalysisSteps(steps => 
          steps.map(step => 
            step.id === currentStep
              ? { ...step, completed: true } 
              : step
          )
        );
        currentStep++;
        // Schedule next step with random interval
        setTimeout(processNextStep, getRandomInterval());
      } else {
        // Show report after steps complete
        setShowSteps(false);
        // Wait 500ms before showing the report to ensure smooth transition
        setTimeout(() => {
          const report = generateAnalysisReport(emotionHistory);
          setAnalysisReport(report);
        }, 500);
      }
    };

    // Start the process after a delay
    setTimeout(() => {
      processNextStep();
    }, 1500); // Add 1.5s delay before starting the loop
  };

  // Add handler to close analysis popup
  const handleCloseAnalysis = () => {
    // Stop any video tracks first
    const videoEl = document.getElementById('video');
    if (videoEl && videoEl.srcObject) {
      const tracks = videoEl.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }

    // Reload the page to reset everything
    window.location.reload();
  };

  // Handle cleanup when navigating away
  const handleNavigateBack = useCallback(() => {
    // Stop all media tracks
    const videoEl = document.getElementById('video');
    if (videoEl && videoEl.srcObject) {
      const tracks = videoEl.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }

    // Clear any intervals
    if (window.analysisInterval) {
      clearInterval(window.analysisInterval);
      delete window.analysisInterval;
    }

    // Remove global functions
    delete window.updateDepressionScore;
    delete window.updateAnalysisTime;

    // Navigate back
    navigate('/dashboard');
  }, [navigate]);

  useEffect(() => {
    // Initialize the vanilla JS code
    const script = document.createElement('script');
    script.textContent = `
      (function() {
        // Get DOM elements
        const videoEl = document.getElementById('video');
        const canvasEl = document.getElementById('canvas');
        const previewCanvasEl = document.getElementById('previewCanvas');
        const previewContextEl = previewCanvasEl.getContext('2d');

        let isProcessing = false;  
        let lastProcessedTime = 0;
        const PROCESS_INTERVAL = 100;  
        let lastDetectedFaces = [];
        let startTime = Date.now();
        
        // Update analysis time every second
        window.analysisInterval = setInterval(() => {
          if (window.updateAnalysisTime) {
            const elapsedSeconds = (Date.now() - startTime) / 1000;
            window.updateAnalysisTime(elapsedSeconds);
          }
        }, 1000);

        // Emotion weights for depression scoring
        const EMOTION_WEIGHTS = {
          'Angry': 0.7,
          'Disgusted': 0.6,
          'Fearful': 0.8,
          'Happy': -1.0,
          'Neutral': 0.2,
          'Sad': 1.0,
          'Surprised': 0.3
        };

        // Calculate depression score based on detected emotions
        function calculateDepressionScore(emotions) {
          if (!emotions || emotions.length === 0) return 0;
          
          // Get the primary emotion (first detected emotion)
          const primaryEmotion = emotions[0].emotion;
          
          // Return the weight for that emotion
          const score = EMOTION_WEIGHTS[primaryEmotion] || 0;
          
          // Normalize to 0-1 range
          return Math.max(0, Math.min(1, (score + 1) / 2));
        }

        async function setupCamera() {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
              video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
              } 
            });
            videoEl.srcObject = stream;
            videoEl.addEventListener('loadeddata', () => {
              requestAnimationFrame(updatePreview);
              processVideoFrame();
            });
          } catch (err) {
            console.error('Error accessing camera:', err);
          }
        }

        function updatePreview() {
          previewCanvasEl.width = videoEl.videoWidth;
          previewCanvasEl.height = videoEl.videoHeight;
          previewContextEl.drawImage(videoEl, 0, 0);
          
          if (lastDetectedFaces.length > 0) {
            drawFaceOverlays(previewContextEl, lastDetectedFaces);
          }
          
          requestAnimationFrame(updatePreview);
        }

        function drawFaceOverlays(context, faces) {
          faces.forEach(face => {
            const { x, y, width: faceWidth, height: faceHeight } = face.face_location;
            
            context.strokeStyle = '#00ff00';
            context.lineWidth = 2;
            context.strokeRect(x, y, faceWidth, faceHeight);
            
            const label = face.emotion;
            
            context.fillStyle = 'rgba(0, 0, 0, 0.5)';
            context.fillRect(x, y - 25, context.measureText(label).width + 10, 25);
            
            context.fillStyle = '#00ff00';
            context.font = '16px Arial';
            context.fillText(label, x + 5, y - 5);
          });
        }

        async function processVideoFrame() {
          const currentTime = Date.now();
          
          if (!isProcessing && currentTime - lastProcessedTime >= PROCESS_INTERVAL) {
            isProcessing = true;
            lastProcessedTime = currentTime;

            canvasEl.width = videoEl.videoWidth;
            canvasEl.height = videoEl.videoHeight;
            const ctx = canvasEl.getContext('2d');
            ctx.drawImage(videoEl, 0, 0);
            
            canvasEl.toBlob(async (blob) => {
              const formData = new FormData();
              formData.append('image', blob, 'capture.jpg');

              try {
                const response = await fetch('http://127.0.0.1:5001/api/emotion-detection/detect', {
                  method: 'POST',
                  body: formData,
                  headers: {
                    'Accept': 'application/json',
                  },
                  mode: 'cors'
                });
                
                const data = await response.json();
                
                if (data.success && data.faces.length > 0) {
                  lastDetectedFaces = data.faces;
                  
                  // Calculate depression score
                  const currentDepressionScore = calculateDepressionScore(data.faces);
                  
                  // Store emotion data using the provided function
                  if (window.updateEmotionHistory) {
                    window.updateEmotionHistory({
                    timestamp: new Date().toISOString(),
                    emotion: data.faces[0].emotion,
                    depressionScore: currentDepressionScore
                  });
                  }
                  
                  // Update depression score
                  if (window.updateDepressionScore) {
                    window.updateDepressionScore(currentDepressionScore);
                  }
                }
              } catch (err) {
                console.error('Error processing frame:', err);
              } finally {
                isProcessing = false;
              }
            }, 'image/jpeg', 0.8); 
          }
          
          requestAnimationFrame(processVideoFrame);
        }

        setupCamera();
      })();
    `;
    document.body.appendChild(script);

    // Add global functions
    window.updateDepressionScore = (score) => {
      setDepressionScore(score);
      setDepressionLevel(calculateDepressionLevel(score));
    };

    window.updateAnalysisTime = (seconds) => {
      setAnalysisTime(seconds);
    };

    window.updateEmotionHistory = (entry) => {
      setEmotionHistory(prev => {
        const newHistory = [...prev, entry];
        // Keep only last 100 entries to prevent memory issues
        if (newHistory.length > 100) {
          return newHistory.slice(-100);
        }
        return newHistory;
      });
    };

    return () => {
      // Cleanup when component unmounts
      const videoEl = document.getElementById('video');
      if (videoEl && videoEl.srcObject) {
        const tracks = videoEl.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }

      if (window.analysisInterval) {
        clearInterval(window.analysisInterval);
        delete window.analysisInterval;
      }

      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }

      delete window.updateDepressionScore;
      delete window.updateAnalysisTime;
      delete window.updateEmotionHistory;
    };
  }, []);

  return (
    <div className="live-depression-detection">
      <div className="header">
        <button className="back-button" onClick={handleNavigateBack}>
          ← Back to Dashboard
        </button>
      <h1 className="heading">Live Depression Detection</h1>
      </div>
      
      <div className="container">
        <div className="camera-section">
          <div className="camera-container">
            <video id="video" autoPlay playsInline style={{ display: 'none' }}></video>
            <canvas id="previewCanvas"></canvas>
            <canvas id="canvas" style={{ display: 'none' }}></canvas>
          </div>
          
          {!isRecording && !showAnalysis ? (
            <button 
              className="start-recording-button"
              onClick={() => setIsRecording(true)}
            >
              Start Recording
            </button>
          ) : isRecording ? (
            <div className="recording-controls">
              <div className="recording-timer">{formatTime(analysisTime)}</div>
              <button 
                className="stop-recording-button"
                onClick={handleStopRecording}
              >
                Stop Recording
              </button>
            </div>
          ) : null}
        </div>
        
        {showAnalysis && (
          <div className="analysis-overlay">
            <div className="analysis-popup">
              <button className="close-analysis" onClick={handleCloseAnalysis}>×</button>
              
              {showSteps ? (
                <div className="analysis-steps">
                  {analysisSteps.map(step => (
                    <div key={step.id} className={`analysis-step ${step.completed ? 'completed' : ''}`}>
                      <div className="step-checkmark">{step.completed ? '✓' : ''}</div>
                      <div className="step-text">{step.text}</div>
                    </div>
                  ))}
                </div>
              ) : (
                analysisReport && (
                  <div className="analysis-report">
                    <h2 className="report-title">Emotional Analysis Report</h2>
                    
                    <div className="report-grid">
                      {/* Dominant Emotion Section */}
                      <div className="report-section">
                        <h3>Dominant Emotion</h3>
                        <div className="dominant-emotion">
                          <span className="emotion-label">{analysisReport.dominantEmotion.emotion}</span>
                          <span className="emotion-percentage">
                            {analysisReport.dominantEmotion.percentage.toFixed(1)}%
                          </span>
                        </div>
                        <div className="chart-container">
                          <Doughnut
                            data={{
                              labels: ['Dominant', 'Other'],
                              datasets: [{
                                data: [
                                  analysisReport.dominantEmotion.percentage,
                                  100 - analysisReport.dominantEmotion.percentage
                                ],
                                backgroundColor: ['#28a745', '#e9ecef'],
                                borderWidth: 0
                              }]
                            }}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: {
                                  display: false
                                }
                              }
                            }}
                          />
                        </div>
                      </div>

                      {/* Emotion Distribution Section */}
                      <div className="report-section">
                        <h3>Emotion Distribution</h3>
                        <div className="chart-container">
                          <Bar
                            data={{
                              labels: analysisReport.emotionDistribution.map(item => item.emotion),
                              datasets: [{
                                data: analysisReport.emotionDistribution.map(item => item.percentage),
                                backgroundColor: '#007bff',
                                borderRadius: 4
                              }]
                            }}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: {
                                  display: false
                                }
                              },
                              scales: {
                                y: {
                                  beginAtZero: true,
                                  max: 100
                                }
                              }
                            }}
                          />
                        </div>
                      </div>

                      {/* Emotion Stability Section */}
                      <div className="report-section">
                        <h3>Emotion Stability</h3>
                        <div className="emotion-stability">
                          {analysisReport.emotionStability.map((item, index) => (
                            <div key={index} className="stability-item">
                              <span className="emotion-label">{item.emotion}</span>
                              <span className="stability-duration">
                                {item.averageDuration.toFixed(1)}s average
                              </span>
                            </div>
                          ))}
                        </div>
                        <div className="chart-container">
                          <Bar
                            data={{
                              labels: analysisReport.emotionStability.map(item => item.emotion),
                              datasets: [{
                                data: analysisReport.emotionStability.map(item => item.averageDuration),
                                backgroundColor: '#007bff',
                                borderRadius: 4
                              }]
                            }}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: {
                                  display: false
                                }
                              },
                              scales: {
                                y: {
                                  beginAtZero: true,
                                  title: {
                                    display: true,
                                    text: 'Average Duration (seconds)'
                                  }
                                }
                              }
                            }}
                          />
                        </div>
                      </div>

                      {/* Depression Assessment Section */}
                      <div className="report-section">
                        <h3>Depression Assessment</h3>
                        <div className="depression-assessment">
                          <div className="depression-level">
                            <div>Level</div>
                            <div>{analysisReport.depressionLevel}</div>
                          </div>
                          <div className="depression-score">
                            <div>Score</div>
                            <div>{(analysisReport.depressionScore * 100).toFixed(1)}%</div>
                          </div>
                        </div>
                        <div className="chart-container">
                          <Doughnut
                            data={{
                              labels: ['Depression Risk', 'Normal'],
                              datasets: [{
                                data: [
                                  analysisReport.depressionScore * 100,
                                  100 - (analysisReport.depressionScore * 100)
                                ],
                                backgroundColor: ['#dc3545', '#e9ecef'],
                                borderWidth: 0
                              }]
                            }}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: {
                                  position: 'bottom'
                                }
                              }
                            }}
                          />
                        </div>
                      </div>

                      {/* Emotion Transitions Section */}
                      <div className="report-section" style={{ gridColumn: '1 / -1' }}>
                        <h3>Emotion Transitions</h3>
                        <div className="transition-stats">
                          <div className="transition-count">
                            <div>Total Transitions</div>
                            <div style={{ fontSize: '2rem', fontWeight: '600', marginTop: '0.5rem' }}>
                              {analysisReport.totalTransitions}
                            </div>
                          </div>
                          {analysisReport.mostCommonTransition && (
                            <div className="common-transition">
                              <div>Most Common Transition</div>
                              <div style={{ fontSize: '1.2rem', fontWeight: '500', marginTop: '0.5rem' }}>
                                {analysisReport.mostCommonTransition.transition}
                                <div style={{ fontSize: '1rem', color: '#666', marginTop: '0.25rem' }}>
                                  ({analysisReport.mostCommonTransition.count} times)
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveDepressionDetection;
