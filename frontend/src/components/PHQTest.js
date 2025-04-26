import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import { phqService } from '../services/phqService';
import './PHQTest.css';

const PHQTest = () => {
  const [answers, setAnswers] = useState(Array(9).fill(0));
  const [score, setScore] = useState(null);
  const [message, setMessage] = useState("");

  const navigate = useNavigate();
  const username = localStorage.getItem('username');
  const userId = localStorage.getItem('userId');

  // Redirect to login if userId is missing
  useEffect(() => {
    if (!userId) {
      navigate('/');
    }
  }, [userId, navigate]);

  const questions = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
  ];

  const options = [
    { label: "Not at all", value: 0 },
    { label: "Several days", value: 1 },
    { label: "More than half the days", value: 2 },
    { label: "Nearly every day", value: 3 },
  ];

  const handleAnswerChange = (index, value) => {
    const updatedAnswers = [...answers];
    updatedAnswers[index] = value;
    setAnswers(updatedAnswers);
  };

  const calculateScore = () => {
    const totalScore = answers.reduce((acc, curr) => acc + curr, 0);
    setScore(totalScore);
    const feedbackMessage = phqService.calculateDepressionLevel(totalScore);
    setMessage(feedbackMessage);
    saveTestResult(totalScore);
  };

  const saveTestResult = async (totalScore) => {
    try {
      if (!username || !userId) {
        toast.error("User information not found. Please log in again.");
        navigate('/');
        return;
      }

      const response = await phqService.saveTestResult({
        score: totalScore
      });
      console.log('Test result saved:', response);
      toast.success("Test result saved successfully!");
    } catch (error) {
      console.error('Error saving test result:', error);
      
      // Handle session expiration
      if (error.message === 'Session expired. Please login again.') {
        toast.error('Your session has expired. Please login again.');
        navigate('/');
        return;
      }
      
      // Handle other errors
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message;
      toast.error(errorMessage);
      
      if (error.response?.status === 401) {
        navigate('/');
      }
    }
  };

  return (
    <div className="phq-test-container container mt-5">
      <ToastContainer />
      <div className="d-flex justify-content-between align-items-center mb-4">
        <button onClick={() => navigate('/dashboard')} className="back-button">
          ← Back to Dashboard
        </button>
        <h1>PHQ-9 Test</h1>
      </div>
      <p className="text-muted text-center mb-4">
        The PHQ-9 is a multipurpose instrument for screening, diagnosing, monitoring, and measuring the severity of depression.
      </p>

      <div className="phq-test-grid">
        {/* Questions Section */}
        <div className="questions-section">
          {questions.map((question, index) => (
            <div key={index} className="question mb-4 p-3 rounded bg-light">
              <p><strong>{index + 1}. {question}</strong></p>
              <div className="options d-flex flex-wrap">
                {options.map((option) => (
                  <label key={option.value} className="me-4">
                    <input
                      type="radio"
                      name={`question-${index}`}
                      value={option.value}
                      onChange={() => handleAnswerChange(index, option.value)}
                    />
                    <span className="ms-1">{option.label}</span>
                  </label>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Submission and Results Section */}
        <div className="submit-section">
          <button onClick={calculateScore} className="btn btn-primary mb-4 w-100">
            Submit and Calculate Score
          </button>

          {score !== null && (
            <div className="results p-3 rounded">
              <h2>Your Score: {score}</h2>
              <p className={score >= 15 ? "text-warning" : "text-success"}>{message}</p>
              {score >= 15 && (
                <div className="helplines alert alert-warning mt-3">
                  <h4>Helplines</h4>
                  <p>National Suicide Prevention Lifeline: 1926</p>
                  <p>For more available helplines go to  https://findahelpline.com/countries/lk</p>
                  <p>For immediate assistance, call 119 / 1926 or go to the nearest hospital.</p>
                </div>
              )}
            </div>
          )}

          {/* Link to Help Line */}
          <div className="help-link mt-4">
            <Link to="/help-line" className="btn btn-link">
              Need Help? Visit Our Help Line Page
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PHQTest;
