import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Onboarding.css';

const Onboarding = () => {
  const [username, setUsername] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    // Retrieve username from localStorage and set it to state
    const storedUsername = localStorage.getItem('username');
    if (storedUsername) {
      setUsername(storedUsername);
    }
  }, []);

  const requestPermissions = async () => {
    try {
      await navigator.mediaDevices.getUserMedia({ video: true });
      alert('Camera access granted');
      navigate('/live-depression-detection');
    } catch (error) {
      alert('Camera access denied');
    }
  };

  return (
    <div className="onboarding-container">
      {/* Left side: Welcome message, description, privacy policy, and warning */}
      <div className="onboarding-left">
        <h2>Hello {username}, Let's Begin Your Mental Well-being Assessment</h2>
        <p>This app provides insights into mental health using image analysis and self-assessment tools.</p>
        <div className="privacy-policy">
          <h3>Privacy Policy</h3>
          <ul>
            <li><strong>Data Collection:</strong> We collect user data, such as username and email, to create and manage accounts. With user permission, the app may access the camera for real-time analysis.</li>
            <li><strong>Data Usage:</strong> Collected data is used solely for research purposes and to provide insights through this application. Depression and emotional analysis data is processed locally and not stored on our servers unless explicitly specified.</li>
            <li><strong>Data Security:</strong> All user data is protected using industry-standard encryption methods. We take reasonable measures to secure all collected data against unauthorized access or misuse.</li>
            <li><strong>User Control:</strong> Users can withdraw permission for camera access at any time through their device settings. Users can delete their account and personal data by contacting our support team.</li>
            <li><strong>Third-Party Services:</strong> We do not share personal data with third parties for marketing or advertising purposes. External services used in the app comply with relevant data protection regulations.</li>
          </ul>
        </div>
        <div className="warning-message">
          <p>
            <strong>⚠️ Warning:</strong> This tool is for research only. Consult a medical professional before making health-related decisions.
          </p>
        </div>
      </div>

      {/* Right side: Buttons with descriptions */}
      <div className="onboarding-right">
        <div className="button-row">
          <div className="button-col">
            <button onClick={requestPermissions} className="btn btn-success">
              Grant Camera Access
            </button>
            <p className="button-description">
              Enable camera access for real-time emotional and behavioral analysis.
            </p>
          </div>
          <div className="button-col">
            <button onClick={() => navigate('/phq-test')} className="btn btn-secondary">
              Take PHQ-9 Test
            </button>
            <p className="button-description">
              Take a quick PHQ-9 questionnaire to assess your mental well-being.
            </p>
          </div>
        </div>
        <div className="full-width-button">
          <button onClick={() => navigate('/dashboard')} className="btn btn-primary">
            Go to Dashboard
          </button>
          <p className="button-description">
            View your analysis results and insights on your personalized dashboard.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Onboarding;
