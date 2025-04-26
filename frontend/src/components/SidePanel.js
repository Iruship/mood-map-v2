import React from 'react';
import { useNavigate } from 'react-router-dom';
import { authService } from '../services/authService';
import './SidePanel.css';

const SidePanel = ({ username, onPHQTestClick }) => {
  const navigate = useNavigate();

  const handleLogout = (e) => {
    e.preventDefault();
    authService.logout();
    navigate('/');
  };

  return (
    <div className="side-panel">
      <div className="profile-section">
          <img src="/profile_picture.png" alt="Profile" className="profile-picture" />
          <h2>Welcome {username}</h2>
        </div>

      <div className="links-section">
        <button className="transparent-button" onClick={() => navigate('/dashboard')}>
          Dashboard
        </button>
        <button className="transparent-button" onClick={() => navigate('/live-depression-detection')}>
          Live Depression Detection
        </button>
        <button className="transparent-button" onClick={() => navigate('/emotion-analysis')}>
          Emotion Analysis
        </button>
        <button className="transparent-button" onClick={() => navigate('/body-language-training')}>
          Body Language Training
        </button>
        <button className="transparent-button" onClick={() => navigate('/body-language-detection')}>
          Body Language Detection
        </button>
        {/* <button className="transparent-button" onClick={() => navigate('/body-language-settings')}>
          Body Language Settings
        </button> */}
        <button className="transparent-button" onClick={() => navigate('/onboarding')}>
          Onboarding
        </button>
        <button className="transparent-button" onClick={onPHQTestClick}>
          PHQ-9 Test
        </button>
        <button className="transparent-button" onClick={() => navigate('/help-line')}>
          Help Line
        </button>
        <button className="transparent-button" onClick={() => navigate('/about-us')}>
          About Us
        </button>
      </div>

      <div className="logout">
        <button onClick={handleLogout}>Logout</button>
      </div>
    </div>
  );
};

export default SidePanel;
