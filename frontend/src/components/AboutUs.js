import React from 'react';
import SidePanel from './SidePanel';
import './AboutUs.css';

const AboutUs = () => {
  return (
    <div className="aboutus-container">
      {/* Use SidePanel */}
      <SidePanel username="User" />

      {/* Main Content */}
      <div className="main-content">
        <h1 className="aboutus-heading">About Us/Project</h1>
        <hr />

        {/* Top Panel */}
        <div className="top-panel">
          {/* Profile Image */}
          <div className="profile-image">
            <img src="/profile_picture.png" alt="Profile" className="profile-picture" />
          </div>

          {/* User Details */}
          <div className="user-details">
            <h2>Name: Irushi Perera</h2>
            <p>University: Computer Science Undergraduate | University of Westminster, UK</p>
            <p>Telephone: +94 #########</p>
            <p>University ID: XYZ123456</p>
            <p>
              <strong>Project Description:</strong> This project aims to provide mental health support using technology. It
              includes features like PHQ-9 tests, live depression detection, and resources to connect users with mental health
              support services.
            </p>
          </div>
        </div>

        {/* Project Details */}
        <div className="project-details">
          <h3>How the Project Works</h3>
          <p>
            This project leverages modern web technologies and machine learning to assess the mental health of users. The PHQ-9
            test provides a baseline score, while live video analysis helps detect emotional states in real-time. Additionally,
            the platform connects users with trusted mental health resources and helplines.
          </p>

          <h3>Available Features</h3>
          <ul>
            <li>PHQ-9 Mental Health Test</li>
            <li>Live Depression Detection Using Video</li>
            <li>Image Upload and Analysis</li>
            <li>Access to Helplines and Mental Health Support</li>
            <li>Admin Dashboard to Manage PHQ-9 Data</li>
          </ul>

          <h3>Additional Details</h3>
          <p>
            The project is designed to be scalable and user-friendly, providing a comprehensive solution for mental health
            assessment and support. It integrates responsive design, secure authentication, and robust data visualization to
            deliver a seamless experience for users and administrators.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AboutUs;
