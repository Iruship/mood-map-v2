import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './Dashboard.css';
import SidePanel from './SidePanel';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDownload } from '@fortawesome/free-solid-svg-icons';
import { jsPDF } from 'jspdf';
import 'jspdf-autotable';

const AdminDashboard = () => {
  const [scores, setScores] = useState([]);

  useEffect(() => {
    const fetchScores = async () => {
      try {
        const response = await axios.get('http://localhost:4990/api/all');
        console.log('API Response:', response.data);
        setScores(response.data);
      } catch (error) {
        console.error('Error fetching scores:', error); // Debug: log errors to console
      }
    };
  
    fetchScores();
  }, []);
  

  const downloadPDF = () => {
    const doc = new jsPDF();
    doc.text('PHQ-9 Test Scores (Admin)', 14, 10);
    const tableColumn = ['Name', 'Date', 'Time', 'Score'];
    const tableRows = scores.map((score) => [
      score.username || 'Unknown',
      new Date(score.date).toLocaleDateString(),
      new Date(score.date).toLocaleTimeString(),
      score.score,
    ]);

    doc.autoTable({
      head: [tableColumn],
      body: tableRows,
      startY: 20,
    });

    doc.save('phq9-admin-scores.pdf');
  };

  return (
    <div className="dashboard-container">
      {/* Use SidePanel */}
      <SidePanel username="Admin" />

      {/* Main Content */}
      <div className="main-content">
        <h1 className="dashboard-heading">Admin Dashboard</h1>
        <hr />

        {/* Table */}
        <div className="table-container">
          <div className="table-header">
            <h3>PHQ-9 Test Scores</h3>
            <button className="download-button" onClick={downloadPDF}>
              <FontAwesomeIcon icon={faDownload} /> Download PDF
            </button>
          </div>
          <table className="score-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Date</th>
                <th>Time</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {scores.length > 0 ? (
                scores.map((score, index) => (
                  <tr key={index}>
                    <td>{score.username || 'Unknown'}</td>
                    <td>{new Date(score.date).toLocaleDateString()}</td>
                    <td>{new Date(score.date).toLocaleTimeString()}</td>
                    <td>{score.score}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="4">No test scores available</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
