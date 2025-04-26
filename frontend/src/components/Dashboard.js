import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import { jsPDF } from 'jspdf';
import 'jspdf-autotable';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import './Dashboard.css';
import SidePanel from './SidePanel';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faClipboard, 
  faChartLine, 
  faMedkit, 
  faCalendarAlt, 
  faDownload, 
  faInfoCircle, 
  faUserNinja, 
  faTrainingGround,
  faVideo,
  faLayerGroup
} from '@fortawesome/free-solid-svg-icons';
import { phqService } from '../services/phqService';

// Register required components for Chart.js
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

// InfoPopup Component
const InfoPopup = ({ isOpen, onClose, title, content }) => {
  if (!isOpen) return null;

  return (
    <div className="info-popup-overlay" onClick={onClose}>
      <div className="info-popup-content" onClick={e => e.stopPropagation()}>
        <div className="info-popup-header">
          <h3>{title}</h3>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        <div className="info-popup-body">
          {content}
        </div>
      </div>
    </div>
  );
};

const Dashboard = () => {
  const [scores, setScores] = useState([]);
  const [activeTooltip, setActiveTooltip] = useState(null);
  const [activePopup, setActivePopup] = useState(null);
  const navigate = useNavigate();

  // Retrieve userId and username from local storage
  const userId = localStorage.getItem('userId');
  const username = localStorage.getItem('username');

  // Redirect to login if no userId is available
  useEffect(() => {
    if (!userId) {
      navigate('/');
    }
  }, [userId, navigate]);

  // Fetch user's PHQ-9 scores from the database
  useEffect(() => {
    const fetchScores = async () => {
      try {
        const data = await phqService.getAllTestScores();
        console.log('Raw API response:', data);
        setScores(data);
      } catch (error) {
        console.error('Error fetching scores:', error);
        if (error.message === 'Session expired. Please login again.') {
          navigate('/');
        }
      }
    };

    if (userId) {
      fetchScores();
    }
  }, [userId, navigate]);

  // Prepare data for the chart
  console.log('Current scores state:', scores);
  const chartData = {
    labels: scores.map((score) => {
      const date = new Date(score.created_at);
      console.log('Processing date:', score.created_at, 'to:', date.toLocaleDateString());
      return date.toLocaleDateString();
    }),
    datasets: [
      {
        label: 'PHQ-9 Test Scores',
        data: scores.map((score) => {
          console.log('Processing score:', score.score);
          return score.score;
        }),
        borderColor: '#88c8f7',
        backgroundColor: 'rgba(136, 200, 247, 0.2)',
        tension: 0.3,
      },
    ],
  };
  console.log('Chart data prepared:', chartData);

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Score',
        },
        beginAtZero: true,
      },
    },
  };

  // Function to download table data as PDF
  const downloadPDF = () => {
    const doc = new jsPDF();
    doc.text('PHQ-9 Test Score Details', 14, 10); // Title of the PDF

    // Define the table headers and rows
    const tableColumn = ['Date', 'Time', 'Score'];
    const tableRows = scores.map((score) => [
      new Date(score.created_at).toLocaleDateString(),
      new Date(score.created_at).toLocaleTimeString(),
      score.score,
    ]);

    // Add table to the PDF
    doc.autoTable({
      head: [tableColumn],
      body: tableRows,
      startY: 20,
    });

    // Save the PDF
    doc.save('phq9-test-scores.pdf');
  };

  // Function to get risk level based on score
  const getRiskLevel = (score) => {
    if (score <= 4) return { level: 'Low', color: '#4CAF50' };
    if (score <= 9) return { level: 'Mild', color: '#FFC107' };
    if (score <= 14) return { level: 'Moderate', color: '#FF9800' };
    if (score <= 19) return { level: 'Moderately Severe', color: '#F44336' };
    return { level: 'Severe', color: '#D32F2F' };
  };

  // Function to get dynamic message based on last score
  const getDynamicMessage = () => {
    if (scores.length === 0) return null;
    const lastScore = scores[scores.length - 1].score;
    const riskLevel = getRiskLevel(lastScore);
    
    const messages = {
      'Low': 'Your recent score suggests minimal symptoms. Keep up the good work!',
      'Mild': 'Your recent score suggests mild symptoms. Consider monitoring your mood regularly.',
      'Moderate': 'Your recent score suggests moderate distress. Consider speaking with a professional.',
      'Moderately Severe': 'Your recent score suggests significant distress. We recommend speaking with a mental health professional.',
      'Severe': 'Your recent score suggests severe distress. Please consider seeking immediate professional help.'
    };

    return messages[riskLevel.level];
  };

  // Info content for each metric
  const metricInfo = {
    totalTests: {
      title: "Total Tests",
      content: (
        <div>
          <p>This represents the total number of PHQ-9 tests you have taken so far.</p>
          <h4>Why is this important?</h4>
          <ul>
            <li>A higher number of tests indicates that you're actively tracking your mental well-being.</li>
            <li>Regular assessments can help identify patterns over time and recognize changes in mental health.</li>
          </ul>
          <p><strong>Example:</strong> If the user has taken 5 tests, the value displayed here will be 5.</p>
        </div>
      )
    },
    averageScore: {
      title: "Average Score",
      content: (
        <div>
          <p>This is the average of all your PHQ-9 test scores.</p>
          <h4>How is it calculated?</h4>
          <p>Formula: (Sum of all scores) ÷ (Total number of tests)</p>
          <h4>Why is this important?</h4>
          <ul>
            <li>Helps you see the overall trend in your mental health.</li>
            <li>A rising average may indicate increasing distress, while a decreasing one suggests improvement.</li>
          </ul>
          <p><strong>Example:</strong> If you took 3 tests with scores 8, 12, and 10, your average score would be: (8+12+10)/3 = 10</p>
          <p><strong>Tip:</strong> A higher score (≥10) may indicate moderate to severe distress.</p>
        </div>
      )
    },
    highestScore: {
      title: "Highest Score",
      content: (
        <div>
          <p>This shows the highest PHQ-9 test score you have ever recorded.</p>
          <h4>Why is this important?</h4>
          <ul>
            <li>A high score may indicate a period of heightened distress.</li>
            <li>If your highest score is recent, consider seeking support or using available resources.</li>
          </ul>
          <h4>Score Meaning Guide:</h4>
          <ul>
            <li>0-4: Minimal distress</li>
            <li>5-9: Mild distress</li>
            <li>10-14: Moderate distress</li>
            <li>15-19: Moderately severe distress</li>
            <li>20-27: Severe distress</li>
          </ul>
          <p><strong>Example:</strong> If the users' previous test scores were 6, 14, and 18, the highest recorded score will be 18 (Moderately Severe).</p>
        </div>
      )
    },
    doctorAppointment: {
      title: "Make Doctor Appointment",
      content: (
        <div>
          <p>This button helps you schedule an appointment with a mental health professional.</p>
          <h4>When should you consider making an appointment?</h4>
          <ul>
            <li>If your latest score is above 10 (Moderate or higher).</li>
            <li>If your distress levels have increased over multiple tests.</li>
            <li>If you feel overwhelmed and need professional guidance.</li>
          </ul>
          <p>If you're not ready to meet a doctor, explore the Help Line section for self-care resources and crisis support.</p>
        </div>
      )
    }
  };

  // PHQ-9 Test Introduction Popup
  const [showPHQIntro, setShowPHQIntro] = useState(false);

  const phqIntroContent = (
    <div>
      <h3>What is the PHQ-9 Test?</h3>
      <p>The PHQ-9 (Patient Health Questionnaire-9) is a clinically validated screening tool used to assess symptoms of depression.</p>
      
      <h4>How it Works:</h4>
      <ul>
        <li>The test consists of 9 questions about your mood, energy levels, sleep, and daily activities.</li>
        <li>Each question is rated on a scale from 0 (Not at all) to 3 (Nearly every day).</li>
        <li>Your total score helps indicate the severity of depressive symptoms.</li>
      </ul>

      <h4>What the Scores Mean:</h4>
      <ul>
        <li>0-4: Minimal or no depression</li>
        <li>5-9: Mild depression</li>
        <li>10-14: Moderate depression</li>
        <li>15-19: Moderately severe depression</li>
        <li>20-27: Severe depression</li>
      </ul>

      <p><strong>Note:</strong> This test is not a diagnosis, but it helps identify potential signs of depression. If you have concerns, consider speaking with a mental health professional.</p>

      <div className="phq-intro-buttons">
        <button 
          className="phq-start-button"
          onClick={() => {
            setShowPHQIntro(false);
            navigate('/phq-test');
          }}
        >
          Yes, Start the Test
        </button>
        <button 
          className="phq-cancel-button"
          onClick={() => setShowPHQIntro(false)}
        >
          Cancel
        </button>
      </div>
    </div>
  );

  return (
    <div className="dashboard-container">
      {/* Use SidePanel */}
      <SidePanel 
        username={username} 
        onPHQTestClick={() => setShowPHQIntro(true)}
      />

      {/* Main Content */}
      <div className="main-content">
        <h1 className="dashboard-heading">Dashboard</h1>
        <hr />

        {/* PHQ-9 Test Introduction Popup */}
        {showPHQIntro && (
          <InfoPopup
            isOpen={showPHQIntro}
            onClose={() => setShowPHQIntro(false)}
            title="PHQ-9 Test Introduction"
            content={phqIntroContent}
          />
        )}

        {/* Dynamic Message */}
        {scores.length > 0 && (
          <div className="dynamic-message" style={{ 
            backgroundColor: getRiskLevel(scores[scores.length - 1].score).color + '20',
            borderLeft: `4px solid ${getRiskLevel(scores[scores.length - 1].score).color}`,
            padding: '10px',
            marginBottom: '20px',
            borderRadius: '4px'
          }}>
            {getDynamicMessage()}
          </div>
        )}

        {/* Statistic Boxes */}
        <div className="stats-boxes">
          <div 
            className="stats-box"
            onMouseEnter={() => setActiveTooltip('totalTests')}
            onMouseLeave={() => setActiveTooltip(null)}
          >
            <FontAwesomeIcon icon={faClipboard} className="box-icon" />
            <div>Total Tests</div>
            <div>{scores.length}</div>
            <button 
              className="info-button"
              onClick={() => setActivePopup('totalTests')}
            >
              <FontAwesomeIcon icon={faInfoCircle} />
            </button>
            {activeTooltip === 'totalTests' && (
              <div className="tooltip">
                <FontAwesomeIcon icon={faInfoCircle} style={{ marginRight: '5px' }} />
                Total number of PHQ-9 tests taken by the user so far
              </div>
            )}
          </div>

          <div 
            className="stats-box"
            onMouseEnter={() => setActiveTooltip('averageScore')}
            onMouseLeave={() => setActiveTooltip(null)}
          >
            <FontAwesomeIcon icon={faChartLine} className="box-icon" />
            <div>Average Score</div>
            <div>
              {scores.length > 0 ? (scores.reduce((a, b) => a + b.score, 0) / scores.length).toFixed(2) : 0}
            </div>
            <button 
              className="info-button"
              onClick={() => setActivePopup('averageScore')}
            >
              <FontAwesomeIcon icon={faInfoCircle} />
            </button>
            {activeTooltip === 'averageScore' && (
              <div className="tooltip">
                <FontAwesomeIcon icon={faInfoCircle} style={{ marginRight: '5px' }} />
                The average score of all PHQ-9 tests taken
              </div>
            )}
          </div>

          <div 
            className="stats-box"
            onMouseEnter={() => setActiveTooltip('highestScore')}
            onMouseLeave={() => setActiveTooltip(null)}
          >
            <FontAwesomeIcon icon={faCalendarAlt} className="box-icon" />
            <div>Highest Score</div>
            {scores.length > 0 ? (
              <div>
                {Math.max(...scores.map((s) => s.score))}
                <span style={{ 
                  color: getRiskLevel(Math.max(...scores.map((s) => s.score))).color,
                  marginLeft: '5px',
                  fontSize: '0.8em'
                }}>
                  ({getRiskLevel(Math.max(...scores.map((s) => s.score))).level})
                </span>
              </div>
            ) : (
              <div>0</div>
            )}
            <button 
              className="info-button"
              onClick={() => setActivePopup('highestScore')}
            >
              <FontAwesomeIcon icon={faInfoCircle} />
            </button>
            {activeTooltip === 'highestScore' && (
              <div className="tooltip">
                <FontAwesomeIcon icon={faInfoCircle} style={{ marginRight: '5px' }} />
                The highest PHQ-9 test score recorded among all the tests taken
              </div>
            )}
          </div>

          <div
            className="stats-box clickable-box"
            onClick={() =>
              window.open('https://nimh.health.gov.lk/en/appointment-form-navodaya-patient-booking-system/', '_blank')
            }
            onMouseEnter={() => setActiveTooltip('doctorAppointment')}
            onMouseLeave={() => setActiveTooltip(null)}
          >
            <FontAwesomeIcon icon={faMedkit} className="box-icon" />
            <div>Make Doctor Appointment</div>
            <button 
              className="info-button"
              onClick={(e) => {
                e.stopPropagation();
                setActivePopup('doctorAppointment');
              }}
            >
              <FontAwesomeIcon icon={faInfoCircle} />
            </button>
            {activeTooltip === 'doctorAppointment' && (
              <div className="tooltip">
                <FontAwesomeIcon icon={faInfoCircle} style={{ marginRight: '5px' }} />
                Click to schedule an appointment with a mental health professional if your scores indicate potential distress
              </div>
            )}
          </div>
        </div>

        {/* Info Popups */}
        {activePopup && (
          <InfoPopup
            isOpen={!!activePopup}
            onClose={() => setActivePopup(null)}
            title={metricInfo[activePopup].title}
            content={metricInfo[activePopup].content}
          />
        )}

        {/* Chart and Table */}
        <div className="chart-and-table">
          <div className="chart-container">
            <h3>PHQ-9 Test Trends</h3>
            <Line data={chartData} options={chartOptions} />
          </div>

          <div className="table-container">
            <div className="table-header">
              <h3>Test Score Details</h3>
              <button className="download-button" onClick={downloadPDF}>
                <FontAwesomeIcon icon={faDownload} /> Download PDF
              </button>
            </div>
            <table className="score-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Time</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                {scores.length > 0 ? (
                  scores.map((score, index) => (
                    <tr key={index}>
                      <td>{new Date(score.created_at).toLocaleDateString()}</td>
                      <td>{new Date(score.created_at).toLocaleTimeString()}</td>
                      <td>{score.score}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="3">No test scores available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
