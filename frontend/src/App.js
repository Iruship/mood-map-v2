import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { authService } from './services/authService';
import Login from './components/Login';
import Onboarding from './components/Onboarding';
import Dashboard from './components/Dashboard';
import PHQTest from './components/PHQTest';
import LiveDepressionDetection from './components/LiveDepressionDetection';
import AdminDashboard from './components/AdminDashboard';
import HelpLine from './components/HelpLine';
import AboutUs from './components/AboutUs';
import EmotionAnalysis from './components/EmotionAnalysis';
import BodyLanguageTraining from './components/BodyLanguageTraining';
import BodyLanguageDetection from './components/BodyLanguageDetection';
import ThresholdSettings from './components/ThresholdSettings';

// Protected Route component
const ProtectedRoute = ({ children }) => {
  const isAuthenticated = authService.isAuthenticated();
  
  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return children;
};

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/onboarding" element={
          <ProtectedRoute>
            <Onboarding />
          </ProtectedRoute>
        } />
        <Route path="/dashboard" element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } />
        <Route path="/phq-test" element={
          <ProtectedRoute>
            <PHQTest />
          </ProtectedRoute>
        } />
        <Route path="/live-depression-detection" element={
          <ProtectedRoute>
            <LiveDepressionDetection />
          </ProtectedRoute>
        } />
        <Route path="/emotion-analysis" element={
          <ProtectedRoute>
            <EmotionAnalysis />
          </ProtectedRoute>
        } />
        <Route path="/body-language-training" element={
          <ProtectedRoute>
            <BodyLanguageTraining />
          </ProtectedRoute>
        } />
        <Route path="/body-language-detection" element={
          <ProtectedRoute>
            <BodyLanguageDetection />
          </ProtectedRoute>
        } />
        <Route path="/body-language-settings" element={
          <ProtectedRoute>
            <ThresholdSettings />
          </ProtectedRoute>
        } />
        <Route path="/admin-dashboard" element={
          <ProtectedRoute>
            <AdminDashboard />
          </ProtectedRoute>
        } />
        <Route path="/help-line" element={
          <ProtectedRoute>
            <HelpLine />
          </ProtectedRoute>
        } />
        <Route path="/about-us" element={
          <ProtectedRoute>
            <AboutUs />
          </ProtectedRoute>
        } />
      </Routes>
    </Router>
  );
}

export default App;
