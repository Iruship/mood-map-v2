import api from './api';
import { authService } from './authService';

export const phqService = {
  saveTestResult: async (testData) => {
    // Check if token is expired
    if (!authService.isAuthenticated()) {
      // Clear storage and throw error
      localStorage.clear();
      throw new Error('Session expired. Please login again.');
    }

    const response = await api.post('/phq9/submit', {
      score: testData.score
    });
    return response.data;
  },

  getTestHistory: async (userId) => {
    const response = await api.get(`/phq-test/history/${userId}`);
    return response.data;
  },

  getLatestResult: async (userId) => {
    const response = await api.get(`/phq-test/latest/${userId}`);
    return response.data;
  },

  getAllTestScores: async (limit = null) => {
    if (!authService.isAuthenticated()) {
      localStorage.clear();
      throw new Error('Session expired. Please login again.');
    }
    const url = limit ? `/phq9/history?limit=${limit}` : '/phq9/history';
    console.log('Making API request to:', url);
    const response = await api.get(url);
    console.log('API response:', response.data);
    return response.data;
  },

  calculateDepressionLevel: (score) => {
    if (score >= 20) return "Severe depression";
    if (score >= 15) return "Moderately severe depression";
    if (score >= 10) return "Moderate depression";
    if (score >= 5) return "Mild depression";
    return "Minimal or no depression";
  }
}; 