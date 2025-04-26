import api from './api';

export const emotionService = {
  detectEmotion: async (imageData) => {
    const response = await api.post('/emotion-detection', imageData);
    return response.data;
  },

  saveEmotionResult: async (emotionData) => {
    const response = await api.post('/emotion', emotionData);
    return response.data;
  },

  getEmotionHistory: async (userId) => {
    const response = await api.get(`/emotion/history/${userId}`);
    return response.data;
  },

  getLatestEmotion: async (userId) => {
    const response = await api.get(`/emotion/latest/${userId}`);
    return response.data;
  }
}; 