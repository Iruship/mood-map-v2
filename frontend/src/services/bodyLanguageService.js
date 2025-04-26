import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

class BodyLanguageService {
  // Get current detection thresholds
  async getThresholds() {
    try {
      const response = await axios.get(`${API_URL}/api/body-language/get-thresholds`);
      return response.data;
    } catch (error) {
      console.error('Error getting thresholds:', error);
      throw error;
    }
  }

  // Update detection thresholds
  async updateThresholds(thresholdData) {
    try {
      const response = await axios.post(
        `${API_URL}/api/body-language/update-thresholds`, 
        thresholdData
      );
      return response.data;
    } catch (error) {
      console.error('Error updating thresholds:', error);
      throw error;
    }
  }

  // Get available models
  async getAvailableModels() {
    try {
      const response = await axios.get(`${API_URL}/api/body-language/available-models`);
      return response.data;
    } catch (error) {
      console.error('Error getting available models:', error);
      throw error;
    }
  }

  // Switch to a different model
  async switchModel(modelType) {
    try {
      const formData = new FormData();
      formData.append('model_type', modelType);

      const response = await axios.post(
        `${API_URL}/api/body-language/switch-model`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error switching model:', error);
      throw error;
    }
  }

  // Process an image for body language detection
  async detectBodyLanguage(imageFile) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await axios.post(
        `${API_URL}/api/body-language/detect`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error detecting body language:', error);
      throw error;
    }
  }

  // Record training data for a body language class
  async recordTrainingData(imageFile, className) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('class_name', className);

      const response = await axios.post(
        `${API_URL}/api/body-language/record-training-data`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error recording training data:', error);
      throw error;
    }
  }

  // Train the body language model
  async trainModel() {
    try {
      const response = await axios.post(`${API_URL}/api/body-language/train-model`);
      return response.data;
    } catch (error) {
      console.error('Error training model:', error);
      throw error;
    }
  }

  // Get available body language classes
  async getAvailableClasses() {
    try {
      const response = await axios.get(`${API_URL}/api/body-language/available-classes`);
      return response.data;
    } catch (error) {
      console.error('Error getting available classes:', error);
      throw error;
    }
  }

  // Delete all training data and models
  async deleteAllData() {
    try {
      const response = await axios.post(`${API_URL}/api/body-language/delete-all-data`);
      return response.data;
    } catch (error) {
      console.error('Error deleting training data:', error);
      throw error;
    }
  }

  // Process a frame and get back an annotated image with landmarks
  async processFrame(imageFile) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await axios.post(
        `${API_URL}/api/body-language/process-frame`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error processing frame:', error);
      throw error;
    }
  }
}

export default new BodyLanguageService();
