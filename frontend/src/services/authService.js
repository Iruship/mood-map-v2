import api from './api';
import { jwtDecode } from 'jwt-decode';

export const authService = {
  login: async (username, password) => {
    const response = await api.post('/auth/login', { username, password });
    if (response.data.access_token) {
      const token = response.data.access_token;
      // Store the token
      localStorage.setItem('token', token);
      
      // Decode and store user info from token
      const decodedToken = jwtDecode(token);
      localStorage.setItem('userId', decodedToken._id);
      localStorage.setItem('username', decodedToken.username);
      localStorage.setItem('email', decodedToken.email);
      localStorage.setItem('name', decodedToken.name);
      localStorage.setItem('tokenExp', decodedToken.exp);
    }
    return response.data;
  },

  signup: async (userData) => {
    const response = await api.post('/auth/register', {
      name: userData.name,
      username: userData.username,
      email: userData.email,
      password: userData.password,
      confirm_password: userData.confirmPassword
    });
    return response.data;
  },

  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('userId');
    localStorage.removeItem('username');
    localStorage.removeItem('email');
    localStorage.removeItem('name');
    localStorage.removeItem('tokenExp');
  },

  isAuthenticated: () => {
    const token = localStorage.getItem('token');
    if (!token) return false;
    
    // Check token expiration
    const tokenExp = localStorage.getItem('tokenExp');
    if (tokenExp) {
      const currentTime = Math.floor(Date.now() / 1000);
      return currentTime < parseInt(tokenExp);
    }
    return false;
  },

  isAdmin: () => {
    return localStorage.getItem('username') === 'admin';
  },

  getCurrentUser: () => {
    return {
      userId: localStorage.getItem('userId'),
      username: localStorage.getItem('username'),
      email: localStorage.getItem('email'),
      name: localStorage.getItem('name')
    };
  }
}; 