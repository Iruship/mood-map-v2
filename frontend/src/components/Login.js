import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { authService } from '../services/authService';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEye, faEyeSlash } from '@fortawesome/free-solid-svg-icons';
import './Login.css';

const Login = () => {
  const [isSignup, setIsSignup] = useState(false);
  const [name, setName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [passwordError, setPasswordError] = useState('');
  const [agreeToTerms, setAgreeToTerms] = useState(false);
  const [signupStep, setSignupStep] = useState(1);
  
  const navigate = useNavigate();

  const validatePassword = (pass) => {
    if (pass.length < 8) {
      setPasswordError('Password must be at least 8 characters long');
      return false;
    }
    if (!/[A-Z]/.test(pass)) {
      setPasswordError('Password must contain at least one uppercase letter');
      return false;
    }
    if (!/[a-z]/.test(pass)) {
      setPasswordError('Password must contain at least one lowercase letter');
      return false;
    }
    if (!/[0-9]/.test(pass)) {
      setPasswordError('Password must contain at least one number');
      return false;
    }
    if (!/[!@#$%^&*]/.test(pass)) {
      setPasswordError('Password must contain at least one special character (!@#$%^&*)');
      return false;
    }
    setPasswordError('');
    return true;
  };

  const handlePasswordChange = (e) => {
    const newPassword = e.target.value;
    setPassword(newPassword);
    if (newPassword) {
      validatePassword(newPassword);
    } else {
      setPasswordError('');
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    localStorage.clear();
    
    // Check if the user is logging in as admin
    if (username === 'admin' && password === 'admin') {
      localStorage.setItem('username', 'admin');
      toast.success('Logged in as Admin');
      navigate('/admin-dashboard');
      return;
    }

    try {
      const data = await authService.login(username, password);
      toast.success(data.message || 'Login successful');
      navigate('/onboarding');
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.response?.data?.error || error.message;
      toast.error('Login failed: ' + errorMessage);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();

    if (!agreeToTerms) {
      toast.error('Please agree to the Terms and Conditions');
      return;
    }

    if (!validatePassword(password)) {
      return;
    }

    if (password !== confirmPassword) {
      toast.error('Passwords do not match');
      return;
    }

    try {
      const data = await authService.signup({
        name,
        username,
        email,
        password,
        confirmPassword
      });
      toast.success(data.message || 'Registration successful');
      setIsSignup(false);
    } catch (error) {
      let errorMessage;
      
      if (error.response?.data?.detail && Array.isArray(error.response.data.detail)) {
        errorMessage = error.response.data.detail
          .map(err => err.msg)
          .join(', ');
      } else {
        errorMessage = error.response?.data?.detail || 
                      error.response?.data?.error || 
                      error.message;
      }
      
      toast.error('Signup failed: ' + errorMessage);
    }
  };

  const nextStep = () => {
    if (signupStep === 1 && (!name || !email)) {
      toast.error('Please fill in all fields');
      return;
    }
    setSignupStep(2);
  };

  const prevStep = () => {
    setSignupStep(1);
  };

  return (
    <div className="login-page">
      <ToastContainer />
      <div className="header-section">
        <div className="title-container">
          <h1 className="left-title">Feeling Down?</h1>
          <h2 className="right-title">Check Your Mental Well-being!</h2>
        </div>
      </div>
      <div className="curved-background"></div>
      <div className="login-container">
        <div className="login-box">
          <h2>{isSignup ? 'Sign Up' : 'Login'}</h2>
          {isSignup && (
            <div className="progress-bar">
              <div 
                className="progress" 
                style={{ width: signupStep === 1 ? '50%' : '100%' }}
              ></div>
              <span>Step {signupStep} of 2</span>
            </div>
          )}
          <form onSubmit={isSignup ? handleSignup : handleLogin}>
            {isSignup && signupStep === 1 && (
              <>
                <input
                  type="text"
                  className="form-control"
                  placeholder="Enter your full name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
                <input
                  type="email"
                  className="form-control"
                  placeholder="Enter your email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
                <button type="button" className="btn btn-primary" onClick={nextStep}>
                  Next
                </button>
                <button
                  type="button"
                  className="btn-cancel"
                  onClick={() => setIsSignup(false)}
                >
                  Cancel
                </button>
              </>
            )}
            {isSignup && signupStep === 2 && (
              <>
                <input
                  type="text"
                  className="form-control"
                  placeholder="Choose a username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
                <div className="password-input-container">
                  <input
                    type={showPassword ? "text" : "password"}
                    className="form-control"
                    placeholder="Create a password"
                    value={password}
                    onChange={handlePasswordChange}
                    required
                  />
                  <button
                    type="button"
                    className="toggle-password"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    <FontAwesomeIcon icon={showPassword ? faEyeSlash : faEye} />
                  </button>
                </div>
                {passwordError && <div className="password-error">{passwordError}</div>}
                <div className="password-input-container">
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    className="form-control"
                    placeholder="Confirm your password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                  />
                  <button
                    type="button"
                    className="toggle-password"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    <FontAwesomeIcon icon={showConfirmPassword ? faEyeSlash : faEye} />
                  </button>
                </div>
                <div className="terms-container">
                  <label className="terms-label">
                    <input
                      type="checkbox"
                      checked={agreeToTerms}
                      onChange={(e) => setAgreeToTerms(e.target.checked)}
                    />
                    <span>I agree to the <a href="/terms" target="_blank">Terms and Conditions</a> and <a href="/privacy" target="_blank">Privacy Policy</a></span>
                  </label>
                </div>
                <button type="submit" className="btn btn-primary">
                  Sign Up
                </button>
                <button type="button" className="btn-secondary" onClick={prevStep}>
                  Back
                </button>
              </>
            )}
            {!isSignup && (
              <>
                <input
                  type="text"
                  className="form-control"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
                <div className="password-input-container">
                  <input
                    type={showPassword ? "text" : "password"}
                    className="form-control"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                  <button
                    type="button"
                    className="toggle-password"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    <FontAwesomeIcon icon={showPassword ? faEyeSlash : faEye} />
                  </button>
                </div>
                <button type="submit" className="btn btn-primary">
                  Login
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setIsSignup(true);
                    setSignupStep(1);
                  }}
                >
                  Sign Up
                </button>
              </>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default Login;
