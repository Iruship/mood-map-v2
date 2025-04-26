import React, { useState, useEffect } from 'react';
import bodyLanguageService from '../services/bodyLanguageService';
import { 
  Box,
  Button,
  Slider,
  Typography,
  FormControlLabel,
  Switch,
  TextField,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Divider,
  Alert,
  Snackbar
} from '@mui/material';

const ThresholdSettings = () => {
  const [thresholds, setThresholds] = useState({
    detection_confidence: 0.7,
    tracking_confidence: 0.7,
    model_complexity: 2,
    prediction_confidence: 0.65,
    pad_missing_features: true,
    handle_nan_values: true,
    min_training_samples: 10
  });
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  useEffect(() => {
    fetchThresholds();
  }, []);
  
  const fetchThresholds = async () => {
    try {
      setLoading(true);
      const response = await bodyLanguageService.getThresholds();
      if (response.success && response.current_values) {
        setThresholds(response.current_values);
      } else {
        setError("Failed to load threshold settings");
      }
    } catch (err) {
      setError("Error loading threshold settings: " + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSliderChange = (name) => (event, newValue) => {
    setThresholds({
      ...thresholds,
      [name]: newValue
    });
  };
  
  const handleSwitchChange = (name) => (event) => {
    setThresholds({
      ...thresholds,
      [name]: event.target.checked
    });
  };
  
  const handleNumberChange = (name) => (event) => {
    const value = parseInt(event.target.value, 10);
    if (!isNaN(value) && value > 0) {
      setThresholds({
        ...thresholds,
        [name]: value
      });
    }
  };
  
  const handleModelComplexityChange = (event) => {
    const value = parseInt(event.target.value, 10);
    if ([0, 1, 2].includes(value)) {
      setThresholds({
        ...thresholds,
        model_complexity: value
      });
    }
  };
  
  const handleResetDefaults = async () => {
    try {
      setLoading(true);
      const defaultThresholds = {
        detection_confidence: 0.7,
        tracking_confidence: 0.7,
        model_complexity: 2,
        prediction_confidence: 0.65,
        pad_missing_features: true,
        handle_nan_values: true,
        min_training_samples: 10
      };
      
      const response = await bodyLanguageService.updateThresholds(defaultThresholds);
      if (response.success) {
        setThresholds(response.current_values);
        setSuccess("Thresholds reset to defaults");
      } else {
        setError(response.error || "Failed to reset thresholds");
      }
    } catch (err) {
      setError("Error resetting thresholds: " + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSaveThresholds = async () => {
    try {
      setLoading(true);
      const response = await bodyLanguageService.updateThresholds(thresholds);
      if (response.success) {
        setSuccess("Thresholds updated successfully");
      } else {
        setError(response.error || "Failed to update thresholds");
      }
    } catch (err) {
      setError("Error updating thresholds: " + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const handleCloseSnackbar = () => {
    setSuccess(null);
    setError(null);
  };
  
  return (
    <Card variant="outlined" sx={{ mb: 2, maxWidth: 700, mx: 'auto' }}>
      <CardHeader 
        title="Body Language Detection Settings" 
        subheader="Adjust thresholds for real-time detection"
      />
      <Divider />
      <CardContent>
        {loading ? (
          <Typography>Loading threshold settings...</Typography>
        ) : (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                MediaPipe Detection Settings
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography id="detection-confidence-slider" gutterBottom>
                Detection Confidence: {thresholds.detection_confidence.toFixed(2)}
              </Typography>
              <Slider
                value={thresholds.detection_confidence}
                onChange={handleSliderChange('detection_confidence')}
                aria-labelledby="detection-confidence-slider"
                valueLabelDisplay="auto"
                step={0.05}
                min={0.1}
                max={1}
                disabled={loading}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography id="tracking-confidence-slider" gutterBottom>
                Tracking Confidence: {thresholds.tracking_confidence.toFixed(2)}
              </Typography>
              <Slider
                value={thresholds.tracking_confidence}
                onChange={handleSliderChange('tracking_confidence')}
                aria-labelledby="tracking-confidence-slider"
                valueLabelDisplay="auto"
                step={0.05}
                min={0.1}
                max={1}
                disabled={loading}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                select
                fullWidth
                label="Model Complexity"
                value={thresholds.model_complexity}
                onChange={handleModelComplexityChange}
                SelectProps={{
                  native: true,
                }}
                disabled={loading}
                helperText="Higher values are more accurate but slower"
              >
                <option value={0}>Low (0)</option>
                <option value={1}>Medium (1)</option>
                <option value={2}>High (2)</option>
              </TextField>
            </Grid>
            
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6" gutterBottom>
                Prediction Settings
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography id="prediction-confidence-slider" gutterBottom>
                Prediction Confidence: {thresholds.prediction_confidence.toFixed(2)}
              </Typography>
              <Slider
                value={thresholds.prediction_confidence}
                onChange={handleSliderChange('prediction_confidence')}
                aria-labelledby="prediction-confidence-slider"
                valueLabelDisplay="auto"
                step={0.05}
                min={0.1}
                max={1}
                disabled={loading}
              />
              <Typography variant="caption" color="text.secondary">
                Minimum confidence needed for a prediction to be shown
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Minimum Training Samples"
                type="number"
                value={thresholds.min_training_samples}
                onChange={handleNumberChange('min_training_samples')}
                InputProps={{ inputProps: { min: 1 } }}
                disabled={loading}
                helperText="Minimum samples needed to train a model"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6" gutterBottom>
                Feature Processing
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={thresholds.pad_missing_features}
                    onChange={handleSwitchChange('pad_missing_features')}
                    disabled={loading}
                  />
                }
                label="Pad Missing Features"
              />
              <Typography variant="caption" display="block" color="text.secondary">
                Automatically handle feature count mismatch
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={thresholds.handle_nan_values}
                    onChange={handleSwitchChange('handle_nan_values')}
                    disabled={loading}
                  />
                }
                label="Handle NaN Values"
              />
              <Typography variant="caption" display="block" color="text.secondary">
                Replace NaN values with zeros or column means
              </Typography>
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={handleResetDefaults}
                  disabled={loading}
                >
                  Reset to Defaults
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleSaveThresholds}
                  disabled={loading}
                >
                  Save Settings
                </Button>
              </Box>
            </Grid>
          </Grid>
        )}
      </CardContent>
      
      <Snackbar 
        open={!!success} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>
      
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Card>
  );
};

export default ThresholdSettings; 