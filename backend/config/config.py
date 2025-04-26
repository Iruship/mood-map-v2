from decouple import config

class Config:
    SECRET_KEY = config('SECRET_KEY')
    MONGODB_URI = config('MONGODB_URI')
    JWT_SECRET_KEY = config('JWT_SECRET_KEY')
    JWT_ACCESS_TOKEN_EXPIRES = 24 * 60 * 60  # 24 hours in seconds

    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set."""
        required_vars = ['SECRET_KEY', 'MONGODB_URI', 'JWT_SECRET_KEY']
        missing_vars = [var for var in required_vars if not config(var, default=None)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")