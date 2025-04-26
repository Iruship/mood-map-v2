from datetime import datetime
import re
import bcrypt
import uuid

class User:
    def __init__(self, name, username, email, password):
        self.id = str(uuid.uuid4())
        self.name = name
        self.username = username
        self.email = email
        self.password = self._hash_password(password)
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def _hash_password(self, password):
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password)

    @staticmethod
    def validate_registration(name, username, email, password, confirm_password):
        errors = []
        
        # Validate name
        if not name or len(name.strip()) < 2:
            errors.append("Name must be at least 2 characters long")

        # Validate username
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters long")
        if not re.match("^[a-zA-Z0-9_]+$", username):
            errors.append("Username can only contain letters, numbers, and underscores")

        # Validate email
        email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not email or not email_regex.match(email):
            errors.append("Invalid email address")

        # Validate password
        if not password or len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        # Validate password confirmation
        if password != confirm_password:
            errors.append("Passwords do not match")

        return errors

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @staticmethod
    def from_dict(data):
        user = User(
            name=data.get("name"),
            username=data.get("username"),
            email=data.get("email"),
            password=data.get("password")
        )
        if "id" in data:
            user.id = data["id"]
        return user 