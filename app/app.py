from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import bcrypt

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Using SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    city = db.Column(db.String(50), nullable=True)
    district = db.Column(db.String(50), nullable=True)
    address = db.Column(db.String(200), nullable=True)
    state = db.Column(db.String(50), nullable=True)
    country = db.Column(db.String(50), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    dob = db.Column(db.String(10), nullable=True)  # Date of birth as string (YYYY-MM-DD)
    gender = db.Column(db.String(10), nullable=True)
    language = db.Column(db.String(50), nullable=True)
    interest_level = db.Column(db.String(50), nullable=True)
    occupation = db.Column(db.String(50), nullable=True)
    communication_channel = db.Column(db.String(50), nullable=True)


# Create the database
with app.app_context():
    db.create_all()


# POST API to add a new user
@app.route('/users', methods=['POST'])
def add_user():
    data = request.get_json()

    # Validate required fields
    required_fields = ['fullName', 'username', 'password', 'email']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

    # Create a new user
    new_user = User(
        full_name=data['fullName'],
        username=data['username'],
        password=hashed_password.decode('utf-8'),  # Store the hashed password as a string
        email=data['email'],
        city=data.get('city'),
        district=data.get('district'),
        address=data.get('address'),
        state=data.get('state'),
        country=data.get('country'),
        phone=data.get('phone'),
        dob=data.get('dob'),
        gender=data.get('gender'),
        language=data.get('language'),
        interest_level=data['interestLevel'],
        occupation=data.get('occupation'),
        communication_channel=data['communicationChannel']
    )

    try:
        db.session.add(new_user)
        db.session.commit()

        return jsonify({
            'message': 'User created successfully',
            'userId': new_user.id  # Include the userId in the response
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# Verify user login (example route)
@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return jsonify({'message': 'Login successful'})
    return jsonify({'error': 'Invalid username or password'}), 401


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
