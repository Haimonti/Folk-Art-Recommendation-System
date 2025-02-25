from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import bcrypt
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=[os.getenv('FRONTEND_DOMAIN', 'https://frontend.com')])

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/mydb')
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


class UserArtPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)

    artists = db.Column(db.String(255), nullable=True)
    rural_art_interest_degree = db.Column(db.Integer, nullable=True)  # Converted from low/medium/high
    south_asian_mythology_familiarity = db.Column(db.Integer, nullable=True)
    south_american_mythology_familiarity = db.Column(db.Integer, nullable=True)
    greek_mythology_familiarity = db.Column(db.Integer, nullable=True)
    roman_mythology_familiarity = db.Column(db.Integer, nullable=True)
    australian_mythology_familiarity = db.Column(db.Integer, nullable=True)
    african_mythology_familiarity = db.Column(db.Integer, nullable=True)

    rajasthan_familiarity = db.Column(db.Integer, nullable=True)
    west_bengal_familiarity = db.Column(db.Integer, nullable=True)
    andra_pradesh_familiarity = db.Column(db.Integer, nullable=True)
    chattisgarh_familiarity = db.Column(db.Integer, nullable=True)
    maharashtra_familiarity = db.Column(db.Integer, nullable=True)
    gujrat_familiarity = db.Column(db.Integer, nullable=True)
    karnataka_familiarity = db.Column(db.Integer, nullable=True)
    tamil_nadu_familiarity = db.Column(db.Integer, nullable=True)

    indonesia_familiarity = db.Column(db.Integer, nullable=True)
    bali_java_borneo_familiarity = db.Column(db.Integer, nullable=True)
    chinese_familiarity = db.Column(db.Integer, nullable=True)
    persian_familiarity = db.Column(db.Integer, nullable=True)
    tibetan_familiarity = db.Column(db.Integer, nullable=True)
    srilankan_familiarity = db.Column(db.Integer, nullable=True)

    scroll_length_preference = db.Column(db.Integer, nullable=True)  # 'long' or 'short'
    art_books_reading_frequency = db.Column(db.Integer, nullable=True)  # 'monthly', 'annually', etc.

    classical_familiarity = db.Column(db.Integer, nullable=True)
    sanskrit_chants_familiarity = db.Column(db.Integer, nullable=True)
    folk_music_familiarity = db.Column(db.Integer, nullable=True)
    rock_familiarity = db.Column(db.Integer, nullable=True)
    hip_hop_familiarity = db.Column(db.Integer, nullable=True)
    bhajans_familiarity = db.Column(db.Integer, nullable=True)
    epic_familiarity = db.Column(db.Integer, nullable=True)
    folk_tales_familiarity = db.Column(db.Integer, nullable=True)
    sanskrit_texts_familiarity = db.Column(db.Integer, nullable=True)
    religious_literature_familiarity = db.Column(db.Integer, nullable=True)

    watched_performances = db.Column(db.Integer, nullable=True)  # Converted from yes/no
    visited_museum = db.Column(db.Integer, nullable=True)  # Converted from yes/no
    own_artwork = db.Column(db.Integer, nullable=True)  # Converted from yes/no
    supported_artists = db.Column(db.Integer, nullable=True)  # Converted from yes/no


class UserScrollPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    phase_id = db.Column(db.Integer, nullable=False)
    scroll_id = db.Column(db.Integer, nullable=False)
    panel_id = db.Column(db.Integer, nullable=False)
    image_id = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    review = db.Column(db.String(255), nullable=True)


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


@app.route('/userArtPreferences', methods=['POST'])
def add_user_art_preferences():
    data = request.get_json()

    new_user_art_preferences = UserArtPreferences(
        user_id=data['userId'],
        artists=data['artists'],
        rural_art_interest_degree=map_degree_level(data['ruralArtInterestDegree']),
        south_asian_mythology_familiarity=map_familiarity_level(data['southAsianMythologyFamiliarity']),
        south_american_mythology_familiarity=map_familiarity_level(data['southAmericanMythologyFamiliarity']),
        greek_mythology_familiarity=map_familiarity_level(data['greekMythologyFamiliarity']),
        roman_mythology_familiarity=map_familiarity_level(data['romanMythologyFamiliarity']),
        australian_mythology_familiarity=map_familiarity_level(data['australianMythologyFamiliarity']),
        african_mythology_familiarity=map_familiarity_level(data['africanMythologyFamiliarity']),
        rajasthan_familiarity=map_familiarity_level(data['rajasthanFamiliarity']),
        west_bengal_familiarity=map_familiarity_level(data['westBengalFamiliarity']),
        andra_pradesh_familiarity=map_familiarity_level(data['andraPradeshFamiliarity']),
        chattisgarh_familiarity=map_familiarity_level(data['chattisgarhFamiliarity']),
        maharashtra_familiarity=map_familiarity_level(data['maharashtraFamiliarity']),
        gujrat_familiarity=map_familiarity_level(data['gujratFamiliarity']),
        karnataka_familiarity=map_familiarity_level(data['karnatakaFamiliarity']),
        tamil_nadu_familiarity=map_familiarity_level(data['tamilNaduFamiliarity']),
        indonesia_familiarity=map_familiarity_level(data['indonesiaFamiliarity']),
        bali_java_borneo_familiarity=map_familiarity_level(data['baliJavaBorneoFamiliarity']),
        chinese_familiarity=map_familiarity_level(data['chineseFamiliarity']),
        persian_familiarity=map_familiarity_level(data['persianFamiliarity']),
        tibetan_familiarity=map_familiarity_level(data['tibetanFamiliarity']),
        srilankan_familiarity=map_familiarity_level(data['srilankanFamiliarity']),
        scroll_length_preference=map_scroll_length(data['scrollLengthPreference']),
        art_books_reading_frequency=map_reading_frequency(data['artBooksReadingFrequency']),
        classical_familiarity=map_familiarity_level(data['classicalFamiliarity']),
        sanskrit_chants_familiarity=map_familiarity_level(data['sanskritChantsFamiliarity']),
        folk_music_familiarity=map_familiarity_level(data['folkMusicFamiliarity']),
        rock_familiarity=map_familiarity_level(data['rockFamiliarity']),
        hip_hop_familiarity=map_familiarity_level(data['hipHopFamiliarity']),
        bhajans_familiarity=map_familiarity_level(data['bhajansFamiliarity']),
        epic_familiarity=map_familiarity_level(data['epicFamiliarity']),
        folk_tales_familiarity=map_familiarity_level(data['folkTalesFamiliarity']),
        sanskrit_texts_familiarity=map_familiarity_level(data['sanskritTextsFamiliarity']),
        religious_literature_familiarity=map_familiarity_level(data['religiousLiteratureFamiliarity']),
        watched_performances=map_boolean_to_int(data['watchedPerformances']),
        visited_museum=map_boolean_to_int(data['visitedMuseum']),
        own_artwork=map_boolean_to_int(data['ownArtwork']),
        supported_artists=map_boolean_to_int(data['supportedArtists'])
    )
    try:
        db.session.add(new_user_art_preferences)
        db.session.commit()

        return jsonify({
            'message': 'User Art Preferences created successfully',
            'userArtPreferencesId': new_user_art_preferences.id
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


def map_familiarity_level(level: str) -> int:
    levels = {
        "beginner": 1,
        "intermediate": 2,
        "advanced": 3,
        "expert": 4
    }
    return levels.get(level.lower(), None)


def map_scroll_length(length: str) -> int:
    scrolls = {
        "short": 1,
        "long": 2
    }
    return scrolls.get(length.lower(), None)


def map_boolean_to_int(answer: str) -> int:
    answers = {
        "yes": 1,
        "no": 0
    }
    return answers.get(answer.lower(), None)


def map_degree_level(level: str) -> int:
    levels = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "very-high": 4
    }

    return levels.get(level.lower(), None)


def map_reading_frequency(level: str) -> int:
    levels = {
        "monthly": 1,
        "sixMonths": 2,
        "annually": 3,
        "never": 0
    }
    return levels.get(level.lower(), None)


@app.route('/userScrollPreferences', methods=['POST'])
def add_user_scroll_preferences():
    data = request.get_json()
    new_user_scroll_preferences = UserScrollPreferences(
        user_id=data['userId'],
        phase_id=data['phaseId'],
        scroll_id=data['scrollId'],
        panel_id=data['panelId'],
        image_id=data['imageId'],
        rating=data['rating'],
        review=data['review']
    )
    try:
        db.session.add(new_user_scroll_preferences)
        db.session.commit()

        return jsonify({
            'message': 'User Scroll Preferences created successfully',
            'userScrollPreferencesId': new_user_scroll_preferences.id
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
        return jsonify({'message': 'Login successful', 'userId': user.id})
    return jsonify({'error': 'Invalid username or password'}), 401


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
