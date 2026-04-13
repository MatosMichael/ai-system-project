from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return typed song dictionaries."""
    songs = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # DictReader automatically uses first row as column headers
            reader = csv.DictReader(csvfile)
            
            # Read each row and convert numerical types
            for row in reader:
                song = {
                    'id': int(row['id']),                    # Integer for ID
                    'title': row['title'],                   # String
                    'artist': row['artist'],                 # String
                    'genre': row['genre'],                   # String
                    'mood': row['mood'],                     # String
                    'energy': float(row['energy']),          # Float for math
                    'tempo_bpm': float(row['tempo_bpm']),    # Float for math
                    'valence': float(row['valence']),        # Float for math
                    'danceability': float(row['danceability']),  # Float for math
                    'acousticness': float(row['acousticness'])   # Float for math
                }
                songs.append(song)
        
        print(f"✓ Loaded {len(songs)} songs from {csv_path}")
        return songs
    
    except FileNotFoundError:
        print(f"✗ Error: File not found at {csv_path}")
        return []
    except ValueError as e:
        print(f"✗ Error converting numerical values: {e}")
        return []
    except KeyError as e:
        print(f"✗ Error: Missing expected column {e}")
        return []

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences and return score plus reasons."""
    total_points = 0.0
    reasons = []
    
    # ============ GENRE MATCH (max +3.0) ============
    if song['genre'].lower() == user_prefs['genre'].lower():
        genre_points = 3.0
        total_points += genre_points
        reasons.append(f"genre match: {song['genre']} (+{genre_points})")
    
    # ============ MOOD MATCH (max +2.0) ============
    if song['mood'].lower() == user_prefs['mood'].lower():
        mood_points = 2.0
        total_points += mood_points
        reasons.append(f"mood match: {song['mood']} (+{mood_points})")
    
    # ============ ENERGY FIT (max +2.0, distance-based) ============
    energy_diff = abs(song['energy'] - user_prefs['energy'])
    if energy_diff < 0.1:
        energy_points = 2.0
    elif energy_diff < 0.2:
        energy_points = 1.5
    elif energy_diff < 0.3:
        energy_points = 1.0
    elif energy_diff < 0.5:
        energy_points = 0.5
    else:
        energy_points = 0.0
    
    if energy_points > 0:
        total_points += energy_points
        reasons.append(f"energy fit: {song['energy']:.2f} vs target {user_prefs['energy']:.2f} (+{energy_points})")
    
    # ============ ACOUSTIC PREFERENCE (max +1.5) ============
    if user_prefs['likes_acoustic']:
        # User likes acoustic: reward high acousticness (>0.7)
        if song['acousticness'] > 0.7:
            acoustic_points = 1.5
            total_points += acoustic_points
            reasons.append(f"acoustic preference: highly acoustic (+{acoustic_points})")
    else:
        # User dislikes acoustic: reward low acousticness (<0.3)
        if song['acousticness'] < 0.3:
            acoustic_points = 1.5
            total_points += acoustic_points
            reasons.append(f"acoustic preference: produced/electric sound (+{acoustic_points})")
    
    # ============ SECONDARY FEATURES BONUS (max +1.5) ============
    # Valence (0-1: sad to happy) + Danceability (0-1: not danceable to very danceable)
    avg_engagement = (song['valence'] + song['danceability']) / 2.0
    if avg_engagement > 0.7:
        secondary_points = 1.5
        engagement_label = "highly engaging"
    elif avg_engagement > 0.5:
        secondary_points = 1.0
        engagement_label = "moderately engaging"
    else:
        secondary_points = 0.5
        engagement_label = "subdued/introspective"
    
    total_points += secondary_points
    reasons.append(f"engagement bonus: {engagement_label} (+{secondary_points})")
    
    # ============ NORMALIZE TO 0.0 - 1.0 SCALE ============
    max_points = 10.0
    normalized_score = min(total_points / max_points, 1.0)
    
    return (normalized_score, reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank songs by score and return top-k recommendations with explanations."""
    # ============ STEP 1 & 2: Score every song ============
    # Create tuples of (song, score, reasons_list) for all songs
    scored_songs = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored_songs.append((song, score, reasons))
    
    # ============ STEP 3: Sort by score (highest to lowest) ============
    # Using sorted() to preserve original song list, sorted() returns new sorted list
    # key=lambda x: x[1] tells sorted() to sort by the score (second element)
    # reverse=True sorts in descending order (highest scores first)
    sorted_songs = sorted(scored_songs, key=lambda x: x[1], reverse=True)
    
    # ============ STEP 4: Take top k and format output ============
    recommendations = []
    for song, score, reasons in sorted_songs[:k]:
        # Format reasons into a readable explanation string
        # Join all reasons with ", " separator
        explanation = " | ".join(reasons)
        recommendations.append((song, score, explanation))
    
    return recommendations

