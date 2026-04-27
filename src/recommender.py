from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv
import logging

logger = logging.getLogger(__name__)


MOOD_SIMILARITY = {
    "focused": {"chill", "relaxed", "moody"},
    "chill": {"focused", "relaxed", "moody"},
    "relaxed": {"chill", "focused", "moody"},
    "happy": {"intense"},
    "intense": {"happy"},
    "moody": {"chill", "relaxed", "focused"},
}

GENRE_SIMILARITY = {
    "lofi": {"ambient", "jazz"},
    "ambient": {"lofi"},
    "jazz": {"lofi"},
    "electronic": {"synthwave", "pop"},
    "synthwave": {"electronic", "pop"},
    "pop": {"synthwave", "electronic"},
    "rock": {"electronic"},
}


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
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                song = {
                    'id': int(row['id']),
                    'title': row['title'],
                    'artist': row['artist'],
                    'genre': row['genre'],
                    'mood': row['mood'],
                    'energy': float(row['energy']),
                    'tempo_bpm': float(row['tempo_bpm']),
                    'valence': float(row['valence']),
                    'danceability': float(row['danceability']),
                    'acousticness': float(row['acousticness'])
                }
                songs.append(song)
        
        logger.info(f"✓ Loaded {len(songs)} songs from {csv_path}")
        return songs
    
    except FileNotFoundError:
        logger.error(f"✗ Error: File not found at {csv_path}")
        return []
    except ValueError as e:
        logger.error(f"✗ Error converting numerical values: {e}")
        return []
    except KeyError as e:
        logger.error(f"✗ Error: Missing expected column {e}")
        return []


def score_song(user_prefs: Dict, song: Dict, inferred_confidence: Dict = None) -> Tuple[float, List[str], List[str]]:
    """
    Score one song against user preferences.
    Returns (score, reasons, assumptions_made) where assumptions_made tracks fallback defaults used.
    """
    if inferred_confidence is None:
        inferred_confidence = {}
    
    total_points = 0.0
    reasons = []
    assumptions = []
    
    # Use inferred preferences with confidence scores, or defaults
    target_genre = user_prefs.get('genre')
    genre_confidence = inferred_confidence.get('genre_confidence', 1.0)
    genre_is_inferred = inferred_confidence.get('genre_confidence', 0.0) > 0
    
    target_mood = user_prefs.get('mood')
    mood_confidence = inferred_confidence.get('mood_confidence', 1.0)
    mood_is_inferred = inferred_confidence.get('mood_confidence', 0.0) > 0
    
    target_energy = user_prefs.get('energy', 0.5)
    energy_confidence = inferred_confidence.get('energy_confidence', 1.0)
    energy_is_inferred = inferred_confidence.get('energy_confidence', 0.0) > 0
    
    likes_acoustic = user_prefs.get('likes_acoustic', False)
    acoustic_confidence = inferred_confidence.get('acoustic_confidence', 1.0)
    acoustic_is_inferred = inferred_confidence.get('acoustic_confidence', 0.0) > 0
    
    song_genre = song['genre'].lower()
    song_mood = song['mood'].lower()

    # ============ GENRE MATCH (max +3.0, +1.5 for related genres) ============
    if target_genre and song_genre == target_genre.lower():
        genre_points = 3.0
        total_points += genre_points
        confidence_label = f"(confidence: {genre_confidence:.2f})" if genre_is_inferred else "(from profile)"
        reasons.append(f"genre match: {song['genre']} {confidence_label} (+{genre_points})")
    elif target_genre and song_genre in GENRE_SIMILARITY.get(target_genre.lower(), set()):
        genre_points = 1.5
        total_points += genre_points
        confidence_label = f"(confidence: {genre_confidence:.2f})" if genre_is_inferred else "(from profile)"
        reasons.append(f"related genre fit: {song['genre']} ~ {target_genre} {confidence_label} (+{genre_points})")
    elif not target_genre:
        assumptions.append("genre preference not specified, using default")
    
    # ============ MOOD MATCH (max +2.0, +1.2 for related moods) ============
    if target_mood and song_mood == target_mood.lower():
        mood_points = 2.0
        total_points += mood_points
        confidence_label = f"(confidence: {mood_confidence:.2f})" if mood_is_inferred else "(from profile)"
        reasons.append(f"mood match: {song['mood']} {confidence_label} (+{mood_points})")
    elif target_mood and song_mood in MOOD_SIMILARITY.get(target_mood.lower(), set()):
        mood_points = 1.2
        total_points += mood_points
        confidence_label = f"(confidence: {mood_confidence:.2f})" if mood_is_inferred else "(from profile)"
        reasons.append(f"related mood fit: {song['mood']} ~ {target_mood} {confidence_label} (+{mood_points})")
    elif not target_mood:
        assumptions.append("mood preference not specified, using default")
    
    # ============ ENERGY FIT (max +2.0, distance-based) ============
    energy_diff = abs(song['energy'] - target_energy)
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
        confidence_label = f"(confidence: {energy_confidence:.2f})" if energy_is_inferred else "(from profile)"
        reasons.append(f"energy fit: {song['energy']:.2f} vs target {target_energy:.2f} {confidence_label} (+{energy_points})")
    
    if not energy_is_inferred:
        assumptions.append("energy used default value 0.5")
    
    # ============ ACOUSTIC PREFERENCE (max +1.5) ============
    if likes_acoustic:
        if song['acousticness'] > 0.7:
            acoustic_points = 1.5
            total_points += acoustic_points
            confidence_label = f"(confidence: {acoustic_confidence:.2f})" if acoustic_is_inferred else "(from profile)"
            reasons.append(f"acoustic preference: highly acoustic {confidence_label} (+{acoustic_points})")
    else:
        if song['acousticness'] < 0.3:
            acoustic_points = 1.5
            total_points += acoustic_points
            confidence_label = f"(confidence: {acoustic_confidence:.2f})" if acoustic_is_inferred else "(from profile)"
            reasons.append(f"acoustic preference: produced/electric {confidence_label} (+{acoustic_points})")
    
    # ============ SECONDARY FEATURES BONUS (max +1.5) ============
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
    
    return (normalized_score, reasons, assumptions)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, inferred_confidence: Dict = None) -> List[Tuple[Dict, float, str]]:
    """
    Rank songs by score and return top-k recommendations with confidence-aware explanations.
    
    inferred_confidence: Dict with keys like 'genre_confidence', 'mood_confidence', etc.
    """
    if inferred_confidence is None:
        inferred_confidence = {}
    
    # Score every song
    scored_songs = []
    for song in songs:
        score, reasons, assumptions = score_song(user_prefs, song, inferred_confidence)
        scored_songs.append((song, score, reasons, assumptions))
    
    # Sort by score (highest to lowest)
    sorted_songs = sorted(scored_songs, key=lambda x: x[1], reverse=True)
    
    # Take top k and format output
    recommendations = []
    for song, score, reasons, assumptions in sorted_songs[:k]:
        explanation = " | ".join(reasons)
        
        # Append assumption notes if any
        if assumptions:
            assumption_note = " [Assumptions: " + ", ".join(assumptions) + "]"
            explanation += assumption_note
        
        recommendations.append((song, score, explanation))
    
    logger.info(f"Generated {len(recommendations)} recommendations using preferences: {user_prefs}")
    return recommendations

