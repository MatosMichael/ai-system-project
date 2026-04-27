from src.recommender import Song, UserProfile, Recommender

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_behavior_change_with_inferred_profiles():
    """CRITICAL: Prove AI inference actually changes recommendation output."""
    from src.recommender import recommend_songs
    
    songs = [
        {
            'id': 1, 'title': "Upbeat Pop", 'artist': "Artist A", 'genre': "pop",
            'mood': "happy", 'energy': 0.9, 'tempo_bpm': 130, 'valence': 0.85,
            'danceability': 0.8, 'acousticness': 0.1
        },
        {
            'id': 2, 'title': "Chill Lofi", 'artist': "Artist B", 'genre': "lofi",
            'mood': "chill", 'energy': 0.3, 'tempo_bpm': 70, 'valence': 0.5,
            'danceability': 0.4, 'acousticness': 0.8
        },
    ]
    
    # Profile A: High energy preference
    prefs_a = {'genre': 'pop', 'mood': 'happy', 'energy': 0.9, 'likes_acoustic': False}
    confidence_a = {'genre_confidence': 0.9, 'mood_confidence': 0.9, 'energy_confidence': 0.9, 'acoustic_confidence': 0.9}
    
    recs_a = recommend_songs(prefs_a, songs, k=2, inferred_confidence=confidence_a)
    top_song_a = recs_a[0][0]['title']
    
    # Profile B: Low energy preference
    prefs_b = {'genre': 'lofi', 'mood': 'chill', 'energy': 0.3, 'likes_acoustic': True}
    confidence_b = {'genre_confidence': 0.9, 'mood_confidence': 0.9, 'energy_confidence': 0.9, 'acoustic_confidence': 0.9}
    
    recs_b = recommend_songs(prefs_b, songs, k=2, inferred_confidence=confidence_b)
    top_song_b = recs_b[0][0]['title']
    
    # ASSERTION: Different preferences MUST produce different top recommendations
    assert top_song_a != top_song_b, f"Different profiles should produce different top songs, but both gave: {top_song_a}"
    assert top_song_a == "Upbeat Pop"
    assert top_song_b == "Chill Lofi"


def test_confidence_aware_explanations_include_fallbacks():
    """Verify that explanations show both inferred fields and fallback assumptions."""
    from src.recommender import recommend_songs
    
    songs = [{
        'id': 1, 'title': "Test Song", 'artist': "Test", 'genre': "pop",
        'mood': "happy", 'energy': 0.8, 'tempo_bpm': 120, 'valence': 0.8,
        'danceability': 0.7, 'acousticness': 0.2
    }]
    
    # Only genre is inferred; energy, mood fallback to defaults
    prefs = {'genre': 'pop', 'mood': None, 'energy': 0.5, 'likes_acoustic': False}
    confidence = {'genre_confidence': 0.9, 'mood_confidence': 0.0, 'energy_confidence': 0.0, 'acoustic_confidence': 0.0}
    
    recs = recommend_songs(prefs, songs, k=1, inferred_confidence=confidence)
    explanation = recs[0][2]
    
    # Should mention assumptions about defaults
    assert "Assumptions:" in explanation or "default" in explanation.lower() or "not specified" in explanation.lower()


def test_deterministic_ranking_for_fixed_input():
    """Verify that same input produces same ranking across runs."""
    from src.recommender import recommend_songs
    
    songs = [
        {'id': 1, 'title': "Song A", 'artist': "Artist", 'genre': "pop", 'mood': "happy", 'energy': 0.8, 'tempo_bpm': 120, 'valence': 0.8, 'danceability': 0.7, 'acousticness': 0.2},
        {'id': 2, 'title': "Song B", 'artist': "Artist", 'genre': "pop", 'mood': "happy", 'energy': 0.7, 'tempo_bpm': 110, 'valence': 0.7, 'danceability': 0.6, 'acousticness': 0.2},
    ]
    
    prefs = {'genre': 'pop', 'mood': 'happy', 'energy': 0.8, 'likes_acoustic': False}
    confidence = {'genre_confidence': 0.9, 'mood_confidence': 0.9, 'energy_confidence': 0.9, 'acoustic_confidence': 0.9}
    
    # Run twice with same input
    recs1 = recommend_songs(prefs, songs, k=2, inferred_confidence=confidence)
    recs2 = recommend_songs(prefs, songs, k=2, inferred_confidence=confidence)
    
    # Should produce identical rankings
    assert recs1[0][0]['id'] == recs2[0][0]['id']
    assert recs1[1][0]['id'] == recs2[1][0]['id']
