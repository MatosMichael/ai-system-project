"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    # Lofi listener profile
    lofi_profile = {"genre": "lofi", "mood":
                     "chill", "energy": 0.40, "likes_acoustic": True}

    recommendations = recommend_songs(lofi_profile, songs, k=5)

    print("\n" + "="*70)
    print("🎵 TOP 5 MUSIC RECOMMENDATIONS 🎵".center(70))
    print("="*70 + "\n")
    
    for idx, rec in enumerate(recommendations, start=1):
        # Unpack recommendation tuple
        song, score, explanation = rec
        
        # Color-code score by range
        if score >= 0.9:
            score_indicator = "🟢 EXCELLENT"
        elif score >= 0.75:
            score_indicator = "🟡 GOOD"
        elif score >= 0.5:
            score_indicator = "🟠 FAIR"
        else:
            score_indicator = "🔴 POOR"
        
        # Print recommendation header
        print(f"#{idx}. {song['title'].upper()}")
        print(f"    Artist: {song['artist']} | Genre: {song['genre']}")
        print(f"    Score: {score:.2f}/1.00 {score_indicator}")
        
        # Parse and display reasons with proper indentation
        reasons = explanation.split(" | ")
        print(f"    Reasons:")
        for reason in reasons:
            print(f"      ✓ {reason}")
        
        # Print separator
        print("-" * 70 + "\n")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
