"""
Command line runner for the Agentic Music Recommender.
This is the PRIMARY execution path - conversational preference discovery is the default.
"""

import logging
import sys
from src.recommender import load_songs, recommend_songs
from src.agent import ConversationManager, ExtractionMode, PartialUserProfile

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('recommender.log')
    ]
)
logger = logging.getLogger(__name__)


def convert_partial_to_full_prefs(partial: PartialUserProfile) -> dict:
    """Convert PartialUserProfile to full preferences dict with defaults."""
    return {
        'genre': partial.genre or 'pop',  # Default fallback
        'mood': partial.mood or 'happy',  # Default fallback
        'energy': partial.energy if partial.energy is not None else 0.5,  # Default 0.5
        'likes_acoustic': partial.likes_acoustic if partial.likes_acoustic is not None else False,
    }


def get_confidence_dict(partial: PartialUserProfile) -> dict:
    """Extract confidence scores from PartialUserProfile."""
    return {
        'genre_confidence': partial.genre_confidence,
        'mood_confidence': partial.mood_confidence,
        'energy_confidence': partial.energy_confidence,
        'acoustic_confidence': partial.acoustic_confidence,
    }


def print_separator(title: str = ""):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'='*70}\n  {title}\n{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def main() -> None:
    """Main entry point: conversational recommendation system."""
    
    logger.info("="*70)
    logger.info("AGENTIC MUSIC RECOMMENDER - Starting")
    logger.info("="*70)
    
    # Load songs
    songs = load_songs("data/songs.csv")
    if not songs:
        logger.error("Failed to load songs. Exiting.")
        return
    
    # Initialize conversation manager with rules-based extraction (reproducible, no API key needed)
    # To use LLM-based extraction, set: mode=ExtractionMode.LLM (requires ANTHROPIC_API_KEY env var)
    conv_manager = ConversationManager(mode=ExtractionMode.RULES)
    
    print_separator("🎵 AGENTIC MUSIC RECOMMENDER 🎵")
    print("  Describe your music preferences in natural language.")
    print("  Examples: 'I want upbeat pop songs' or 'Show me chill lofi tracks'")
    print("  Type 'quit' or 'done' to generate recommendations.")
    print("  Type 'reset' to start over.")
    print_separator()
    
    # Multi-turn conversation loop
    turn_limit = 5
    confidence_threshold = 0.6
    
    while conv_manager.turn_count < turn_limit:
        current_profile = conv_manager.get_current_profile()
        
        # Show current understanding
        understood_fields = []
        if current_profile.genre:
            understood_fields.append(f"genre={current_profile.genre}")
        if current_profile.mood:
            understood_fields.append(f"mood={current_profile.mood}")
        if current_profile.energy is not None:
            understood_fields.append(f"energy={current_profile.energy:.2f}")
        if current_profile.likes_acoustic is not None:
            acoustic_str = "acoustic" if current_profile.likes_acoustic else "electric"
            understood_fields.append(f"prefers {acoustic_str}")
        
        if understood_fields:
            print(f"  Currently understood: {', '.join(understood_fields)}")
        else:
            print("  [No preferences extracted yet]")
        
        print()
        user_input = input("You: ").strip()
        
        # Handle commands
        if user_input.lower() in ['quit', 'done', 'q']:
            logger.info("User requested recommendations to be generated")
            break
        elif user_input.lower() == 'reset':
            logger.info("User reset conversation")
            conv_manager = ConversationManager(mode=ExtractionMode.RULES)
            print("\n✓ Conversation reset. Start over!\n")
            continue
        elif not user_input:
            print("  [Please enter something or type 'quit' to see recommendations]\n")
            continue
        
        # Process user input
        result = conv_manager.process_turn(user_input)
        
        if result.success:
            print(f"\n  ✓ Extracted: {', '.join(result.extracted_fields)}")
            if result.profile.parse_warnings:
                for warning in result.profile.parse_warnings:
                    print(f"  ⚠ {warning}")
        else:
            print(f"\n  ✗ Could not extract preferences: {result.error_message}")
        
        # Keep the session interactive until the user explicitly finishes.
        # We still surface readiness so users know recommendations are available.
        if not conv_manager.should_continue_conversation(confidence_threshold):
            print("\n  ✓ I have enough preference signal now. Type 'done' to generate recommendations, or add more details.\n")
            logger.info("Confidence threshold reached; waiting for explicit user finish command")
        
        print()
    
    # Generate recommendations
    final_profile = conv_manager.get_current_profile()
    full_prefs = convert_partial_to_full_prefs(final_profile)
    confidence_dict = get_confidence_dict(final_profile)
    
    logger.info(f"Final inferred profile: {full_prefs} | Confidence: {confidence_dict}")
    
    recommendations = recommend_songs(full_prefs, songs, k=5, inferred_confidence=confidence_dict)
    
    # Display results
    print_separator("🎵 TOP 5 MUSIC RECOMMENDATIONS 🎵")
    
    for idx, rec in enumerate(recommendations, start=1):
        song, score, explanation = rec
        
        if score >= 0.85:
            score_indicator = "🟢 EXCELLENT"
        elif score >= 0.65:
            score_indicator = "🟡 GOOD"
        elif score >= 0.40:
            score_indicator = "🟠 FAIR"
        else:
            score_indicator = "🔴 POOR"
        
        print(f"#{idx}. {song['title'].upper()}")
        print(f"    Artist: {song['artist']} | Genre: {song['genre']}")
        print(f"    Score: {score:.2f}/1.00 {score_indicator}")
        print(f"    Why: {explanation}")
        print()
    
    print_separator()
    logger.info("AGENTIC MUSIC RECOMMENDER - Complete")


if __name__ == "__main__":
    main()
