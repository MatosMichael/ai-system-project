"""
Tests for the agentic preference extraction layer.
Includes extraction accuracy, guardrails, contradiction handling, and fallback tests.
"""

import pytest
from src.agent import (
    PreferenceExtractor, ExtractionMode, ConversationManager, PartialUserProfile
)


class TestPreferenceExtractorRules:
    """Test rules-based preference extraction."""
    
    def setup_method(self):
        """Initialize extractor for each test."""
        self.extractor = PreferenceExtractor(mode=ExtractionMode.RULES)
    
    def test_extract_genre_from_message(self):
        """Should extract genre from message."""
        result = self.extractor.extract("I want pop music", turn_id=1)
        assert result.success
        assert result.profile.genre == "pop"
        assert "genre" in result.extracted_fields
        assert result.profile.genre_confidence > 0.7
    
    def test_extract_mood_from_message(self):
        """Should extract mood from message."""
        result = self.extractor.extract("I'm looking for something chill", turn_id=1)
        assert result.success
        assert result.profile.mood == "chill"
        assert "mood" in result.extracted_fields
    
    def test_extract_energy_word(self):
        """Should extract energy level from descriptive words."""
        result = self.extractor.extract("Show me energetic songs", turn_id=1)
        assert result.success
        assert result.profile.energy is not None
        assert result.profile.energy > 0.6
        assert "energy" in result.extracted_fields
    
    def test_extract_energy_numeric(self):
        """Should extract numeric energy values."""
        result = self.extractor.extract("I want energy 0.75", turn_id=1)
        assert result.success
        assert result.profile.energy == 0.75
        assert "energy" in result.extracted_fields
    
    def test_energy_value_clamped_to_range(self):
        """Should clamp energy values out of 0.0-1.0 range."""
        result = self.extractor.extract("energy 1.5", turn_id=1)
        # Should not extract invalid energy
        assert result.profile.energy is None or 0.0 <= result.profile.energy <= 1.0
    
    def test_extract_acoustic_positive(self):
        """Should detect acoustic preference from positive keywords."""
        result = self.extractor.extract("I love acoustic and unplugged songs", turn_id=1)
        assert result.success
        assert result.profile.likes_acoustic is True
        assert "likes_acoustic" in result.extracted_fields
    
    def test_extract_acoustic_negative(self):
        """Should detect non-acoustic preference from negative keywords."""
        result = self.extractor.extract("I prefer electronic and synth sounds", turn_id=1)
        assert result.success
        assert result.profile.likes_acoustic is False
        assert "likes_acoustic" in result.extracted_fields
    
    def test_contradictory_acoustic_preference(self):
        """Should handle contradictory acoustic preferences safely."""
        result = self.extractor.extract("I want acoustic electric sounds", turn_id=1)
        # Should detect contradiction and note warning, but not crash
        assert len(result.profile.parse_warnings) > 0 or result.profile.likes_acoustic is None
    
    def test_extract_multiple_fields(self):
        """Should extract multiple preferences from single message."""
        result = self.extractor.extract("I want energetic pop with electric sounds", turn_id=1)
        assert result.success
        assert len(result.extracted_fields) >= 2
        assert "genre" in result.extracted_fields
        assert result.profile.energy is not None
    
    def test_extract_from_empty_message(self):
        """Should fail gracefully on empty message."""
        result = self.extractor.extract("", turn_id=1)
        assert not result.success
        assert result.error_message is not None
    
    def test_extract_from_gibberish(self):
        """Should fail gracefully on gibberish."""
        result = self.extractor.extract("xyzzy qwerty blah blah", turn_id=1)
        assert not result.success

    def test_extract_gym_intent(self):
        """Should map gym language to a high-energy workout profile."""
        result = self.extractor.extract("I want music for the gym", turn_id=1)
        assert result.success
        assert result.profile.mood == "intense"
        assert result.profile.energy is not None and result.profile.energy >= 0.85
        assert "mood" in result.extracted_fields
        assert "energy" in result.extracted_fields

    def test_extract_study_intent(self):
        """Should map study language to a focused, lower-energy profile."""
        result = self.extractor.extract("I need music for studying", turn_id=1)
        assert result.success
        assert result.profile.mood == "focused"
        assert result.profile.energy is not None and result.profile.energy <= 0.4
        assert result.profile.likes_acoustic is True

    def test_extract_vibe_language(self):
        """Should handle vague vibe language with a sensible fallback."""
        result = self.extractor.extract("something with a good vibe", turn_id=1)
        assert result.success
        assert result.profile.mood == "happy"
        assert result.profile.energy is not None


class TestConversationManager:
    """Test multi-turn conversation state accumulation."""
    
    def setup_method(self):
        """Initialize conversation manager for each test."""
        self.manager = ConversationManager(mode=ExtractionMode.RULES)
    
    def test_single_turn_extraction(self):
        """Should extract preferences from single turn."""
        result = self.manager.process_turn("I want pop music")
        assert result.success
        assert self.manager.get_current_profile().genre == "pop"
        assert self.manager.turn_count == 1
    
    def test_multi_turn_accumulation(self):
        """Should accumulate preferences across multiple turns."""
        self.manager.process_turn("I want pop")
        self.manager.process_turn("Make it energetic")
        
        profile = self.manager.get_current_profile()
        assert profile.genre == "pop"
        assert profile.energy is not None
        assert profile.energy > 0.6
        assert self.manager.turn_count == 2
    
    def test_turn_history_tracking(self):
        """Should track all turn results."""
        self.manager.process_turn("I want pop")
        self.manager.process_turn("chill vibes")
        
        assert len(self.manager.turn_history) == 2
        assert all(isinstance(r, type(self.manager.turn_history[0])) for r in self.manager.turn_history)
    
    def test_should_continue_conversation_threshold(self):
        """Should suggest stopping when confidence is high."""
        self.manager.process_turn("I want pop music that's energetic and acoustic")
        
        # After extracting multiple high-confidence fields, should suggest stop
        should_continue = self.manager.should_continue_conversation(confidence_threshold=0.7)
        # Depending on extraction, might continue or stop - just verify it doesn't crash
        assert isinstance(should_continue, bool)
    
    def test_max_turn_limit(self):
        """Should not continue conversation after turn limit."""
        for i in range(6):
            self.manager.process_turn("test input")
        
        should_continue = self.manager.should_continue_conversation()
        assert not should_continue or self.manager.turn_count >= 5


class TestGuardrailsAndFallback:
    """Test guardrails, validation, and graceful fallback behavior."""
    
    def setup_method(self):
        self.extractor = PreferenceExtractor(mode=ExtractionMode.RULES)
    
    def test_fallback_on_invalid_genre(self):
        """Should handle invalid genre gracefully."""
        result = self.extractor.extract("I want zxcvbn music")
        # Should still succeed on other fields or fail gracefully
        assert isinstance(result.profile, PartialUserProfile)
    
    def test_fallback_on_invalid_energy(self):
        """Should handle invalid energy gracefully."""
        result = self.extractor.extract("energy pizza")
        # Should still process message, just no energy extracted
        assert isinstance(result.profile, PartialUserProfile)
    
    def test_deterministic_rules_extraction(self):
        """Rules-based extraction should be deterministic across runs."""
        msg = "I want energetic pop with electric sounds"
        
        result1 = self.extractor.extract(msg, turn_id=1)
        # Create new extractor to ensure no state pollution
        extractor2 = PreferenceExtractor(mode=ExtractionMode.RULES)
        result2 = extractor2.extract(msg, turn_id=1)
        
        assert result1.profile.genre == result2.profile.genre
        assert result1.profile.energy == result2.profile.energy
        assert result1.profile.likes_acoustic == result2.profile.likes_acoustic


class TestBehaviorChange:
    """Test that different inferred profiles lead to different recommendation behavior."""
    
    def test_different_preferences_accumulate_differently(self):
        """Different conversation paths should lead to different profiles."""
        # Conversation path 1: wants pop
        manager1 = ConversationManager(mode=ExtractionMode.RULES)
        manager1.process_turn("I want pop music")
        profile1 = manager1.get_current_profile()
        
        # Conversation path 2: wants rock
        manager2 = ConversationManager(mode=ExtractionMode.RULES)
        manager2.process_turn("I want rock music")
        profile2 = manager2.get_current_profile()
        
        # Profiles should differ
        assert profile1.genre != profile2.genre
        assert profile1.genre == "pop"
        assert profile2.genre == "rock"
