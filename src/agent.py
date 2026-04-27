"""
Agentic preference extraction layer.
Converts conversational user input into structured, confidence-scored preferences.
Includes guardrails, validation, and fallback extraction logic.
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Any
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ExtractionMode(Enum):
    """Extraction strategy: LLM-based or rules-based."""
    LLM = "llm"
    RULES = "rules"


@dataclass
class PartialUserProfile:
    """User preferences extracted from conversation, with confidence scores."""
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy: Optional[float] = None
    likes_acoustic: Optional[bool] = None
    
    # Confidence scores (0.0 to 1.0)
    genre_confidence: float = 0.0
    mood_confidence: float = 0.0
    energy_confidence: float = 0.0
    acoustic_confidence: float = 0.0
    
    # Extraction metadata
    turn_id: int = 0
    parse_warnings: List[str] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class ProfileInferenceResult:
    """Result of a single preference extraction attempt."""
    profile: PartialUserProfile
    extraction_mode: str  # "llm" or "rules"
    raw_input: str
    success: bool
    error_message: Optional[str] = None
    extracted_fields: List[str] = field(default_factory=list)


class PreferenceExtractor:
    """Extract structured preferences from free-form user messages."""
    
    VALID_GENRES = ["pop", "rock", "lofi", "ambient", "jazz", "synthwave", "electronic"]
    VALID_MOODS = ["happy", "chill", "intense", "moody", "relaxed", "focused"]
    VALID_ENERGY_RANGES = {
        "low": 0.2,
        "chill": 0.3,
        "moderate": 0.5,
        "energetic": 0.7,
        "high": 0.85
    }
    INTENT_PRESETS = {
        "gym": {"mood": "intense", "energy": 0.9, "likes_acoustic": False},
        "workout": {"mood": "intense", "energy": 0.9, "likes_acoustic": False},
        "exercise": {"mood": "intense", "energy": 0.9, "likes_acoustic": False},
        "training": {"mood": "intense", "energy": 0.9, "likes_acoustic": False},
        "study": {"mood": "focused", "energy": 0.4, "likes_acoustic": True},
        "focus": {"mood": "focused", "energy": 0.4, "likes_acoustic": True},
        "homework": {"mood": "focused", "energy": 0.4, "likes_acoustic": True},
        "reading": {"mood": "relaxed", "energy": 0.35, "likes_acoustic": True},
        "relax": {"mood": "relaxed", "energy": 0.3, "likes_acoustic": True},
        "chill": {"mood": "chill", "energy": 0.3, "likes_acoustic": True},
        "party": {"mood": "happy", "energy": 0.85, "likes_acoustic": False},
        "hype": {"mood": "intense", "energy": 0.9, "likes_acoustic": False},
        "drive": {"mood": "moody", "energy": 0.6, "likes_acoustic": False},
        "commute": {"mood": "moody", "energy": 0.55, "likes_acoustic": False},
    }
    
    def __init__(self, mode: ExtractionMode = ExtractionMode.RULES):
        """Initialize extractor with specified mode."""
        self.mode = mode
        self.extraction_attempts = []
        logger.info(f"PreferenceExtractor initialized with mode: {mode.value}")
    
    def extract(self, user_message: str, turn_id: int = 0) -> ProfileInferenceResult:
        """
        Extract preferences from user message.
        Falls back to rules-based extraction if LLM fails.
        """
        logger.info(f"Turn {turn_id}: Starting extraction | Input: {user_message[:80]}...")
        
        # Try LLM extraction first if enabled
        if self.mode == ExtractionMode.LLM:
            result = self._extract_with_llm(user_message, turn_id)
            if result.success:
                logger.info(f"Turn {turn_id}: LLM extraction successful | Fields: {result.extracted_fields}")
                return result
            else:
                logger.warning(f"Turn {turn_id}: LLM extraction failed, falling back to rules | Error: {result.error_message}")
        
        # Fall back to rules-based extraction
        result = self._extract_with_rules(user_message, turn_id)
        result.fallback_used = (self.mode == ExtractionMode.LLM and not result.success)
        
        if result.success:
            logger.info(f"Turn {turn_id}: Rules extraction successful | Fields: {result.extracted_fields}")
        else:
            logger.warning(f"Turn {turn_id}: Rules extraction failed | Error: {result.error_message}")
        
        return result

    def _record_extracted_field(self, extracted_fields: List[str], field_name: str) -> None:
        """Add a field name once, preserving deterministic output order."""
        if field_name not in extracted_fields:
            extracted_fields.append(field_name)

    def _apply_intent_presets(self, lower_msg: str, profile: PartialUserProfile, extracted_fields: List[str], turn_id: int) -> bool:
        """Apply the first matching intent preset to turn broad user language into usable slots."""
        for keyword, preset in self.INTENT_PRESETS.items():
            if keyword in lower_msg:
                if "mood" in preset:
                    profile.mood = preset["mood"]
                    profile.mood_confidence = max(profile.mood_confidence, 0.8)
                    self._record_extracted_field(extracted_fields, "mood")
                if "energy" in preset:
                    profile.energy = preset["energy"]
                    profile.energy_confidence = max(profile.energy_confidence, 0.8)
                    self._record_extracted_field(extracted_fields, "energy")
                if "likes_acoustic" in preset:
                    profile.likes_acoustic = preset["likes_acoustic"]
                    profile.acoustic_confidence = max(profile.acoustic_confidence, 0.8)
                    self._record_extracted_field(extracted_fields, "likes_acoustic")
                logger.debug(f"Turn {turn_id}: Detected intent preset '{keyword}' -> {preset}")
                return True
        return False
    
    def _extract_with_llm(self, user_message: str, turn_id: int) -> ProfileInferenceResult:
        """
        Extract preferences using LLM with schema validation.
        Uses Claude API with JSON schema constraints.
        """
        try:
            import anthropic
        except ImportError:
            return ProfileInferenceResult(
                profile=PartialUserProfile(turn_id=turn_id),
                extraction_mode="llm",
                raw_input=user_message,
                success=False,
                error_message="Anthropic library not installed. Install with: pip install anthropic"
            )
        
        try:
            client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
            
            prompt = f"""Extract music preferences from this user message. Return JSON with these fields:
            - genre (string, one of {self.VALID_GENRES}, or null)
            - mood (string, one of {self.VALID_MOODS}, or null)
            - energy (float 0.0-1.0, or null)
            - likes_acoustic (bool, or null)
            - genre_confidence (float 0.0-1.0)
            - mood_confidence (float 0.0-1.0)
            - energy_confidence (float 0.0-1.0)
            - acoustic_confidence (float 0.0-1.0)
            
            User message: "{user_message}"
            
            Return ONLY valid JSON, no markdown or extra text."""
            
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text
            
            # Parse and validate JSON
            extracted = json.loads(response_text)
            
            # Validate fields
            profile = self._validate_extracted_fields(extracted, turn_id)
            
            extracted_fields = [k for k, v in asdict(profile).items() 
                              if v is not None and not k.endswith("_confidence") and k != "turn_id" and k != "parse_warnings" and k != "fallback_used"]
            
            return ProfileInferenceResult(
                profile=profile,
                extraction_mode="llm",
                raw_input=user_message,
                success=True,
                extracted_fields=extracted_fields
            )
        
        except json.JSONDecodeError as e:
            return ProfileInferenceResult(
                profile=PartialUserProfile(turn_id=turn_id),
                extraction_mode="llm",
                raw_input=user_message,
                success=False,
                error_message=f"JSON decode error: {str(e)}"
            )
        except Exception as e:
            return ProfileInferenceResult(
                profile=PartialUserProfile(turn_id=turn_id),
                extraction_mode="llm",
                raw_input=user_message,
                success=False,
                error_message=f"API error: {str(e)}"
            )
    
    def _extract_with_rules(self, user_message: str, turn_id: int) -> ProfileInferenceResult:
        """
        Extract preferences using deterministic rules.
        Works offline and is guaranteed reproducible.
        """
        profile = PartialUserProfile(turn_id=turn_id)
        extracted_fields = []
        warnings = []
        
        lower_msg = user_message.lower()

        # High-level conversational intent presets turn broad requests into usable slots.
        if not self._apply_intent_presets(lower_msg, profile, extracted_fields, turn_id):
            # Fallback conversational synonyms for especially short requests.
            if "upbeat" in lower_msg:
                profile.mood = "happy"
                profile.mood_confidence = 0.75
                self._record_extracted_field(extracted_fields, "mood")
                if profile.energy is None:
                    profile.energy = 0.75
                    profile.energy_confidence = 0.75
                    self._record_extracted_field(extracted_fields, "energy")
                logger.debug(f"Turn {turn_id}: Detected upbeat preference -> mood=happy, energy=0.75")
            elif "vibe" in lower_msg:
                profile.mood = "happy"
                profile.mood_confidence = 0.65
                self._record_extracted_field(extracted_fields, "mood")
                if profile.energy is None:
                    profile.energy = 0.65
                    profile.energy_confidence = 0.65
                    self._record_extracted_field(extracted_fields, "energy")
                logger.debug(f"Turn {turn_id}: Detected vibe preference -> mood=happy, energy=0.65")
        
        # Genre detection
        for genre in self.VALID_GENRES:
            if genre in lower_msg:
                profile.genre = genre
                profile.genre_confidence = 0.85
                self._record_extracted_field(extracted_fields, "genre")
                logger.debug(f"Turn {turn_id}: Detected genre: {genre}")
                break
        
        # Mood detection
        for mood in self.VALID_MOODS:
            if mood in lower_msg:
                profile.mood = mood
                profile.mood_confidence = 0.85
                self._record_extracted_field(extracted_fields, "mood")
                logger.debug(f"Turn {turn_id}: Detected mood: {mood}")
                break
        
        # Energy detection
        for energy_word, energy_val in self.VALID_ENERGY_RANGES.items():
            if energy_word in lower_msg:
                profile.energy = energy_val
                profile.energy_confidence = 0.80
                self._record_extracted_field(extracted_fields, "energy")
                logger.debug(f"Turn {turn_id}: Detected energy level: {energy_word} ({energy_val})")
                break
        
        # Numeric energy detection (if user says "0.8 energy" or similar)
        energy_match = re.search(r'energy[:\s]+([0-1]\.?\d*)', lower_msg)
        if energy_match:
            try:
                energy_val = float(energy_match.group(1))
                if 0.0 <= energy_val <= 1.0:
                    profile.energy = energy_val
                    profile.energy_confidence = 0.90
                    self._record_extracted_field(extracted_fields, "energy")
                    logger.debug(f"Turn {turn_id}: Detected numeric energy: {energy_val}")
                else:
                    warnings.append(f"Energy value {energy_val} out of range [0.0, 1.0]")
            except ValueError:
                pass
        
        # Acoustic preference detection
        acoustic_positive = ["acoustic", "unplugged", "live"]
        acoustic_negative = ["electric", "produced", "electronic", "synth"]
        
        has_positive = any(word in lower_msg for word in acoustic_positive)
        has_negative = any(word in lower_msg for word in acoustic_negative)
        
        if has_positive and not has_negative:
            profile.likes_acoustic = True
            profile.acoustic_confidence = 0.85
            self._record_extracted_field(extracted_fields, "likes_acoustic")
            logger.debug(f"Turn {turn_id}: Detected acoustic preference: True")
        elif has_negative and not has_positive:
            profile.likes_acoustic = False
            profile.acoustic_confidence = 0.85
            self._record_extracted_field(extracted_fields, "likes_acoustic")
            logger.debug(f"Turn {turn_id}: Detected acoustic preference: False")
        elif has_positive and has_negative:
            warnings.append("Contradictory acoustic preferences detected; no preference set")
        
        profile.parse_warnings = warnings
        
        # Determine success: at least one field extracted
        success = len(extracted_fields) > 0
        
        error_msg = None
        if not success:
            error_msg = "No preferences could be extracted from the message"
        elif warnings:
            logger.warning(f"Turn {turn_id}: Extraction warnings: {warnings}")
        
        return ProfileInferenceResult(
            profile=profile,
            extraction_mode="rules",
            raw_input=user_message,
            success=success,
            error_message=error_msg,
            extracted_fields=extracted_fields
        )
    
    def _validate_extracted_fields(self, extracted: Dict[str, Any], turn_id: int) -> PartialUserProfile:
        """Validate and clamp extracted fields to valid ranges."""
        profile = PartialUserProfile(turn_id=turn_id)
        warnings = []
        
        # Validate genre
        if extracted.get("genre"):
            genre = str(extracted["genre"]).lower()
            if genre in self.VALID_GENRES:
                profile.genre = genre
            else:
                warnings.append(f"Invalid genre: {genre}")
        profile.genre_confidence = max(0.0, min(1.0, float(extracted.get("genre_confidence", 0.0))))
        
        # Validate mood
        if extracted.get("mood"):
            mood = str(extracted["mood"]).lower()
            if mood in self.VALID_MOODS:
                profile.mood = mood
            else:
                warnings.append(f"Invalid mood: {mood}")
        profile.mood_confidence = max(0.0, min(1.0, float(extracted.get("mood_confidence", 0.0))))
        
        # Validate energy
        if extracted.get("energy") is not None:
            try:
                energy = float(extracted["energy"])
                if 0.0 <= energy <= 1.0:
                    profile.energy = energy
                else:
                    warnings.append(f"Energy out of range [0.0, 1.0]: {energy}")
            except (ValueError, TypeError):
                warnings.append(f"Invalid energy value: {extracted.get('energy')}")
        profile.energy_confidence = max(0.0, min(1.0, float(extracted.get("energy_confidence", 0.0))))
        
        # Validate acoustic
        if extracted.get("likes_acoustic") is not None:
            profile.likes_acoustic = bool(extracted["likes_acoustic"])
        profile.acoustic_confidence = max(0.0, min(1.0, float(extracted.get("acoustic_confidence", 0.0))))
        
        profile.parse_warnings = warnings
        return profile


class ConversationManager:
    """Manage multi-turn conversation and accumulating profile state."""
    
    def __init__(self, mode: ExtractionMode = ExtractionMode.RULES):
        self.extractor = PreferenceExtractor(mode)
        self.accumulated_profile = PartialUserProfile()
        self.turn_count = 0
        self.turn_history: List[ProfileInferenceResult] = []
        logger.info(f"ConversationManager initialized with mode: {mode.value}")
    
    def process_turn(self, user_message: str) -> ProfileInferenceResult:
        """
        Process one turn of conversation and accumulate profile state.
        Returns the inference result; accumulated profile is updated internally.
        """
        self.turn_count += 1
        result = self.extractor.extract(user_message, self.turn_count)
        self.turn_history.append(result)
        
        # Accumulate/update profile
        if result.success:
            # Update with new inferred values, keeping confidence scores
            if result.profile.genre is not None:
                self.accumulated_profile.genre = result.profile.genre
                self.accumulated_profile.genre_confidence = result.profile.genre_confidence
            
            if result.profile.mood is not None:
                self.accumulated_profile.mood = result.profile.mood
                self.accumulated_profile.mood_confidence = result.profile.mood_confidence
            
            if result.profile.energy is not None:
                self.accumulated_profile.energy = result.profile.energy
                self.accumulated_profile.energy_confidence = result.profile.energy_confidence
            
            if result.profile.likes_acoustic is not None:
                self.accumulated_profile.likes_acoustic = result.profile.likes_acoustic
                self.accumulated_profile.acoustic_confidence = result.profile.acoustic_confidence
            
            self.accumulated_profile.turn_id = self.turn_count
            self.accumulated_profile.parse_warnings.extend(result.profile.parse_warnings)
        
        logger.info(f"Turn {self.turn_count}: Accumulated profile state: genre={self.accumulated_profile.genre}, "
                   f"mood={self.accumulated_profile.mood}, energy={self.accumulated_profile.energy}")
        
        return result
    
    def get_current_profile(self) -> PartialUserProfile:
        """Return the accumulated profile state."""
        return self.accumulated_profile
    
    def should_continue_conversation(self, confidence_threshold: float = 0.7) -> bool:
        """Determine if more turns are needed to reach confidence threshold."""
        current = self.accumulated_profile
        confident_fields = sum([
            current.genre is not None and current.genre_confidence >= confidence_threshold,
            current.mood is not None and current.mood_confidence >= confidence_threshold,
            current.energy is not None and current.energy_confidence >= confidence_threshold,
            current.likes_acoustic is not None and current.acoustic_confidence >= confidence_threshold,
        ])
        
        # Continue if fewer than 3 fields are confident, or if we've had < 2 turns
        return confident_fields < 3 and self.turn_count < 5
