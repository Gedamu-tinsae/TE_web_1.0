import re
from difflib import SequenceMatcher
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common license plate patterns with their descriptions
LICENSE_PATTERNS = [
    # UK Formats
    {"pattern": r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$', "name": "UK New Style (AB12 CDE)"},
    {"pattern": r'^[A-Z]{1,2}\d{1,4}$', "name": "UK Old Style (A123)"},
    {"pattern": r'^[A-Z]{3}\d{1,4}$', "name": "UK Diplomatic (XYZ123)"},
    
    # US Formats (examples from different states)
    {"pattern": r'^\d{1,3}[A-Z]{3}$', "name": "US - 3 Letters (123ABC)"},
    {"pattern": r'^[A-Z]{3}\d{3,4}$', "name": "US - 3 Letters (ABC1234)"},
    {"pattern": r'^\d{3}[A-Z]{2}\d{1}$', "name": "CA Format (123AB4)"},
    
    # European Formats
    {"pattern": r'^[A-Z]{1,3}-\d{1,4}-[A-Z]{1,2}$', "name": "EU with Hyphens (AB-1234-C)"},
    {"pattern": r'^[A-Z]{1,2} \d{1,4} [A-Z]{1,2}$', "name": "EU with Spaces (AB 1234 C)"},
    
    # Special formats
    {"pattern": r'^COVID\d{2}$', "name": "Special - COVID19"},
    {"pattern": r'^[A-Z]{5}$', "name": "All Letters (ABCDE)"},
    
    # General formats
    {"pattern": r'^[A-Z]{2}\d{2}[A-Z]{2}$', "name": "Format AB12CD"},
    {"pattern": r'^[A-Z]{2}\d{3}[A-Z]{2}$', "name": "Format AB123CD"},
    {"pattern": r'^[A-Z]{1,3}\d{1,4}[A-Z]{0,2}$', "name": "General Format (ABC1234DE)"},
]

# Character confusion mapping (commonly misread characters)
CHAR_CONFUSION = {
    '0': ['O', 'D', 'Q'],
    'O': ['0', 'D', 'Q'],
    '1': ['I', 'L', 'T'],
    'I': ['1', 'L', 'T'],
    'L': ['1', 'I', 'T'],
    '2': ['Z', 'S'],
    'Z': ['2', 'S'],
    '5': ['S', 'B'],
    'S': ['5', '3', 'B'],
    '8': ['B', '3'],
    'B': ['8', '3', 'P', 'R'],
    'D': ['0', 'O', 'Q'],
    'G': ['6', 'C'],
    '6': ['G', 'C'],
    'U': ['V', 'Y'],
    'V': ['U', 'Y'],
    'W': ['VV', 'M'],
    'M': ['N', 'H'],
    'N': ['M', 'H'],
    'P': ['R', 'F'],
    'R': ['P', 'F', 'B']
}

def matches_pattern(text):
    """Check if text matches any of the common license plate patterns."""
    # Clean text for pattern matching (remove spaces)
    cleaned_text = ''.join(text.split())
    
    for pattern_dict in LICENSE_PATTERNS:
        if re.match(pattern_dict["pattern"], cleaned_text):
            return True, pattern_dict["name"]
    return False, None

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def generate_pattern_variants(text):
    """Generate possible variants based on common OCR confusions."""
    variants = [text]
    
    # Generate all possible first-level substitutions
    for i, char in enumerate(text):
        if char in CHAR_CONFUSION:
            for confused_char in CHAR_CONFUSION[char]:
                new_text = text[:i] + confused_char + text[i+1:]
                if new_text not in variants:
                    variants.append(new_text)
    
    # For texts with fewer than 8 characters, also generate limited second-level substitutions
    if len(text) < 8:
        initial_variants = variants.copy()
        for variant in initial_variants:
            for i, char in enumerate(variant):
                if char in CHAR_CONFUSION:
                    for confused_char in CHAR_CONFUSION[char]:
                        new_text = variant[:i] + confused_char + variant[i+1:]
                        if new_text not in variants:
                            variants.append(new_text)
    
    return variants

def correct_plate_text(text, confidence_threshold=0.5):
    """
    Attempt to correct the plate text based on pattern matching and character confusion.
    
    Args:
        text: The OCR-detected license plate text
        confidence_threshold: Minimum confidence to apply corrections
        
    Returns:
        tuple: (corrected_text, is_valid_pattern, confidence, pattern_name)
    """
    cleaned_text = ''.join(c for c in text if c.isalnum())
    
    # Check if it already matches a pattern
    is_match, pattern_name = matches_pattern(cleaned_text)
    if is_match:
        return cleaned_text, True, 1.0, pattern_name
    
    # Try common substitutions for full text
    variants = generate_pattern_variants(cleaned_text)
    
    best_match = None
    best_pattern = None
    best_distance = float('inf')
    
    # Check all variants against all patterns
    for variant in variants:
        is_match, pattern_name = matches_pattern(variant)
        if is_match:
            return variant, True, 0.9, pattern_name  # Direct match through substitution
        
        # If no direct match, find the closest pattern match
        for pattern_dict in LICENSE_PATTERNS:
            # Extract pattern without the anchors and character classes
            simplified_pattern = pattern_dict["pattern"].replace('^', '').replace('$', '').replace('?', '')
            
            # Get the expected format by replacing pattern symbols with representative characters
            expected_format = re.sub(r'\[A-Z\]', 'A', simplified_pattern)
            expected_format = re.sub(r'\\\d', '1', expected_format)
            expected_format = re.sub(r'\{(\d+)(,\d+)?\}', lambda m: '1' * int(m.group(1)), expected_format)
            
            # Calculate distance to this pattern
            distance = levenshtein_distance(variant, expected_format)
            
            # Normalize distance by the length of the longer string
            normalized_distance = distance / max(len(variant), len(expected_format))
            
            if normalized_distance < best_distance:
                best_distance = normalized_distance
                best_match = variant
                best_pattern = pattern_dict["name"]
    
    # If we found a reasonably close match
    if best_match and best_distance < 0.3:  # Threshold for acceptable distance
        confidence = 1.0 - best_distance
        return best_match, False, confidence, best_pattern
    
    # No good matches found
    return cleaned_text, False, 0.0, None

def correct_candidates(candidates):
    """
    Process a list of text candidates, attempting to correct each one
    and adjust confidence scores based on pattern matching.
    
    Args:
        candidates: List of candidate dictionaries with text and confidence
        
    Returns:
        List of corrected candidates with pattern match information
    """
    corrected_candidates = []
    
    for candidate in candidates:
        original_text = candidate["text"]
        original_confidence = candidate["confidence"]
        
        # Try to correct the text
        corrected_text, is_pattern, correction_confidence, pattern_name = correct_plate_text(original_text)
        
        # Adjust confidence based on pattern matching
        pattern_bonus = 0.15 if is_pattern else 0.0
        final_confidence = min(1.0, original_confidence + pattern_bonus)
        
        # If corrected is different from original but matches a pattern well
        if corrected_text != original_text and correction_confidence > 0.7:
            corrected_candidates.append({
                "text": corrected_text,
                "confidence": final_confidence,
                "pattern_match": is_pattern,
                "pattern_name": pattern_name,
                "original_text": original_text
            })
        
        # Always include the original (with updated confidence if it's a pattern match)
        corrected_candidates.append({
            "text": original_text,
            "confidence": final_confidence if is_pattern else original_confidence,
            "pattern_match": is_pattern,
            "pattern_name": pattern_name if is_pattern else None,
            "char_positions": candidate.get("char_positions", [])
        })
    
    # Sort by confidence
    corrected_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Remove duplicates (keeping highest confidence)
    seen_texts = set()
    unique_candidates = []
    for candidate in corrected_candidates:
        if candidate["text"] not in seen_texts:
            seen_texts.add(candidate["text"])
            unique_candidates.append(candidate)
    
    return unique_candidates[:10]  # Return top 10 unique candidates
