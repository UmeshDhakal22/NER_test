import json
from typing import List, Dict, Tuple, Set
from fuzzywuzzy import fuzz
from collections import defaultdict

class LocationNER:
    def __init__(self, fuzzy_threshold: int = 90):
        self.locations = self._load_json('places.json')
        self.types = self._load_json('type.json')
        
        self.locations_set = set(loc.lower() for loc in self.locations)
        self.types_set = set(t.lower() for t in self.types)
        
        # Create a dictionary of first two letters to possible matches for faster fuzzy matching
        self.locations_index = self._build_fuzzy_index(self.locations)
        self.types_index = self._build_fuzzy_index(self.types)
        
        self.fuzzy_threshold = fuzzy_threshold  # Default threshold of 90% similarity
        
        # Model loading removed as we're using exact matching
    
    def _load_json(self, filename: str) -> List[str]:
        """Load JSON file containing locations or types."""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _find_phrases(self, text: str, word_list: set, label: str) -> List[Dict]:
        """Find exact whole word matches in text from a given word list."""
        words = text.split()
        matches = []
        
        # First check for multi-word phrases (longer matches first)
        for phrase_length in range(3, 0, -1):  # Check 3-word, then 2-word, then 1-word phrases
            for i in range(len(words) - phrase_length + 1):
                # Get the phrase with original case for the match
                phrase = ' '.join(words[i:i+phrase_length])
                phrase_lower = phrase.lower()
                
                # Check if this phrase exists in our word list (case-insensitive)
                if phrase_lower in word_list:
                    # Find the exact case in the original text
                    start = text.lower().find(phrase_lower)
                    if start != -1:
                        end = start + len(phrase)
                        original_phrase = text[start:end]
                        
                        matches.append({
                            'entity': original_phrase,
                            'type': label,
                            'start': start,
                            'end': end
                        })
                        
                        text = text[:start] + (' ' * (end - start)) + text[end:]
        
        return matches
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text with the following rules:
        1. Don't split words when matching
        2. Any word not in type or location is a name
        3. If location comes before type, they are both part of a name
        """
        # First, try to find multi-word locations (up to 3 words) before splitting into individual words
        words = []
        word_list = text.split()
        i = 0
        n = len(word_list)
        
        while i < n:
            # Try to match the longest possible phrase first (up to 3 words)
            matched = False
            for j in range(min(3, n - i), 0, -1):
                phrase = ' '.join(word_list[i:i+j]).lower()
                if phrase in self.locations_set:
                    # Found a multi-word location
                    phrase_text = ' '.join(word_list[i:i+j])
                    start = text.find(phrase_text, i)
                    end = start + len(phrase_text)
                    words.append({
                        'text': phrase_text,
                        'start': start,
                        'end': end,
                        'type': 'LOC',
                        'source': 'exact_match',
                        'match': phrase_text,
                        'score': 100
                    })
                    i += j
                    matched = True
                    break
                elif phrase in self.types_set:
                    # Found a multi-word type
                    phrase_text = ' '.join(word_list[i:i+j])
                    start = text.find(phrase_text, i)
                    end = start + len(phrase_text)
                    words.append({
                        'text': phrase_text,
                        'start': start,
                        'end': end,
                        'type': 'TYPE',
                        'source': 'exact_match',
                        'match': phrase_text,
                        'score': 100
                    })
                    i += j
                    matched = True
                    break
            
            if not matched:
                # No multi-word match found, add as single word
                word = word_list[i]
                start = text.find(word, i)
                end = start + len(word)
                words.append({
                    'text': word,
                    'start': start,
                    'end': end,
                    'type': 'UNKNOWN',
                    'source': 'unmatched',
                    'match': '',
                    'score': 0
                })
                i += 1
        
        # Now process any remaining UNKNOWN words as single words
        for word in words:
            if word['type'] == 'UNKNOWN':
                word_lower = word['text'].lower()
                if word_lower in self.locations_set:
                    word['type'] = 'LOC'
                    word['source'] = 'exact_match'
                    word['match'] = word_lower
                    word['score'] = 100
                elif word_lower in self.types_set:
                    word['type'] = 'TYPE'
                    word['source'] = 'exact_match'
                    word['match'] = word_lower
                    word['score'] = 100
                else:
                    # Check for fuzzy matches
                    loc_matches = self._get_fuzzy_matches(word_lower, self.locations_set, self.locations_index)
                    type_matches = self._get_fuzzy_matches(word_lower, self.types_set, self.types_index)
                    
                    if loc_matches:
                        best_match = max(loc_matches, key=lambda x: fuzz.ratio(word_lower, x))
                        word['type'] = 'LOC'
                        word['source'] = 'fuzzy_match'
                        word['match'] = best_match
                        word['score'] = fuzz.ratio(word_lower, best_match)
                    elif type_matches:
                        best_match = max(type_matches, key=lambda x: fuzz.ratio(word_lower, x))
                        word['type'] = 'TYPE'
                        word['source'] = 'fuzzy_match'
                        word['match'] = best_match
                        word['score'] = fuzz.ratio(word_lower, best_match)
        
        # Second pass: Handle LOC followed by TYPE
        # Any LOC that comes before a TYPE should be marked as NAME
        i = 0
        while i < len(words) - 1:
            current = words[i]
            next_word = words[i + 1]
            
            # If current is LOC and next is TYPE, mark current as NAME
            if current['type'] == 'LOC' and next_word['type'] == 'TYPE':
                if current.get('source') == 'unmatched':
                    current['source'] = 'unmatched_text'
                    current['match'] = ''
                    current['score'] = 0
                current['type'] = 'NAME'
                current['source'] = 'loc_before_type'
            i += 1
        
        # Third pass: Mark remaining UNKNOWN as NAME
        for word in words:
            if word['type'] == 'UNKNOWN':
                word['type'] = 'NAME'
                word['source'] = 'unmatched_text'
                word['match'] = ''
                word['score'] = 0
        
        # Combine adjacent same-type words
        combined_entities = []
        if words:
            current_entity = dict(words[0])
            
            for word in words[1:]:
                # If same type and adjacent, merge
                if (word['type'] == current_entity['type'] and 
                    word['source'] == current_entity['source'] and
                    word['start'] == current_entity['end'] + 1):  # +1 for space
                    current_entity['text'] += ' ' + word['text']
                    current_entity['end'] = word['end']
                    current_entity['match'] = current_entity.get('match', current_entity['text'])
                    current_entity['score'] = current_entity.get('score', 100)
                else:
                    combined_entities.append({
                        'entity': current_entity['text'],
                        'type': current_entity['type'],
                        'source': current_entity['source'],
                        'match': current_entity.get('match', current_entity['text']),
                        'score': current_entity.get('score', 100)
                    })
                    current_entity = dict(word)
            
            # Add the last entity
            combined_entities.append({
                'entity': current_entity['text'],
                'type': current_entity['type'],
                'source': current_entity['source'],
                'match': current_entity.get('match', current_entity['text']),
                'score': current_entity.get('score', 100)
            })
        
        return combined_entities
    
    def _build_fuzzy_index(self, words: List[str]) -> Dict[str, Set[str]]:
        """Build an index of words by their first two letters for faster fuzzy matching."""
        index = defaultdict(set)
        for word in words:
            word_lower = word.lower()
            if len(word_lower) >= 2:
                key = word_lower[:2]
                index[key].add(word_lower)
        return index
    
    def _get_fuzzy_matches(self, text: str, word_set: Set[str], index: Dict[str, Set[str]]) -> List[str]:
        """Find fuzzy matches for the given text in the word set using the index."""
        text_lower = text.lower()
        if len(text_lower) < 2:
            return []
            
        # Check exact match first
        if text_lower in word_set:
            return [text_lower]
            
        # Get potential matches using the first two letters
        key = text_lower[:2]
        potential_matches = index.get(key, set())
        
        # Find matches above threshold
        matches = []
        for word in potential_matches:
            if abs(len(word) - len(text_lower)) > 2:  # Skip if lengths differ by more than 2
                continue
            ratio = fuzz.ratio(text_lower, word)
            if ratio >= self.fuzzy_threshold:
                matches.append(word)
        
        return matches
    
    def is_location(self, text: str) -> bool:
        """Check if the given text is a known location with fuzzy matching."""
        text_lower = text.lower()
        if text_lower in self.locations_set:
            return True
        return len(self._get_fuzzy_matches(text, self.locations_set, self.locations_index)) > 0
    
    def is_type(self, text: str) -> bool:
        """Check if the given text is a known type with fuzzy matching."""
        text_lower = text.lower()
        if text_lower in self.types_set:
            return True
        return len(self._get_fuzzy_matches(text, self.types_set, self.types_index)) > 0
    
    def extract_locations(self, text: str) -> List[Dict]:
        """
        Extract locations and types from the given text with fuzzy matching.
        """
        entities = []
        words = text.split()
        
        # Check for multi-word locations (up to 5 words)
        for n in range(5, 0, -1):
            for i in range(len(words) - n + 1):
                if not any(words[i:i+n]):  # Skip if any word is already matched
                    continue
                    
                phrase = ' '.join(words[i:i+n])
                phrase_lower = phrase.lower()
                
                # Check for exact match first
                if phrase_lower in self.locations_set:
                    entities.append({
                        'entity': phrase,
                        'type': 'LOC',
                        'source': 'exact_match',
                        'score': 100
                    })
                    # Mark these words as matched
                    for j in range(i, i+n):
                        if j < len(words):
                            words[j] = ""
                    continue
                
                # Check for fuzzy matches
                fuzzy_matches = self._get_fuzzy_matches(phrase, self.locations_set, self.locations_index)
                if fuzzy_matches:
                    # Get the best match
                    best_match = max(fuzzy_matches, key=lambda x: fuzz.ratio(phrase_lower, x))
                    score = fuzz.ratio(phrase_lower, best_match)
                    entities.append({
                        'entity': phrase,
                        'type': 'LOC',
                        'source': 'fuzzy_match',
                        'match': best_match,
                        'score': score
                    })
                    # Mark these words as matched
                    for j in range(i, i+n):
                        if j < len(words):
                            words[j] = ""
        
        # Check for type words with fuzzy matching
        for word in text.split():
            if not word:
                continue
                
            word_lower = word.lower()
            
            # Check exact match first
            if word_lower in self.types_set:
                entities.append({
                    'entity': word,
                    'type': 'TYPE',
                    'source': 'exact_match',
                    'score': 100
                })
                continue
                
            # Check for fuzzy matches
            fuzzy_matches = self._get_fuzzy_matches(word, self.types_set, self.types_index)
            if fuzzy_matches:
                # Get the best match
                best_match = max(fuzzy_matches, key=lambda x: fuzz.ratio(word_lower, x))
                score = fuzz.ratio(word_lower, best_match)
                entities.append({
                    'entity': word,
                    'type': 'TYPE',
                    'source': 'fuzzy_match',
                    'match': best_match,
                    'score': score
                })
        
        # Remove duplicates (keeping the first occurrence)
        seen = set()
        unique_entities = []
        for ent in entities:
            if ent['entity'] not in seen:
                seen.add(ent['entity'])
                unique_entities.append(ent)
        
        return unique_entities

def main():
    # Initialize the NER system
    ner_system = LocationNER()
    
    # Example usage
    while True:
        text = input("\nEnter text to analyze (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
            
        print("\nAnalyzing text:", text)
        entities = ner_system.extract_locations(text)
        
        if not entities:
            print("No locations or types found in the text.")
        else:
            print("\nFound entities:")
            for i, ent in enumerate(entities, 1):
                print(f"{i}. {ent['entity']} - Type: {ent['type']} (Source: {ent['source']})")

if __name__ == "__main__":
    main()
