import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple

class LocationNER:
    def __init__(self):
        self.locations = self._load_json('places.json')
        self.types = self._load_json('type.json')
        
        self.locations_set = set(loc.lower() for loc in self.locations)
        self.types_set = set(t.lower() for t in self.types)
        
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def _ensure_model_loaded(self):
        """Lazy-load the NER model only when needed."""
        if self.model is None:
            try:
                self.model_name = "dslim/bert-base-NER"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load NER model: {e}")
                print("Falling back to exact matching only.")
                self.model = None
    
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
                        'source': 'exact_match'
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
                        'source': 'exact_match'
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
                    'source': 'unmatched'
                })
                i += 1
        
        # Now process any remaining UNKNOWN words as single words
        for word in words:
            if word['type'] == 'UNKNOWN':
                word_lower = word['text'].lower()
                if word_lower in self.locations_set:
                    word['type'] = 'LOC'
                    word['source'] = 'exact_match'
                elif word_lower in self.types_set:
                    word['type'] = 'TYPE'
                    word['source'] = 'exact_match'
        
        # Second pass: Handle LOC followed by TYPE
        # Any LOC that comes before a TYPE should be marked as NAME
        i = 0
        while i < len(words) - 1:
            current = words[i]
            next_word = words[i + 1]
            
            # If current is LOC and next is TYPE, mark current as NAME
            if current['type'] == 'LOC' and next_word['type'] == 'TYPE':
                current['type'] = 'NAME'
                current['source'] = 'loc_before_type'
            i += 1
        
        # Third pass: Mark remaining UNKNOWN as NAME
        for word in words:
            if word['type'] == 'UNKNOWN':
                word['type'] = 'NAME'
                word['source'] = 'unmatched_text'
        
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
                else:
                    combined_entities.append({
                        'entity': current_entity['text'],
                        'type': current_entity['type'],
                        'source': current_entity['source']
                    })
                    current_entity = dict(word)
            
            # Add the last entity
            combined_entities.append({
                'entity': current_entity['text'],
                'type': current_entity['type'],
                'source': current_entity['source']
            })
        
        return combined_entities
    
    def is_location(self, text: str) -> bool:
        """Check if the given text is a known location."""
        return text.lower() in self.locations_set
    
    def is_type(self, text: str) -> bool:
        """Check if the given text is a known type."""
        return text.lower() in self.types_set
    
    def extract_locations(self, text: str) -> List[Dict]:
        """
        Extract locations from the given text using both:
        1. Exact match with known locations
        2. NER model predictions
        """
        # Check for exact matches first
        exact_matches = []
        words = text.split()
        
        # Check for multi-word locations (up to 5 words)
        for n in range(5, 0, -1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                if self.is_location(phrase):
                    exact_matches.append({
                        'entity': phrase,
                        'type': 'LOC',
                        'source': 'exact_match'
                    })
                    # Mark these words as matched
                    for j in range(i, i+n):
                        if j < len(words):
                            words[j] = ""
        
        # Check for type words
        type_matches = []
        for word in text.split():
            if self.is_type(word):
                type_matches.append({
                    'entity': word,
                    'type': 'TYPE',
                    'source': 'type_match'
                })
        
        # Get NER predictions for remaining text
        remaining_text = ' '.join(word for word in words if word)
        ner_entities = self.predict_entities(remaining_text)
        
        # Filter only LOCATION entities
        ner_locations = [
            {'entity': ent['entity'], 'type': ent['type'], 'source': 'ner'}
             for ent in ner_entities 
             if ent['type'] in ['LOC', 'GPE', 'FAC', 'ORG']
        ]
        
        # Combine all results
        all_entities = exact_matches + ner_locations + type_matches
        
        # Remove duplicates (keeping the first occurrence)
        seen = set()
        unique_entities = []
        for ent in all_entities:
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
