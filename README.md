# Location NER System

This is a Named Entity Recognition (NER) system designed to identify location names and types in text using a pre-trained BERT model from Hugging Face.

## Features

- Identifies locations using:
  - Exact matches with a predefined list of places
  - Pre-trained BERT NER model for location detection
  - Type matching from a predefined list of location types
- Combines multiple matching strategies for better accuracy
- Handles multi-word location names

## Setup

1. Make a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the following files in the same directory:
   - `places.json`: List of known locations
   - `type.json`: List of location types
   - `ner_system.py`: The main NER system
   - `api.py`: The FastAPI web server

## Usage

Run the API server with:

```bash
python api.py
```

Then enter the text you want to analyze when prompted. The system will identify:
- Exact location matches from your places list
- Location types from your types list
- Locations detected by the NER model

## Example

Input:
```
I visited Kathmandu and Pokhara last summer. The hotel was near Thamel Chowk.
```

Output:
```
Analyzing text: I visited Kathmandu and Pokhara last summer. The hotel was near Thamel Chowk.

Found entities:
1. Kathmandu - Type: LOC (Source: exact_match)
2. Pokhara - Type: LOC (Source: exact_match)
3. Thamel Chowk - Type: LOC (Source: exact_match)
4. hotel - Type: TYPE (Source: type_match)
```

## Files

- `ner_system.py`: Main NER system implementation
- `places.json`: List of known locations
- `type.json`: List of location types
- `requirements.txt`: Python dependencies

## Note

The first run will download the pre-trained BERT model (about 440MB).
