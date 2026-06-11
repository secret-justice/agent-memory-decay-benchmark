# Data Description

## personachat_enhanced.json (3.0 MB)

**Source:** Derived from PersonaChat (Zhang et al., ACL 2018)

**Description:** 100 synthetic users with memory banks derived from PersonaChat dialogue data.

**Structure:**
- `user_id`: unique user identifier
- `memory_bank`: list of memory entries, each containing:
  - `query`: 64-dimensional embedding vector (float32 precision)
  - `category`: one of 8 categories (food, hobbies, movies, music, pets, sports, travel, work)
  - `timestamp`: temporal timestamp
  - `step`: sequential step number
- `test_queries`: held-out queries for evaluation

**Preprocessing:** Embeddings generated via sentence transformer, rounded to float32 precision.

## lpt_200users.json (25.4 MB)

**Source:** Long-term Persona Tracking dataset

**Description:** 200 users with long-term memory interactions spanning 63 days.

**Structure:**
- `user_id`: unique user identifier
- `memory_bank`: list of memory entries (~189 per user), each containing:
  - `query`: 64-dimensional embedding vector (float32 precision)
  - `category`: integer category label (0-4)
  - `timestamp`: day-level timestamp (0-63)
  - `day`: day number
  - `step`: sequential step number

**Preprocessing:** Embeddings rounded to float32 precision. Test queries excluded to reduce file size.
