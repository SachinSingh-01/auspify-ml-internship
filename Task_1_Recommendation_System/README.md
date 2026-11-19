# Netflix Content Recommendation System (Task 1)

## Overview
A content-based recommendation system built using the Netflix catalog dataset to suggest similar movies and TV shows based on metadata attributes including title, content type, director, country, maturity rating, and genre.

## Dataset & Preprocessing
- **Raw Dataset:** 8,790 titles (6,126 Movies, 2,664 TV Shows) across 10 columns
- **Data Cleaning:**
  - Removed 1 invalid record where `title` was `"Unknown"`, leaving 8,789 rows.
  - Replaced `"Not Given"` placeholders in `director` (2,588 rows) and `country` (287 rows) with empty strings.
  - Deduplicated 4 repeated title groups (`15-Aug`, `22-Jul`, `9-Feb`, `Consequences`) that shared identical metadata under different `show_id` entries, resulting in a final modeling dataset of **8,785 unique titles**.
- **Text Normalization:** Lowercased text, stripped punctuation (`&`, `,`), and formatted multi-word director names with underscores (e.g., `theodore_melfi`) so first and last names are treated as single tokens during vectorization.

## Methodology
1. **Feature Combination:** Merged cleaned `title`, `type`, `director`, `country`, `rating`, and `listed_in` columns into a single `combined_features` text field.
2. **TF-IDF Vectorization:** Applied `TfidfVectorizer(stop_words='english', ngram_range=(1, 2))` to convert the combined metadata into a sparse matrix of shape `(8785, 39362)`.
3. **Cosine Similarity:** Computed an `8785 × 8785` pairwise cosine similarity matrix to rank and return the top 10 most similar titles (excluding the queried title itself).

## Sample Recommendations
| Queried Title | Top Recommended Match | Type | Similarity Score |
| :--- | :--- | :---: | :---: |
| *The Starling* | *The Break-Up* | Movie | 0.2135 |
| *Breaking Bad* | *When They See Us* | TV Show | 0.5668 |
| *The Queen* | *Philomena* | Movie | 0.6811 |
| *Stranger Things* | *Beyond Stranger Things* | TV Show | 0.7720 |

## Limitations
- Relies solely on catalog metadata (`director`, `genre`, `rating`, `country`) rather than plot descriptions or user viewing history.
- Cosine similarity measures metadata overlap rather than true semantic similarity or viewer preference.

## How to Run
1. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn