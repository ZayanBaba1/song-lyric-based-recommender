# Song Similarity Recommender

A lyric-based song recommendation algorithm that uses sentence embeddings to find thematically similar songs given an input song.

## What it does

The script loads a dataset of 5000 songs, converts each song's full lyrics into a 384-dimensional embedding using a pretrained sentence transformer model, and finds the most similar songs to a given input using cosine similarity.

## Example output

Songs similar to '400 Lux' by Lorde:
- 'Power Of Two' by Indigo Girls (similarity: 0.641)
- 'You And I by' Lady Gaga (similarity: 0.609)

Songs similar to 'Let Down' by Radiohead:
- 'Virtual Death' by Black Sabbath (similarity: 0.601)
- 'All Fall Down' by OneRepublic (similarity: 0.576)

## Analysis

I input specific songs from a dataset and ran a lyric similarity search to find songs with similar lyrics and sentiment to the input. After loading the dataset into CoLab, I transformed a set of 5000 songs each into vectors of 384 dimensions, and then used cosine similarity to compare and find the songs most similar semantically to the input song. Prior to inputting entire songs, I used song lyric snippets and the score, even when lyrics were similar, would reach from around 0.2 to 0.3. This is dissimilar from my results when inputting full songs reaching a score of over 0.6. The results from the inputs of 400 Lux by Lorde and Let Down by Radiohead were sometimes inconsistent, suggesting the necessity of having a higher similarity threshold or having additional filtering. 400 Lux is a very intimate song and the songs that were recommended were not always intimate, having a very violent theme. Meanwhile, Let Down's suggestions were more accurate with themes of hopelessness while also trying to transcend it. The embeddings capture tone and emotion to a level well enough to make reasonable predictions, though it can be inaccurate, using just lyrics. 

## Limitations

- Dataset has heavier focus on older music, less relevance for listeners of more modern/ contemporary music and artists 
- Lyrics-only approach captures theme and emotion but lacks genre, era, and cultural context
- Brute force cosine similarity works for 5000 songs but wouldn't work well at a larger scale — a production system would use approximate nearest neighbor search (e.g. FAISS) for efficiency

## Tools

Python, HuggingFace Sentence Transformers, scikit-learn, pandas, NumPy
