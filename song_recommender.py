
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def load_data( filepath, sample_size = 5000 ):
    print( "Loading dataset" )
    df = pd.read_csv( filepath ).sample( sample_size, random_state = 42 ).reset_index( drop = True )
    df = df.dropna( subset=[ 'text' ] )
    df = df[ [ 'artist', 'song', 'text' ] ]
    print( f"Loaded {len( df )} songs" )
    return df

def embed_songs( df ):
    print( "Loading embedding model" )
    model = SentenceTransformer( 'all-MiniLM-L6-v2' )
    print( "Embedding songs" )
    embeddings = model.encode( df[ 'text' ].tolist(), show_progress_bar = True )
    print( "Done" )
    return embeddings

def recommend_songs( song_title, df, embeddings, n = 5 ):
    matches = df[ df[ 'song' ].str.lower() == song_title.lower() ]
    
    if len( matches ) == 0:
        print( f"Song '{song_title}' not found in dataset" )
        return
    
    idx = matches.index[ 0 ]
    song_embedding = embeddings[ idx ].reshape( 1, -1 )
    similarities = cosine_similarity( song_embedding, embeddings )[ 0 ]
    similar_indices = similarities.argsort()[ ::-1 ][ 1 : n+1 ]
    
    print( f"\nSongs similar to '{df.iloc[ idx ][ 'song' ]}' by {df.iloc[ idx ][ 'artist' ]}:\n" )
    for i in similar_indices:
        print( f"  {df.iloc[ i ][ 'song' ]} - {df.iloc[ i ][ 'artist' ]} (similarity: {similarities[ i ]:.3f})")

def main():
    df = load_data( 'spotify_millsongdata.csv' )
    embeddings = embed_songs( df )
    
    # Try it out
    recommend_songs( "400 lux", df, embeddings )
    recommend_songs( "Let Down", df, embeddings )

if __name__ == '__main__':
    main()
