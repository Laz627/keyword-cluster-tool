import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import io

# --- Streamlit UI Setup (Identical to Original) ---
st.title('Keyword Research Cluster Analysis Tool')
st.subheader('Leverage AI to cluster similar keywords into groups from your keyword list.')

# (Instructions and sample data setup remains the same as before)
st.markdown("""
## How to Use This Tool

1.  **Prepare Your Data**:
    *   Create a CSV file with columns: 'Keywords', 'Search Volume', and 'CPC'. Blank or N/A values for volume/CPC are acceptable.
    *   Ensure keywords in the 'Keywords' column are unique. You can use the button below to remove duplicates.

2.  **Get Your OpenAI API Key**:
    *   If you don't have one, sign up at [OpenAI](https://openai.com).
    *   Access to `gpt-3.5-turbo` is recommended for speed and cost-efficiency.

3.  **Upload Your File & Enter Key**: The analysis will begin automatically.
4.  **Review & Download Results**.

## Sample CSV Template
""")
sample_data = pd.DataFrame({
    'Keywords': ['buy shoes online', 'purchase shoes online', 'best running shoes', 'comfortable walking shoes'],
    'Search Volume': [1000, 800, 1500, 1200],
    'CPC': [0.5, 0.4, 0.7, 0.6]
})
csv_buffer = io.StringIO()
sample_data.to_csv(csv_buffer, index=False)
csv_str = csv_buffer.getvalue()
st.download_button(
    label="Download Sample CSV Template",
    data=csv_str,
    file_name="sample_keyword_data.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
api_key = st.text_input("Enter your OpenAI API key", type="password")

# --- Optimized & Corrected Functions ---

async def fetch_embedding(session, text, model="text-embedding-3-small", max_retries=3):
    """Asynchronously fetches a single embedding with a retry mechanism."""
    retries = 0
    while retries < max_retries:
        try:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": text, "model": model},
                timeout=20 # Increased timeout for robustness
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['data'][0]['embedding']
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            st.warning(f"Embedding error for '{text}': {e}. Retrying... ({retries + 1}/{max_retries})")
            retries += 1
            await asyncio.sleep(2 ** retries)
    st.error(f"Failed to fetch embedding for '{text}' after {max_retries} retries.")
    return None

async def generate_all_embeddings(keywords):
    """Generates embeddings for a list of keywords concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, keyword) for keyword in keywords]
        embeddings = await asyncio.gather(*tasks)
    embedding_map = {kw: emb for kw, emb in zip(keywords, embeddings) if emb is not None}
    return embedding_map

async def choose_best_keyword(session, keyword1, keyword2, model="gpt-3.5-turbo"):
    """Uses an LLM to determine the best of two keywords for SEO."""
    prompt = f"Between these two keywords, which one has stronger user search intent on Google for SEO: '{keyword1}' or '{keyword2}'? Respond with only the keyword you choose. If they are equivalent, return the first one."
    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 25, "temperature": 0.0,
            }
        ) as response:
            response.raise_for_status()
            result = await response.json()
            best_keyword = result['choices'][0]['message']['content'].strip()
            # More robustly check which keyword was returned
            if keyword2.lower() in best_keyword.lower():
                return keyword2
            return keyword1 # Default to the first keyword
    except aiohttp.ClientError:
        return keyword1 # Default to the first one on error

async def identify_primary_variants(session, cluster_data):
    """Identifies the primary keyword for each cluster in parallel without creating duplicates."""
    primary_variant_tasks = []
    
    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keywords'].tolist()
        
        if len(keywords) == 2:
            task = choose_best_keyword(session, keywords[0], keywords[1])
            primary_variant_tasks.append((cluster_id, task))
        elif len(keywords) > 2:
            embeddings = np.array(group['embedding'].tolist())
            similarity_matrix = cosine_similarity(embeddings)
            primary_idx = np.argmax(np.mean(similarity_matrix, axis=1))
            primary = keywords[primary_idx]
            primary_variant_tasks.append((cluster_id, asyncio.create_task(asyncio.sleep(0, result=primary))))

    primary_keywords_results = await asyncio.gather(*[task for _, task in primary_variant_tasks])
    primary_map = {cid: primary for (cid, _), primary in zip(primary_variant_tasks, primary_keywords_results)}
    
    # FIX: Use .map() to assign primary keywords. This is safe and prevents duplication.
    cluster_data['Primary Keyword'] = cluster_data['Cluster ID'].map(primary_map)
    cluster_data['Is Primary'] = np.where(cluster_data['Keywords'] == cluster_data['Primary Keyword'], 'Yes', 'No')
    
    return cluster_data

async def process_data(df):
    """Main data processing pipeline."""
    st.write("### Analysis Progress")
    progress_bar = st.progress(0, text="Step 1/4: Generating keyword embeddings...")

    # Step 1: Generate all embeddings once.
    keywords = df['Keywords'].unique().tolist()
    embedding_map = await generate_all_embeddings(keywords)
    
    if not embedding_map:
        st.error("Could not generate embeddings. Check your API key and OpenAI model access.")
        return

    df['embedding'] = df['Keywords'].map(embedding_map)
    df.dropna(subset=['embedding'], inplace=True)

    progress_bar.progress(0.25, text="Step 2/4: Clustering keywords...")

    # Step 2: Group and cluster. This correctly handles NA/blank values by creating separate groups for them.
    grouped = df.groupby(['Search Volume', 'CPC'], dropna=False)
    df['Cluster ID'] = -1
    cluster_id_counter = 0

    for _, group in grouped:
        if len(group) > 1:
            embeddings = np.array(group['embedding'].tolist())
            linkage_matrix = linkage(1 - cosine_similarity(embeddings), method='average')
            # The distance threshold 't' can be adjusted (0.1-0.3 is a good range)
            group_clusters = fcluster(linkage_matrix, t=0.2, criterion='distance')
            
            # Ensure unique cluster IDs across all groups
            df.loc[group.index, 'Cluster ID'] = group_clusters + cluster_id_counter
            if group_clusters.max() > 0:
                cluster_id_counter += group_clusters.max()

    df_clustered = df[df['Cluster ID'] != -1].copy()
    unique_keywords_df = df[df['Cluster ID'] == -1]

    progress_bar.progress(0.5, text="Step 3/4: Identifying primary variants...")

    # Step 3: Identify primary variants and finalize columns.
    if not df_clustered.empty:
        async with aiohttp.ClientSession() as session:
            combined_data = await identify_primary_variants(session, df_clustered)

        progress_bar.progress(0.9, text="Step 4/4: Finalizing results...")
        
        # FIX: Define columns to show, excluding the 'embedding' vector.
        output_columns = ['Cluster ID', 'Keywords', 'Search Volume', 'CPC', 'Is Primary', 'Primary Keyword']
        
        st.write("### Clustered Keywords Analysis")
        st.dataframe(combined_data[output_columns])

        st.download_button(
            label='Download Clustered Analysis Results',
            data=combined_data[output_columns].to_csv(index=False).encode('utf-8'),
            file_name='clustered_analysis_results.csv',
            mime='text/csv'
        )
    else:
        st.info("No keyword clusters were formed. All keywords were unique.")

    # Display and allow download of unique keywords.
    if not unique_keywords_df.empty:
        unique_columns = ['Keywords', 'Search Volume', 'CPC']
        st.write("### Unique Keywords (Not Clustered)")
        st.dataframe(unique_keywords_df[unique_columns])

        st.download_button(
            label='Download Unique Keywords',
            data=unique_keywords_df[unique_columns].to_csv(index=False).encode('utf-8'),
            file_name='unique_keywords.csv',
            mime='text/csv'
        )

    progress_bar.progress(1.0, text="Analysis Complete!")


# --- Main Execution Logic ---
if uploaded_file is not None and api_key:
    data = pd.read_csv(uploaded_file, keep_default_na=False) # keep_default_na=False treats blanks as blanks

    initial_row_count = len(data)
    # Remove duplicate keywords to prevent processing errors and ensure clean output
    data.drop_duplicates(subset=['Keywords'], inplace=True, keep='first')
    final_row_count = len(data)

    if final_row_count < initial_row_count:
        st.info(f"Removed {initial_row_count - final_row_count} duplicate keyword rows from your file.")

    asyncio.run(process_data(data))

elif not api_key and uploaded_file:
    st.warning("Please enter your OpenAI API key to proceed.")
elif api_key and not uploaded_file:
    st.warning("Please upload a CSV file to proceed.")
else:
    st.info("Upload a CSV file and enter your API key to start the analysis.")
