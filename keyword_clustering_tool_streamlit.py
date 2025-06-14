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
st.subheader('Leverage OpenAI to cluster similar keywords into groups from your keyword list.')

st.markdown("""
## How to Use This Tool

1.  **Prepare Your Data**:
    *   Create a CSV file with the following columns: 'Keywords', 'Search Volume', and 'CPC'.
    *   Ensure your data is clean and formatted correctly.

2.  **Get Your OpenAI API Key**:
    *   If you don't have an OpenAI API key, sign up at [OpenAI](https://openai.com).
    *   Ensure you have access to an appropriate model (e.g., GPT-3.5-Turbo).

3.  **Upload Your File**:
    *   Use the file uploader below to upload your CSV file.

4.  **Enter Your API Key**:
    *   Input your OpenAI API key in the text box provided.

5.  **Run the Analysis**:
    *   The tool will automatically process your data once both the file and API key are provided.

6.  **Review Results**:
    *   Examine the clustered keywords and primary variants in the displayed table.

7.  **Download Results**:
    *   Use the download button to get a CSV file of your analysis results.

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
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    required_columns = {'Keywords', 'Search Volume', 'CPC'}
    if not required_columns.issubset(data.columns):
        st.error("CSV file must include the following columns: Keywords, Search Volume, CPC")

api_key = st.text_input("Enter your OpenAI API key", type="password")

# --- Optimized Functions ---

# OPTIMIZATION: Switched to a more cost-effective and faster embedding model.
async def fetch_embedding(session, text, model="text-embedding-3-small", max_retries=3):
    """Asynchronously fetches a single embedding with a retry mechanism."""
    retries = 0
    while retries < max_retries:
        try:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": text, "model": model},
                timeout=10
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['data'][0]['embedding']
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            st.warning(f"Error fetching embedding for '{text}': {e}. Retrying... ({retries + 1}/{max_retries})")
            retries += 1
            await asyncio.sleep(2 ** retries) # Exponential backoff
    st.error(f"Failed to fetch embedding for '{text}' after {max_retries} retries.")
    return None

# OPTIMIZATION: This function now runs only ONCE for all keywords.
async def generate_all_embeddings(keywords):
    """Generates embeddings for a list of keywords concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, keyword) for keyword in keywords]
        embeddings = await asyncio.gather(*tasks)
    
    # Create a dictionary mapping each keyword to its embedding
    embedding_map = {kw: emb for kw, emb in zip(keywords, embeddings) if emb is not None}
    return embedding_map

# OPTIMIZATION: Switched to gpt-3.5-turbo for faster and cheaper processing.
async def choose_best_keyword(session, keyword1, keyword2, model="gpt-4o-mini"):
    """Uses an LLM to determine the best of two keywords for SEO."""
    prompt = f"Identify which keyword users are more likely to search on Google for SEO: '{keyword1}' or '{keyword2}'. Only include the keyword in the response. If both keywords are similar, select the first one."
    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an SEO expert. Your task is to select the best keyword for search optimization based on user intent."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20,
                "temperature": 0.0,
            }
        ) as response:
            response.raise_for_status()
            result = await response.json()
            best_keyword = result['choices'][0]['message']['content'].strip()
            # More robustly check which keyword was returned
            if keyword2.lower() in best_keyword.lower():
                return keyword2
            return keyword1 # Default to the first keyword
    except aiohttp.ClientError as e:
        st.warning(f"LLM call failed for ({keyword1}, {keyword2}): {e}. Defaulting to first keyword.")
        return keyword1 # Default to the first one on error

# OPTIMIZATION: This function is now fully parallelized and reuses embeddings.
async def identify_primary_variants(session, cluster_data):
    """
    Identifies the primary keyword for each cluster.
    - For clusters of 2, it uses concurrent LLM calls.
    - For clusters of 3+, it calculates the most central keyword using pre-computed embeddings.
    """
    primary_variant_tasks = []
    
    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keywords'].tolist()
        
        if len(keywords) == 2:
            # Create a task for an LLM to choose the best variant
            task = choose_best_keyword(session, keywords[0], keywords[1])
            primary_variant_tasks.append((cluster_id, task))
        elif len(keywords) > 2:
            # For larger clusters, find the primary variant by centrality without a new API call
            embeddings = np.array(group['embedding'].tolist())
            similarity_matrix = cosine_similarity(embeddings)
            avg_similarity = np.mean(similarity_matrix, axis=1)
            primary_idx = np.argmax(avg_similarity)
            primary = keywords[primary_idx]
            primary_variant_tasks.append((cluster_id, asyncio.create_task(asyncio.sleep(0, result=primary)))) # Wrap in a resolved task
    
    # Execute all 'choose_best_keyword' calls concurrently
    primary_keywords_results = await asyncio.gather(*[task for _, task in primary_variant_tasks])
    
    primary_map = {cid: primary for (cid, _), primary in zip(primary_variant_tasks, primary_keywords_results)}
    
    # Map the results back to the dataframe
    cluster_data['Primary Keyword'] = cluster_data['Cluster ID'].map(primary_map)
    cluster_data['Is Primary'] = np.where(cluster_data['Keywords'] == cluster_data['Primary Keyword'], 'Yes', 'No')
    
    return cluster_data

# OPTIMIZATION: The main processing logic is rewritten for efficiency.
async def process_data(df):
    """Main data processing pipeline."""
    st.write("### Analysis Progress")
    progress_bar = st.progress(0, text="Step 1/4: Generating keyword embeddings...")

    # Step 1: Generate all embeddings at once
    keywords = df['Keywords'].unique().tolist()
    embedding_map = await generate_all_embeddings(keywords)
    
    if not embedding_map:
        st.error("Could not generate any embeddings. Please check your API key and network connection.")
        return

    df['embedding'] = df['Keywords'].map(embedding_map)
    df.dropna(subset=['embedding'], inplace=True) # Remove rows where embedding failed

    progress_bar.progress(0.25, text="Step 2/4: Clustering keywords...")

    # Step 2: Group by Search Volume/CPC and cluster using pre-computed embeddings
    grouped = df.groupby(['Search Volume', 'CPC'])
    df['Cluster ID'] = -1
    cluster_id_counter = 0

    for _, group in grouped:
        if len(group) > 1:
            embeddings = np.array(group['embedding'].tolist())
            
            # Use cosine distance for clustering; t=0.2 is a reasonable starting point
            linkage_matrix = linkage(1 - cosine_similarity(embeddings), method='average')
            group_clusters = fcluster(linkage_matrix, t=0.2, criterion='distance')
            
            # Assign unique cluster IDs globally
            df.loc[group.index, 'Cluster ID'] = group_clusters + cluster_id_counter
            cluster_id_counter += group_clusters.max()

    df_clustered = df[df['Cluster ID'] != -1].copy()
    unique_keywords_df = df[df['Cluster ID'] == -1][['Keywords', 'Search Volume', 'CPC']]

    progress_bar.progress(0.5, text="Step 3/4: Identifying primary keyword variants...")

    # Step 3: Identify primary variants in parallel
    if not df_clustered.empty:
        async with aiohttp.ClientSession() as session:
            # The 'embedding' column is passed implicitly via the df_clustered dataframe
            combined_data = await identify_primary_variants(session, df_clustered)
        
        progress_bar.progress(0.9, text="Step 4/4: Finalizing results...")
        
        # --- Display and Download Results ---
        st.write("### Clustered Keywords Analysis")
        st.dataframe(combined_data[['Cluster ID', 'Keywords', 'Search Volume', 'CPC', 'Is Primary', 'Primary Keyword']])

        st.download_button(
            label='Download Clustered Analysis Results',
            data=combined_data.to_csv(index=False).encode('utf-8'),
            file_name='clustered_analysis_results.csv',
            mime='text/csv'
        )

        if not unique_keywords_df.empty:
            st.write("### Unique Keywords (Not Clustered)")
            st.dataframe(unique_keywords_df)

            st.download_button(
                label='Download Unique Keywords',
                data=unique_keywords_df.to_csv(index=False).encode('utf-8'),
                file_name='unique_keywords.csv',
                mime='text/csv'
            )
        progress_bar.progress(1.0, text="Analysis Complete!")
    else:
        st.error("No clusters could be formed. All keywords were unique based on Search Volume and CPC.")
        progress_bar.progress(1.0, text="Analysis Complete.")


# --- Main Execution Logic ---
if uploaded_file is not None and api_key:
    # Pass the entire dataframe to the processing function
    asyncio.run(process_data(data))
elif uploaded_file is None and api_key:
    st.warning("Please upload a CSV file to proceed.")
elif uploaded_file is not None and not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    st.info("Please upload a CSV file and enter your OpenAI API key to start the analysis.")
