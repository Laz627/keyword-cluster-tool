import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import io

# ==============================================================================
# 1. STREAMLIT UI CONFIGURATION
# ==============================================================================
st.set_page_config(layout="wide")
st.title('AI-Powered Keyword Clustering Tool')
st.subheader('Efficiently group large keyword lists using semantic similarity.')

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload your Keyword CSV", type="csv")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    st.markdown("---")
    st.header("How to Use")
    st.markdown("""
    1.  **Prepare Your Data**: A CSV file with columns: `Keywords`, `Search Volume`, and `CPC`. Blanks are okay.
    2.  **Upload & Configure**: Upload the CSV and enter your OpenAI API key.
    3.  **Run Analysis**: The process starts automatically.
    4.  **Review Results**: The tool will output three tables:
        - **Clustered Keywords**: Keywords grouped by semantic similarity.
        - **Unique Keywords**: Keywords that didn't fit into any cluster.
        - **Failed Keywords**: Keywords that couldn't be processed due to API errors.
    """)

    st.markdown("---")
    st.header("Sample CSV Template")
    sample_data = pd.DataFrame({
        'Keywords': ['buy shoes online', 'purchase shoes online', 'best running shoes', 'comfortable walking shoes'],
        'Search Volume': [1000, 800, 1500, 1200],
        'CPC': [0.5, 0.4, 0.7, 0.6]
    })
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    st.download_button(
        label="Download Sample CSV",
        data=csv_str,
        file_name="sample_keyword_data.csv",
        mime="text/csv"
    )

# ==============================================================================
# 2. ASYNCHRONOUS HELPER FUNCTIONS (API CALLS)
# ==============================================================================

async def fetch_with_semaphore(semaphore, session, keyword, model, max_retries):
    """Wrapper to control concurrency using an asyncio.Semaphore."""
    async with semaphore:
        return await fetch_embedding(session, keyword, model, max_retries)

async def fetch_embedding(session, text, model, max_retries):
    """Asynchronously fetches a single keyword embedding with robust error handling."""
    retries = 0
    while retries < max_retries:
        try:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": text, "model": model},
                timeout=30  # Increased timeout for stability
            ) as response:
                if response.status == 429:  # Rate limit error
                    retry_after = int(response.headers.get("Retry-After", 30))
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                result = await response.json()
                return text, result['data'][0]['embedding']
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            retries += 1
            if retries >= max_retries:
                break
            await asyncio.sleep(2 ** retries)
    return text, None  # Return keyword and None on failure

async def generate_all_embeddings(keywords, concurrency_limit=50):
    """
    Generates embeddings for a list of keywords with controlled concurrency to prevent
    API rate limit failures. It returns a map of successful embeddings and a list of failures.
    """
    semaphore = asyncio.Semaphore(concurrency_limit)
    embedding_map = {}
    failed_keywords = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_semaphore(semaphore, session, kw, "text-embedding-3-small", 3) for kw in keywords]
        
        # UI progress bar for the embedding step
        progress_text = f"Fetching embeddings for {len(keywords)} keywords..."
        st_progress = st.progress(0, text=progress_text)
        
        for i, future in enumerate(asyncio.as_completed(tasks)):
            keyword, embedding = await future
            if embedding is not None:
                embedding_map[keyword] = embedding
            else:
                failed_keywords.append(keyword)
            st_progress.progress((i + 1) / len(tasks), text=f"Fetching embeddings: {i + 1}/{len(keywords)} complete")
        st_progress.empty()

    return embedding_map, failed_keywords

async def choose_best_keyword(session, keyword1, keyword2):
    """Uses an LLM to determine the best of two keywords for SEO."""
    prompt = f"Between these two keywords, which one has stronger user search intent on Google for SEO: '{keyword1}' or '{keyword2}'? Respond with only the keyword you choose. If they are equivalent, return the first one."
    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": 25, "temperature": 0.0}
        ) as response:
            response.raise_for_status()
            result = await response.json()
            # More robust check for returned keyword
            if keyword2.lower() in result['choices'][0]['message']['content'].strip().lower():
                return keyword2
            return keyword1
    except aiohttp.ClientError:
        return keyword1  # Default to the first keyword on any API error

# ==============================================================================
# 3. CORE PROCESSING LOGIC
# ==============================================================================

async def identify_primary_variants(session, cluster_data):
    """
    Identifies the primary keyword for each cluster in parallel.
    This function modifies the DataFrame in place and does not create duplicates.
    """
    primary_variant_tasks = []
    # Group by the cluster ID to process each cluster
    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keywords'].tolist()
        
        if len(keywords) == 2:
            # For 2-keyword clusters, use LLM to decide
            task = choose_best_keyword(session, keywords[0], keywords[1])
            primary_variant_tasks.append((cluster_id, task))
        elif len(keywords) > 2:
            # For larger clusters, find the most central keyword using embeddings
            embeddings = np.array([emb for emb in group['embedding']])
            similarity_matrix = cosine_similarity(embeddings)
            # The keyword with the highest average similarity to all others is the primary
            primary_keyword = keywords[np.argmax(np.mean(similarity_matrix, axis=1))]
            primary_variant_tasks.append((cluster_id, asyncio.create_task(asyncio.sleep(0, result=primary_keyword))))
            
    # Execute all API calls and tasks concurrently
    primary_keyword_results = await asyncio.gather(*[task for _, task in primary_variant_tasks])
    # Create a mapping from cluster ID to the determined primary keyword
    primary_map = {cid: primary for (cid, _), primary in zip(primary_variant_tasks, primary_keyword_results)}
    
    # Assign the results back to the DataFrame using the safe .map() method
    cluster_data['Primary Keyword'] = cluster_data['Cluster ID'].map(primary_map)
    cluster_data['Is Primary'] = np.where(cluster_data['Keywords'] == cluster_data['Primary Keyword'], 'Yes', 'No')
    return cluster_data

async def process_data(df):
    """Main data processing pipeline with robust error handling and clear steps."""
    st.info("Step 1/3: Generating keyword embeddings... (This may take a while for large lists)")
    
    unique_keywords = df['Keywords'].unique().tolist()
    embedding_map, failed_keywords = await generate_all_embeddings(unique_keywords)
    
    # Separate failed keywords for later reporting
    failed_df = df[df['Keywords'].isin(failed_keywords)]
    
    # Continue processing only with keywords that were successfully embedded
    df['embedding'] = df['Keywords'].map(embedding_map)
    df_to_process = df.dropna(subset=['embedding']).copy()

    if df_to_process.empty:
        st.error("Could not generate any embeddings. Please check your API key, network connection, and OpenAI account status.")
        if not failed_df.empty:
            st.write("### Failed Keywords")
            st.dataframe(failed_df[['Keywords', 'Search Volume', 'CPC']])
        return

    st.info("Step 2/3: Clustering successful keywords...")
    # Group by volume and CPC. `dropna=False` ensures that rows with N/A are grouped together.
    grouped = df_to_process.groupby(['Search Volume', 'CPC'], dropna=False)
    df_to_process['Cluster ID'] = -1
    cluster_id_counter = 0

    for _, group in grouped:
        if len(group) > 1:
            embeddings = np.array(group['embedding'].tolist())
            linkage_matrix = linkage(1 - cosine_similarity(embeddings), method='average')
            # The distance threshold 't' controls how similar keywords must be to be clustered.
            group_clusters = fcluster(linkage_matrix, t=0.2, criterion='distance')
            
            # Assign unique cluster IDs globally to prevent overlap between groups
            valid_indices = group.index
            df_to_process.loc[valid_indices, 'Cluster ID'] = group_clusters + cluster_id_counter
            if group_clusters.max() > 0:
                cluster_id_counter += group_clusters.max()

    df_clustered = df_to_process[df_to_process['Cluster ID'] != -1].copy()
    unique_keywords_df = df_to_process[df_to_process['Cluster ID'] == -1]

    st.info("Step 3/3: Identifying primary variants and finalizing results...")
    
    # --- Final Output Generation ---
    st.markdown("---")
    st.header("Analysis Results")
    
    if not df_clustered.empty:
        async with aiohttp.ClientSession() as session:
            combined_data = await identify_primary_variants(session, df_clustered)

        output_columns = ['Cluster ID', 'Keywords', 'Search Volume', 'CPC', 'Is Primary', 'Primary Keyword']
        st.subheader(f"Clustered Keywords ({len(combined_data)})")
        st.dataframe(combined_data[output_columns])
        st.download_button('Download Clustered Results', combined_data[output_columns].to_csv(index=False).encode('utf-8'), 'clustered_results.csv', 'text/csv')

    # Display unique keywords
    unique_columns = ['Keywords', 'Search Volume', 'CPC']
    if not unique_keywords_df.empty:
        st.subheader(f"Unique Keywords ({len(unique_keywords_df)})")
        st.dataframe(unique_keywords_df[unique_columns])
        st.download_button('Download Unique Keywords', unique_keywords_df[unique_columns].to_csv(index=False).encode('utf-8'), 'unique_keywords.csv', 'text/csv')

    # Display failed keywords
    if not failed_df.empty:
        st.warning(f"{len(failed_df)} keywords could not be processed, likely due to API errors.")
        st.subheader(f"Failed Keywords ({len(failed_df)})")
        st.dataframe(failed_df[unique_columns])
        st.download_button('Download Failed Keywords', failed_df[unique_columns].to_csv(index=False).encode('utf-8'), 'failed_keywords.csv', 'text/csv')

    st.success("Analysis Complete!")
    st.balloons()


# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
if uploaded_file is not None and api_key:
    # Read CSV, treating blank cells as empty strings, not NaN, which helps with grouping.
    data = pd.read_csv(uploaded_file, keep_default_na=False)

    # Basic data validation
    if 'Keywords' not in data.columns:
        st.error("CSV file must include a 'Keywords' column. Please check your file.")
    else:
        # Pre-processing: Remove duplicate keywords to ensure a clean run
        initial_row_count = len(data)
        data.drop_duplicates(subset=['Keywords'], inplace=True, keep='first')
        if len(data) < initial_row_count:
            st.sidebar.info(f"Removed {initial_row_count - len(data)} duplicate keyword rows.")
        
        # Add 'Search Volume' and 'CPC' if they are missing
        if 'Search Volume' not in data.columns:
            data['Search Volume'] = ""
        if 'CPC' not in data.columns:
            data['CPC'] = ""

        # Run the main asynchronous process
        asyncio.run(process_data(data))

elif not uploaded_file and not api_key:
    st.info("Welcome! Please upload a CSV file and enter your API key in the sidebar to begin.")
elif not uploaded_file:
    st.warning("Please upload a CSV file.")
elif not api_key:
    st.warning("Please enter your OpenAI API key.")
