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
st.subheader('Group large keyword lists into a single, unified output file.')

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload your Keyword CSV", type="csv")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    st.markdown("---")
    # Adjustable slider to control cluster tightness
    cluster_threshold = st.slider(
        'Clustering Tightness (Distance Threshold)',
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Controls how similar keywords must be to be clustered. Lower values = Tighter clusters. Higher values = Looser clusters."
    )
    
    st.markdown("---")
    st.header("How to Use")
    st.markdown("""
    1.  **Prepare Data**: A CSV with `Keywords`, `Search Volume`, and `CPC`.
    2.  **Configure**: Upload your CSV, enter your API key, and adjust the **Clustering Tightness**. A value around **0.5** is a good starting point.
    3.  **Run & Review**: The analysis generates a single table with all keywords, each with a status: "Clustered", "Unique", or "Failed".
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
                timeout=30
            ) as response:
                if response.status == 429:  # Rate limit error
                    retry_after = int(response.headers.get("Retry-After", 30))
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                result = await response.json()
                return text, result['data'][0]['embedding']
        except (asyncio.TimeoutError, aiohttp.ClientError):
            retries += 1
            if retries >= max_retries:
                break
            await asyncio.sleep(2 ** retries)
    return text, None

async def generate_all_embeddings(keywords, concurrency_limit=50):
    """Generates embeddings with controlled concurrency, returning successful and failed results."""
    semaphore = asyncio.Semaphore(concurrency_limit)
    embedding_map, failed_keywords = {}, []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_semaphore(semaphore, session, kw, "text-embedding-3-small", 3) for kw in keywords]
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
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 25, "temperature": 0.0}
        ) as response:
            response.raise_for_status()
            result = await response.json()
            if keyword2.lower() in result['choices'][0]['message']['content'].strip().lower():
                return keyword2
            return keyword1
    except aiohttp.ClientError:
        return keyword1

# ==============================================================================
# 3. CORE PROCESSING LOGIC
# ==============================================================================

async def identify_primary_variants(session, cluster_data):
    """Identifies the primary keyword for each cluster in parallel."""
    primary_variant_tasks = []
    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keywords'].tolist()
        
        if len(keywords) == 2:
            task = choose_best_keyword(session, keywords[0], keywords[1])
            primary_variant_tasks.append((cluster_id, task))
        elif len(keywords) > 2:
            embeddings = np.array([emb for emb in group['embedding']])
            similarity_matrix = cosine_similarity(embeddings)
            primary_keyword = keywords[np.argmax(np.mean(similarity_matrix, axis=1))]
            primary_variant_tasks.append((cluster_id, asyncio.create_task(asyncio.sleep(0, result=primary_keyword))))
            
    primary_keyword_results = await asyncio.gather(*[task for _, task in primary_variant_tasks])
    primary_map = {cid: primary for (cid, _), primary in zip(primary_variant_tasks, primary_keyword_results)}
    
    cluster_data['Primary Keyword'] = cluster_data['Cluster ID'].map(primary_map)
    cluster_data['Is Primary'] = np.where(cluster_data['Keywords'] == cluster_data['Primary Keyword'], 'Yes', 'No')
    return cluster_data

async def process_data(df, threshold):
    """Main data processing pipeline that produces a single, unified output."""
    st.info("Step 1/3: Generating keyword embeddings...")
    unique_keywords = df['Keywords'].unique().tolist()
    embedding_map, failed_keywords = await generate_all_embeddings(unique_keywords)
    
    failed_df = df[df['Keywords'].isin(failed_keywords)].copy()
    failed_df['Status'] = 'Failed'
    
    df['embedding'] = df['Keywords'].map(embedding_map)
    df_to_process = df.dropna(subset=['embedding']).copy()

    if df_to_process.empty:
        st.error("Could not generate any embeddings.")
        if not failed_df.empty:
            st.write("### Analysis Results")
            st.dataframe(failed_df[['Status', 'Keywords', 'Search Volume', 'CPC']])
        return

    st.info("Step 2/3: Clustering successful keywords...")
    grouped = df_to_process.groupby(['Search Volume', 'CPC'], dropna=False)
    df_to_process['Cluster ID'] = -1
    cluster_id_counter = 1

    for _, group in grouped:
        if len(group) > 1:
            embeddings = np.array(group['embedding'].tolist())
            linkage_matrix = linkage(1 - cosine_similarity(embeddings), method='average')
            group_clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
            
            # Map local cluster numbers to a globally unique ID
            unique_cluster_ids_in_group = pd.Series(group_clusters).unique()
            cluster_id_map = {old_id: cluster_id_counter + i for i, old_id in enumerate(unique_cluster_ids_in_group)}
            df_to_process.loc[group.index, 'Cluster ID'] = pd.Series(group_clusters, index=group.index).map(cluster_id_map)
            cluster_id_counter += len(unique_cluster_ids_in_group)

    df_clustered = df_to_process[df_to_process['Cluster ID'] != -1].copy()
    unique_keywords_df = df_to_process[df_to_process['Cluster ID'] == -1].copy()
    unique_keywords_df['Status'] = 'Unique'
    
    st.info("Step 3/3: Identifying primary variants and finalizing results...")
    
    all_results = []
    if not df_clustered.empty:
        # ** THE FIX: Re-classify single-item clusters as "Unique" **
        cluster_counts = df_clustered['Cluster ID'].value_counts()
        single_item_cluster_ids = cluster_counts[cluster_counts == 1].index
        
        # Move these single-item rows to the 'unique' dataframe
        single_item_df = df_clustered[df_clustered['Cluster ID'].isin(single_item_cluster_ids)]
        unique_keywords_df = pd.concat([unique_keywords_df, single_item_df])
        unique_keywords_df['Status'] = 'Unique'

        # Process only true clusters (with 2+ items)
        df_clustered = df_clustered[~df_clustered['Cluster ID'].isin(single_item_cluster_ids)]
        
        if not df_clustered.empty:
            async with aiohttp.ClientSession() as session:
                clustered_data = await identify_primary_variants(session, df_clustered)
            clustered_data['Status'] = 'Clustered'
            all_results.append(clustered_data)

    all_results.append(unique_keywords_df)
    all_results.append(failed_df)
    
    final_df = pd.concat(all_results, ignore_index=True)
    final_df['Cluster ID'] = final_df['Cluster ID'].replace(-1, np.nan).astype('Int64')
    
    output_columns = ['Status', 'Cluster ID', 'Primary Keyword', 'Is Primary', 'Keywords', 'Search Volume', 'CPC']
    for col in output_columns:
        if col not in final_df.columns:
            final_df[col] = ''
    final_df = final_df[output_columns]
    
    # Sort for readability: Clustered first, then by Cluster ID, with Primary keyword at the top.
    final_df.sort_values(
        by=['Status', 'Cluster ID', 'Is Primary'],
        ascending=[True, True, False],
        na_position='last',
        inplace=True
    )

    st.markdown("---")
    st.header("Analysis Results")
    st.dataframe(final_df)
    st.download_button('Download Full Analysis Results', final_df.to_csv(index=False).encode('utf-8'), 'full_keyword_analysis.csv', 'text/csv')
    st.success("Analysis Complete!")
    st.balloons()

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
if uploaded_file is not None and api_key:
    data = pd.read_csv(uploaded_file, keep_default_na=False, dtype=str)
    if 'Keywords' not in data.columns:
        st.error("CSV must include a 'Keywords' column.")
    else:
        initial_row_count = len(data)
        data.drop_duplicates(subset=['Keywords'], inplace=True, keep='first')
        if len(data) < initial_row_count:
            st.sidebar.info(f"Removed {initial_row_count - len(data)} duplicate keyword rows.")
        if 'Search Volume' not in data.columns: data['Search Volume'] = ''
        if 'CPC' not in data.columns: data['CPC'] = ''
        
        asyncio.run(process_data(data, cluster_threshold))

elif not uploaded_file and not api_key:
    st.info("Welcome! Configure your analysis in the sidebar to begin.")
elif not uploaded_file:
    st.warning("Please upload a CSV file.")
elif not api_key:
    st.warning("Please enter your OpenAI API key.")
