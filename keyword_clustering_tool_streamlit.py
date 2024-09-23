import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import io
from openai import OpenAI

st.title('Keyword Research Cluster Analysis Tool')
st.subheader('Leverage OpenAI to cluster similar keywords into groups from your keyword list.')

st.markdown("""
## How to Use This Tool

1. **Prepare Your Data**: 
   - Create a CSV file with the following columns: 'Keywords', 'Search Volume', and 'CPC'.
   - Ensure your data is clean and formatted correctly.

2. **Get Your OpenAI API Key**:
   - If you don't have an OpenAI API key, sign up at [OpenAI](https://openai.com).
   - Ensure you have access to the GPT-4 model.

3. **Upload Your File**:
   - Use the file uploader below to upload your CSV file.

4. **Enter Your API Key**:
   - Input your OpenAI API key in the text box provided.

5. **Run the Analysis**:
   - The tool will automatically process your data once both the file and API key are provided.

6. **Review Results**:
   - Examine the clustered keywords and primary variants in the displayed table.

7. **Download Results**:
   - Use the download button to get a CSV file of your analysis results.

## Sample CSV Template

You can download a sample CSV template to see the required format:
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
    else:
        st.write(f'Loaded {len(data)} rows from the spreadsheet.')

api_key = st.text_input("Enter your OpenAI API key", type="password")

async def fetch_embedding(session, text, model="text-embedding-ada-002", max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": text, "model": model},
                timeout=10
            ) as response:
                result = await response.json()
                return result['data'][0]['embedding']
        except asyncio.TimeoutError:
            st.warning(f"Timeout occurred for keyword: {text}. Retrying... ({retries + 1}/{max_retries})")
        except Exception as e:
            st.error(f"Error fetching embedding for '{text}': {e}. Retrying... ({retries + 1}/{max_retries})")
        retries += 1
    st.error(f"Failed to fetch embedding for '{text}' after {max_retries} attempts.")
    return None

async def generate_embeddings(keywords):
    embeddings = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_keywords = len(keywords)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, keyword) for keyword in keywords]
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks), start=1):
            try:
                result = await asyncio.wait_for(task, timeout=30)  # 30-second timeout
                results.append(result)
            except asyncio.TimeoutError:
                st.warning(f"Timeout occurred for keyword {i}. Skipping...")
                results.append(None)
            except Exception as e:
                st.error(f"Error processing keyword {i}: {str(e)}")
                results.append(None)
            
            progress_bar.progress(i / total_keywords)
            progress_text.text(f"Processed {i}/{total_keywords} keywords")

    embeddings = [res for res in results if res is not None]
    valid_keywords = [kw for kw, res in zip(keywords, results) if res is not None]

    st.write(f"Successfully generated embeddings for {len(embeddings)} out of {total_keywords} keywords.")
    return embeddings, valid_keywords

async def choose_best_keyword(session, keyword1, keyword2):
    prompt = f"Identify which keyword users are more likely to search on Google for SEO: '{keyword1}' or '{keyword2}'. Only include the keyword in the response. If both keywords are similar, select the first one."
    try:
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are an SEO expert tasked with selecting the best keyword for search optimization."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20
            }
        )
        result = await response.json()
        best_keyword = result['choices'][0]['message']['content'].strip()
        if keyword1.lower() in best_keyword.lower():
            return keyword1
        elif keyword2.lower() in best_keyword.lower():
            return keyword2
        else:
            st.warning(f"Unexpected response from GPT-4: {best_keyword}")
            return keyword1
    except Exception as e:
        st.error(f"Error choosing best keyword: {e}")
        return keyword1

async def identify_primary_variants(session, cluster_data):
    primary_variant_df = pd.DataFrame(columns=['Cluster ID', 'Keywords', 'Is Primary', 'Primary Keyword', 'GPT-4 Reason'])
    new_rows = []
    total_clusters = cluster_data['Cluster ID'].nunique()
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, (cluster_id, group) in enumerate(cluster_data.groupby('Cluster ID'), start=1):
        keywords = group['Keywords'].tolist()
        primary = None

        if len(keywords) == 2:
            primary = await choose_best_keyword(session, keywords[0], keywords[1])
        else:
            embeddings, valid_keywords = await generate_embeddings(keywords)
            if not embeddings:
                st.error(f"Failed to generate embeddings for cluster {cluster_id}. Skipping...")
                continue
            similarity_matrix = cosine_similarity(np.array(embeddings))
            avg_similarity = np.mean(similarity_matrix, axis=1)
            primary_idx = np.argmax(avg_similarity)
            primary = valid_keywords[primary_idx]

        for keyword in keywords:
            is_primary = 'Yes' if keyword == primary else 'No'
            new_row = {
                'Cluster ID': cluster_id,
                'Keywords': keyword,
                'Is Primary': is_primary,
                'Primary Keyword': primary,
                'GPT-4 Reason': primary
            }
            new_rows.append(new_row)
        progress_bar.progress(i / total_clusters)
        progress_text.empty()
        progress_text.write(f"Processed {i}/{total_clusters} clusters", unsafe_allow_html=True)

    return pd.DataFrame(new_rows)

async def process_data(keywords, search_volumes, cpcs):
    st.write("Generating embeddings for the keywords...")
    embeddings, valid_keywords = await generate_embeddings(keywords)

    if embeddings:
        st.write("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)

        st.write("Clustering keywords...")
        try:
            clusters = fcluster(linkage(1 - similarity_matrix, method='average'), t=0.2, criterion='distance')
            if len(clusters) != len(valid_keywords):
                st.error("Mismatch between the number of clusters and keywords. Check embedding process for errors.")
                return
            filtered_data = data[data['Keywords'].isin(valid_keywords)].copy()
            filtered_data['Cluster ID'] = clusters

            st.write("Identifying primary keywords within clusters...")
            async with aiohttp.ClientSession() as session:
                primary_variant_df = await identify_primary_variants(session, filtered_data[['Cluster ID', 'Keywords']])
                combined_data = pd.merge(filtered_data, primary_variant_df, on=['Cluster ID', 'Keywords'], how='left')

            st.write("Analysis complete. Review the clusters below:")
            st.dataframe(combined_data)

            st.download_button(
                label='Download Analysis Results',
                data=combined_data.to_csv(index=False).encode('utf-8'),
                file_name='analysis_results.csv',
                mime='text/csv',
                key='download-csv'
            )
        except ValueError as e:
            st.error(f"Error during clustering: {e}. Ensure that embeddings are properly generated.")
    else:
        st.error("Failed to generate embeddings for all keywords.")

if uploaded_file is not None and api_key:
    keywords = data['Keywords'].tolist()
    search_volumes = data['Search Volume'].tolist()
    cpcs = data['CPC'].tolist()
    asyncio.run(process_data(keywords, search_volumes, cpcs))
elif uploaded_file is None and api_key:
    st.warning("Please upload a CSV file to proceed.")
elif uploaded_file is not None and not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    st.info("Please upload a CSV file and enter your OpenAI API key to start the analysis.")
