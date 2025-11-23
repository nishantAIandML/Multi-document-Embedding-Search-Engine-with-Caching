import streamlit as st
import requests

st.title('Lightweight Embedding Search')

# API configuration
API_URL = st.sidebar.text_input('API URL', value='http://localhost:8000', help='URL of the search API server')

# Check API health
try:
    health_res = requests.get(f'{API_URL}/health', timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success('API connected')
        health_data = health_res.json()
        if 'docs_loaded' in health_data:
            st.sidebar.info(f"{health_data.get('docs_loaded', 0)} documents loaded")
    else:
        st.sidebar.warning('API returned an error')
except requests.exceptions.ConnectionError:
    st.sidebar.error('Cannot connect to API. Make sure the server is running:\n\n`uvicorn src.api:app --reload`')
except requests.exceptions.RequestException as e:
    st.sidebar.error(f'API error: {str(e)}')

# Search interface
query = st.text_input('Query', placeholder='Enter your search query...')
top_k = st.slider('Top K', 1, 20, 5)
expand = st.checkbox('Query expansion (WordNet + embedding neighbors)', value=False)

if st.button('Search', type='primary'):
    if not query.strip():
        st.warning('Please enter a search query')
    else:
        try:
            payload = {"query": query, "top_k": top_k, "expand": expand}
            with st.spinner('Searching...'):
                res = requests.post(f'{API_URL}/search', json=payload, timeout=30)
                res.raise_for_status()
                data = res.json()
            
            if data.get('results'):
                st.success(f"Found {data.get('count', 0)} results")
                for r in data.get('results', []):
                    with st.container():
                        st.subheader(f"{r['doc_id']} (score={r['score']:.3f})")
                        st.write(r['preview'])
                        with st.expander("Match details"):
                            st.json(r['match_info'])
                        st.divider()
            else:
                st.info('No results found')
                
        except requests.exceptions.ConnectionError:
            st.error('Cannot connect to API server. Please make sure it is running:\n\n```bash\nuvicorn src.api:app --reload --port 8000\n```')
        except requests.exceptions.Timeout:
            st.error('Request timed out. The search is taking too long.')
        except requests.exceptions.HTTPError as e:
            st.error(f'API error: {e.response.status_code} - {e.response.text}')
        except Exception as e:
            st.error(f'Error: {str(e)}')