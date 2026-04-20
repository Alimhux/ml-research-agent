import logging
logging.basicConfig(level=logging.INFO)

from src.agent.tools import arxiv_search

papers = arxiv_search.invoke({'query': 'group relative policy optimization', 'max_results': 3})
for p in papers:
    print(p.pdf_url, '|', p.published_at, '|', p.title[:70])

print('OK, total:', len(papers))
print('Tool schema:', arxiv_search.args_schema.model_json_schema())
