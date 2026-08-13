import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model="openrouter/free"

def spell_check_query(query:str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
                        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                        Preserve punctuation and capitalization unless a change is required for a typo fix.
                        If there are no spelling errors, or if you're unsure, output the original query unchanged.
                        Output only the final query text, nothing else.
                        User query: "{query}"
                        """
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    corrected = (response.choices[0].message.content or "").strip().strip('"')
    return corrected if corrected else query

def rewrite_query(query:str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                            Consider:
                            - Common movie knowledge (famous actors, popular films)
                            - Genre conventions (horror = scary, animation = cartoon)
                            - Keep the rewritten query concise (under 10 words)
                            - It should be a Google-style search query, specific enough to yield relevant results
                            - Don't use boolean logic

                            Examples:
                            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                            If you cannot improve the query, output the original unchanged.
                            Output only the rewritten query text, nothing else.

                            User query: "{query}"
                            """
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    rewritten = (response.choices[0].message.content or "").strip().strip('"')
    return rewritten if rewritten else query

def expand_query(query:str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.
                            Add synonyms and related concepts that might appear in movie descriptions.
                            Keep expansions relevant and focused.
                            Output only the additional terms; they will be appended to the original query.

                            Examples:
                            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                            - "action movie with bear" -> "action thriller bear chase fight adventure"
                            - "comedy with bear" -> "comedy funny bear humor lighthearted"

                            User query: "{query}"
                            """
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    expanded_terms = (response.choices[0].message.content or "").strip().strip('"')
    return f"{query} {expanded_terms}".strip()
    
def rerank_results(query,doc, method="individual"):
    if method == "individual":
        prompt = f"""Rate how well this movie matches the search query.
                                Query: "{query}"
                                Movie: {doc.get("title", "")} - {doc.get("document", "")}

                                Consider:
                                - Direct relevance to query
                                - User intent (what they're looking for)
                                - Content appropriateness

                                Rate 0-10 (10 = perfect match).
                                Output ONLY the number in your response, no other text or explanation.

                                Score:"""

    elif method == "batch":
            prompt = f"""Rank the movies listed below by relevance to the following search query.

                    Query: "{query}"

                    Movies:
                    {doc}

                    Return the movie IDs in order of relevance, best match first.

                    Your response must be a raw JSON array of integers.
                    Do not wrap the JSON in Markdown. Do not use a ```json code block.
                    Do not include any explanatory text.

                    For example:
                    [75, 12, 34, 2, 1]

                    Ranking:"""
    response = client.chat.completions.create(
    model = model,
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
)
    rerank = (response.choices[0].message.content or "").strip().strip('"')
    if not rerank:
        return "0" if method == "individual" else "[]"
    return rerank

def evaluate_results(query, results):
    formatted_results = [
    f"{i}. {r['title']} - {r['document'][:200]}"
    for i, r in enumerate(results, start=1)
    ]
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(formatted_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers other than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""
                
    response = client.chat.completions.create(
    model = model,
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
)
    evaluation = (response.choices[0].message.content or "").strip().strip('"')
    return evaluation if evaluation else "[]"

def call_rag_agent(query:str, docs:str) -> str:
    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {docs}

            Answer:"""
            
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = (response.choices[0].message.content or "").strip()
    return answer if answer else "I don't know."

def summarize_text(query:str,results) -> str:
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

            This should be tailored to Webflyx users. Webflyx is a movie streaming service.

            Query: {query}

            Search results:
            {results}

            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
                        
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    summary = (response.choices[0].message.content or "").strip()
    return summary if summary else "No summary available."

def generate_citations(query:str,documents) -> str:
    prompt = f"""Answer the query below and give information based on the provided documents.

            The answer should be tailored to users of Webflyx, a movie streaming service.
            If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

            Query: {query}

            Documents:
            {documents}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources in the format [1], [2], etc. when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the provided documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
                                    
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    citations = (response.choices[0].message.content or "").strip()
    return citations if citations else "No citations available."

def generate_answer(query:str,question,context) -> str:
    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

    Question: {question}

    Documents:
    {context}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""
                                    
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = (response.choices[0].message.content or "").strip()
    return answer if answer else "No answer available."