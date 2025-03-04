template = """
Please analyze the context provided below and generate 10 high-level, diverse, and insightful questions that probe critical aspects of the content. For each question, also provide a concise and accurate answer that reflects a deep understanding of the material.

Requirements:
* Generate exactly 30 questions.
* Question types: Include factual, inferential, conceptual, and definitional questions that encourage analysis and critical thinking.
* Relevance: Every question and answer must be directly derived from the provided context. Do not reference the source or mention the document's format (avoid using words like "PDF" or similar).
* Tone: Maintain a formal and professional tone.
* Clarity: Ensure that both questions and answers are clearly worded and easy to understand.
* Conciseness: Answers should be brief yet informative, capturing the core ideas.
* Depth: Focus on generating high-level questions that stimulate deeper insight without stating obvious information.
* Consistency: Present the output in a numbered list format exactly as shown below.

Output format:
Present each question and answer pair in the following format:
Question: [Question 1]
Answer: [Answer 1]

Context: {context}
"""
