from ingest import load_data, build_index


data = load_data()
index = build_index(data)


INSTRUCTIONS = """
Your task is to answer questions based on the provided context.
Use the context to find relevant information and provide accurate answers.
If the answer is not found in the context, respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
    Question: {question}

    Context: {context}
"""

class RAGBase:
    def __init__(
            self, 
            index,
            llm_client,
            instructions=INSTRUCTIONS,
            category=None,
            prompt_template=USER_PROMPT_TEMPLATE,
            model='gpt-5.4-mini'
        ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.category = category
        self.prompt_template = prompt_template
        self.model = model

    def search(self, question):
        boost_dict = {'question': 2, 'answer': 1}
        filter_dict = {'category': self.category} if self.category else {}

        return self.index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=5
        )


    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()


    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        
        prompt = self.prompt_template.format(
            question=question,
            context=context
        )
        
        return prompt.strip()
    
    def llm(self, prompt):
        message_history = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
                model=self.model,
                input=message_history
            )
        
        return response.output_text
        
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer