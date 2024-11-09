from gist_memory_storage import GistMemoryStorage
import re
from episode import  Episode
from nlp.page_selector  import PageSelector
from nlp.qa_chain import QAchain
import logging
from pagination_strategy import paginate_with_natural_breakpoints, pagination_with_dynamic_paragraphs

logging.basicConfig(level=logging.INFO)


    
class ReadAgent:
    """ReadAgent class to handle the reading and answering of queries."""
    def __init__(self, text: str, storage: GistMemoryStorage, page_selector: PageSelector, qa_chain: QAchain, model_name: str = 'sshleifer/distilbart-cnn-12-6'):
        """Initialize the ReadAgent with the given text and model name.
        Args:
            text (str): The text to process.
            model_name (str): The model name to use for QA.
        """

        self.text = text
        self.storage = GistMemoryStorage(model_name=model_name)
        self.page_selector = PageSelector()
        self.qa_chain = QAchain()

    async def paginate_text(self, min_words, max_words: int = 512, overlap: int = 32, pagination_type: str = 'logical_breakpoints', paragraph_separator: str = "\n\n") -> list[str]:
        """Paginate the text using the specified pagination strategy.
        Args:
            min_words (int): The minimum number of words in each episode.
            max_words (int): The maximum number of words in each episode.
            overlap (int): The number of overlapping words between episodes.
            pagination_type (str): The pagination strategy to use.
            paragraph_separator (str): The separator between paragraphs.
        Returns:
            list[str]: A list of paginated episodes.
        """
        episodes = []
        
        if pagination_type == 'logical_breakpoints':
            episodes =   await paginate_with_natural_breakpoints(text=self.text, min_words=min_words, max_words=max_words, overlap=overlap, paragraph_separator=paragraph_separator)
        elif pagination_type == 'dynamic_paragraphs':
            episodes = await pagination_with_dynamic_paragraphs(text=self.text, min_words=min_words, max_words=max_words, overlap=overlap, paragraph_separator=paragraph_separator) 
        elif pagination_type == 'splitby_patterns':
            pattern = r"\n\s*(I|II|III|IV|V|VI|VII|VIII|IX|X+|Chapter|Section|-{10,})\s*\n"
            breakpoints = [match.start() for match in re.finditer(pattern, self.text)]
            start_idx = 0
            for end_idx in breakpoints:
                episodes.append(self.text[start_idx:end_idx].strip())
                start_idx = end_idx
            episodes.append(self.text[start_idx:].strip())  # Add the last segment
        elif pagination_type == 'sliding_window':
            words = self.text.split()
            for i in range(0, len(words), max_words - overlap):
                episodes.append(' '.join(words[i:i + max_words]))
        else:  # Default to 'paragraph' pagination
            paragraphs = [para.strip() for para in self.text.split('\n\n') if para.strip()]
            buffer = ""
            char_limit = max_words * 6  # Rough estimate assuming average word length
            for para in paragraphs:
                # Append paragraph to buffer, checking if it exceeds the character limit after addition
                next_buffer = buffer + (" " if buffer else "") + para
                if len(next_buffer) > char_limit:
                    if buffer:
                        episodes.append(buffer)
                        buffer = para  # Start a new buffer with the current paragraph
                else:
                    buffer = next_buffer
            if buffer:
                episodes.append(buffer)

        episodes = set(episodes)  # Remove duplicates
        for content in episodes:
            if content not in self.storage.episodes and content.strip():
                self.storage.add_episode(content)

        return episodes
    
    async def merge_gists_using_breakpoints(self) -> None:
        """Merge the stored gists using natural breakpoints."""

        # Initial steps to prepare and find breakpoints, as previously described
        
        # Assuming self.storage.episodes is prepared and breakpoints are identified
        merged_gists, merged_contents = await self.prepare_and_merge_gists_and_contents()
        
        # Clear the existing episodes to replace with the new, merged ones
        self.storage.episodes.clear()
        
        for idx, (gist, content) in enumerate(zip(merged_gists, merged_contents)):
            # Create a new episode for each merged gist and content
            gist_summary = await self.storage._summarize_text(gist)

            new_episode = Episode(content=content, id=idx+1, gist=gist_summary)
            self.storage.episodes.append(new_episode)


    async def prepare_and_merge_gists_and_contents(self) -> tuple[list[str], list[str]]:
        """Prepare and merge the stored gists using natural breakpoints.
        Returns:
            tuple[list[str], list[str]]: The merged gists and contents.
        """
        # Step 1: Prepare concatenated gists with separators
        concatenated_gists = ["\n({}) {}]\n".format(i, episode.gist) for i, episode in enumerate(self.storage.episodes)]
        concatenated_gists = "\n".join(concatenated_gists)

        concatenated_content = ["\n({}) {}]\n".format(i, episode.content) for i, episode in enumerate(self.storage.episodes)]
        concatenated_content = "\n".join(concatenated_content)


        # Step 2: Find natural breakpoints within the concatenated gists
        breakpoints, _ = await self.find_natural_breakpoints(
            concatenated_gists,
            min_words=100,  # Adjust based on desired granularity
            max_words=1000,  # Adjust based on desired granularity
            overlap=0,  # no overlap needed for gists
        )


        merged_gists = []
        merged_contents = []
        
        current_gist_group = []
        current_content_group = []

        for episode in self.storage.episodes:
            current_gist_group.append(episode.gist)
            current_content_group.append(episode.content)
            
            if episode.id in breakpoints:
                # When reaching a breakpoint, merge the current groups and start new ones
                merged_gists.append("\n".join(filter(None, current_gist_group)))  # Join non-empty gists
                merged_contents.append("\n".join(current_content_group))
                current_gist_group = []
                current_content_group = []
        
        # Don't forget to add the last group if not empty
        if current_gist_group or current_content_group:
            merged_gists.append("\n".join(filter(None, current_gist_group)))  # Join non-empty gists
            merged_contents.append("\n".join(current_content_group))

        return merged_gists, merged_contents


    async def process_text(self, 
                           min_words: int = 32, 
                           max_words: int = 512, 
                           pagination_type: str = 'logical_breakpoints',
                           paragraph_separator: str = '\n\n',
                           overlap: int = 32,
                           ) -> list[str]:
        """Process the text and return the paginated episodes.
        Args:
            min_words (int): The minimum number of words in each episode.
            max_words (int): The maximum number of words in each episode.
            pagination_type (str): The pagination strategy to use.
            paragraph_separator (str): The separator between paragraphs.
            overlap (int): The number of overlapping words between episodes.
        Returns:
            list[str]: A list of paginated episodes.
        """
        # Choose appropriate pagination type or parameters based on your text's structure
        episodes = await self.paginate_text(min_words=min_words, max_words=max_words,  overlap=overlap, pagination_type=pagination_type, paragraph_separator=paragraph_separator)

        await self.storage.summarize_episodes()
        # self.storage.vectorize_gists()
        return episodes
    
    def answer_query_using_vector_search(self, query: str) -> list[Episode]:
        """Answer a query using the stored gists.
        Args:
            query (str): The query to answer.
        Returns:
            list[Episode]: A list of relevant episodes.
        """
        # Use self.storage.search_gists to find relevant episodes based on the query
        return self.storage.search_gists(query)
    
    async def answer_using_gists(self, query: str, use_sliding_window: bool = False) -> list[Episode]:
        """Answer a query using the stored gists. 
        Which are loaded into context into ChatGPT.
        ChatGPT selects the appropriate gist to answer the query.
        Args:
            query (str): The query to answer.
            use_sliding_window (bool): Whether to use sliding window for gists.
        Returns:
            list[Episode]: A list of relevant episodes.
        """

        # get gists
        gists = [episode.gist for episode in self.storage.episodes]
        # Concatenate gists into string. Before each gist add Page number.
        # This is to help the model to select the correct gist.
        gists = [f"\nPage {i+1}:\n{gist}" for i, gist in enumerate(gists)]

        total_chars_gist = sum([len(episode.gist) for episode in self.storage.episodes])
        avg_token_length = 4
        total_number_of_tokens = total_chars_gist / avg_token_length

        total_pages = []

        if use_sliding_window:
            # use sliding window over gists, chunking
            
            start = 0
            end = 6
            while start <= len(gists):
                gists_chunk = gists[start:end]
                gists_chunk = " ".join(gists_chunk)
                
                pages, _ = await self.page_selector.generate(gists_chunk, query)
                total_pages.extend(pages)
                start += 6
                end += 6
                print(start, end, pages)
        else:
            gists = "\n".join(gists)
            pages, _ = await self.page_selector.generate(gists, query)
            total_pages.extend(pages)
        # select content according page index

        for page in total_pages:
            self.storage.episodes[page-1].content
        content = [self.storage.episodes[page-1].content for page in list(set(total_pages))]
        content = "\n\n".join(content)

        answer = await self.qa_chain.generate(content, query)

        relevant_episodes = [self.storage.episodes[page - 1] for page in list(set(total_pages))]
        return relevant_episodes, answer
