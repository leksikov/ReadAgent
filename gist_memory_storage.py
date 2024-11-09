from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer
from typing import List, Optional
from episode import Episode
import numpy as np
import json
import logging
from nlp.summarizer import Summarizer
import logging

# Configure logging to display INFO level logs or higher in the console
logging.basicConfig(
    level=logging.DEBUG,  # Set the level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Include time, module name, and message in logs
    handlers=[logging.StreamHandler()]  # Ensure logs are output to the console
)

logger = logging.getLogger(__name__)

def serialize_episodes(episodes: List[Episode], filename: str) -> None:
    """
    Serializes a list of Episode objects and saves them to a JSON file.
    
    Args:
        episodes (List[Episode]): List of episode objects to serialize.
        filename (str): Name of the file to save the episodes.
    """
    try:
        episodes_data = []
        for episode in episodes:
            episode_data = episode.model_dump(exclude={"vector"})
            if episode.vector is not None:
                if hasattr(episode.vector, "toarray"):
                    dense_array = episode.vector.toarray()
                else:
                    dense_array = episode.vector
                episode_data['vector'] = dense_array.tolist()
            episodes_data.append(episode_data)

        with open(filename, 'w') as f:
            json.dump(episodes_data, f)
        logger.info(f"Episodes successfully saved to {filename}.")
    except Exception as e:
        logger.error(f"Error serializing episodes: {str(e)}")


def deserialize_episodes(filename: str) -> List[Episode]:
    """
    Deserializes episodes from a JSON file into a list of Episode objects.
    
    Args:
        filename (str): The name of the file containing serialized episodes.
    
    Returns:
        List[Episode]: List of deserialized episode objects.
    """
    try:
        with open(filename, 'r') as f:
            episodes_data = json.load(f)

        episodes = []
        for episode_data in episodes_data:
            vector_data = episode_data.pop('vector', None)
            episode = Episode(**episode_data)
            if vector_data is not None:
                episode.vector = np.array(vector_data)
            episodes.append(episode)

        logger.info(f"Episodes successfully loaded from {filename}.")
        return episodes
    except Exception as e:
        logger.error(f"Error deserializing episodes: {str(e)}")
        return []


class GistMemoryStorage:
    def __init__(self, model_name: str = 'sshleifer/distilbart-cnn-12-6') -> None:
        """
        Initializes the GistMemoryStorage with summarization model and settings.
        
        Args:
            model_name (str): The name of the summarization model to use.
        """
        self.episodes: List[Episode] = []
        self.vectorizer = TfidfVectorizer()
        self.model_name = model_name
        try:
            if model_name == "gpt-4o-mini":
                self.summarizer = Summarizer(temperature=0.6, max_tokens=1024)
                self.max_token_size = 1024
            else:
                self.summarizer = pipeline("summarization", model=model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.max_token_size = self.tokenizer.model_max_length
            logger.info(f"Model {model_name} successfully loaded.")
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {str(e)}")
            raise

        self.min_words = 32
        self.portion_of_max_token_size = 0.9
        self.summary_overlap = 0.1
        self.avg_token_length = 3

    def add_episode(self, content: str) -> None:
        """
        Adds a new episode to the storage.
        
        Args:
            content (str): The content of the episode to be added.
        """
        try:
            episode_id = len(self.episodes) + 1
            episode = Episode(content=content, id=episode_id)
            if episode not in self.episodes:
                self.episodes.append(episode)
            logger.info(f"Episode {episode_id} added successfully.")
        except Exception as e:
            logger.error(f"Error adding episode: {str(e)}")

    def clear_episodes(self) -> None:
        """
        Clears all existing episodes from the storage.
        """
        self.episodes = []
        logger.info("All episodes cleared from storage.")

    async def summarize_episodes(self) -> None:
        """
        Summarizes all episodes in the storage asynchronously.
        """
        for episode in self.episodes:
            try:
                if len(episode.content.split()) > self.min_words:
                    episode.gist = await self._summarize_text(episode.content)
                else:
                    episode.gist = episode.content
                logger.info(f"Summarized episode {episode.id}")
            except Exception as e:
                logger.error(f"Error summarizing episode {episode.id}: {str(e)}")

    async def _summarize_text(self, text: str) -> str:
        """
        Splits text into windows and summarizes each window asynchronously.
        
        Args:
            text (str): The text to be summarized.
        
        Returns:
            str: The final summarized text.
        """
        try:
            tokens = text
            token_window_size = int(self.max_token_size * self.portion_of_max_token_size) * self.avg_token_length
            windows = []
            start_index = 0

            while start_index < len(tokens):
                end_index = min(start_index + token_window_size, len(tokens))
                window = "".join(tokens[start_index:end_index])
                windows.append(window)
                start_index += int(token_window_size * (1 - self.summary_overlap))

            summaries = [await self.summarize_window(window) for window in windows]
            return " ".join(summaries)
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return text

    async def summarize_window(self, window: str) -> str:
        """
        Summarizes a given text window.
        
        Args:
            window (str): The window of text to summarize.
        
        Returns:
            str: The summarized text.
        """
        try:
            if self.model_name == "gpt-4o-mini":
                summary = await self.summarizer.summarize(window)
                return summary.replace("Summary: ", "")
            else:
                return self.summarizer(window, do_sample=False)[0]['summary_text']
        except Exception as e:
            logger.error(f"Error summarizing window: {str(e)}")
            return window

    def vectorize_gists(self) -> None:
        """
        Converts all episode gists into vectors using TF-IDF vectorization.
        """
        try:
            gists = [episode.gist for episode in self.episodes if episode.gist]
            vectors = self.vectorizer.fit_transform(gists)
            for episode, vector in zip(self.episodes, vectors):
                episode.vector = vector
            logger.info("Gists vectorized successfully.")
        except Exception as e:
            logger.error(f"Error vectorizing gists: {str(e)}")

    def search_gists(self, query: str) -> List[Episode]:
        """
        Searches for episodes whose gists are similar to the given query using cosine similarity.
        
        Args:
            query (str): The search query.
        
        Returns:
            List[Episode]: A list of matching episodes ordered by similarity.
        """
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = []
            for episode in self.episodes:
                if episode.vector is not None:
                    similarity = cosine_similarity(query_vec, episode.vector)
                    similarities.append((episode, similarity[0][0]))
            sorted_episodes = sorted(similarities, key=lambda x: x[1], reverse=True)
            return [episode[0] for episode in sorted_episodes if episode[1] > 0]
        except Exception as e:
            logger.error(f"Error searching gists: {str(e)}")
            return []

    def answer_query(self, query: str) -> List[Episode]:
        """
        Returns episodes that match the search query based on gist similarity.
        
        Args:
            query (str): The search query.
        
        Returns:
            List[Episode]: A list of matching episodes.
        """
        return self.search_gists(query)

    def save_to_disk(self, filename: str) -> None:
        """
        Saves all episodes to disk.
        
        Args:
            filename (str): The filename to save the episodes.
        """
        serialize_episodes(self.episodes, filename)

    def load_from_disk(self, filename: str) -> None:
        """
        Loads episodes from disk and re-vectorizes gists.
        
        Args:
            filename (str): The filename to load episodes from.
        """
        self.episodes = deserialize_episodes(filename)
        self.vectorize_gists()

def main() -> None:
    """
    Main function to test the GistMemoryStorage class with basic operations.
    Includes adding episodes, summarization, saving/loading, and searching.
    """
    try:
        # Initialize the GistMemoryStorage
        storage = GistMemoryStorage(model_name='sshleifer/distilbart-cnn-12-6')
        
        # Add episodes
        storage.add_episode("This is the first episode content for testing.")
        storage.add_episode("Here is some more content in the second episode to summarize.")
        storage.add_episode("A brief episode with less than 32 words.")

        # Assert that episodes were added correctly
        assert len(storage.episodes) == 3, "Episodes not added correctly."

        # Run summarization
        import asyncio
        asyncio.run(storage.summarize_episodes())

        # Check if gists are created properly
        assert all(episode.gist is not None for episode in storage.episodes), "Summarization failed."

        # Vectorize the gists for searching
        storage.vectorize_gists()

        # Test searching for a gist
        search_results = storage.search_gists("testing")
        assert len(search_results) > 0, "Search did not return expected results."

        # Test query answering
        query_results = storage.answer_query("summarize")
        assert len(query_results) > 0, "Answer query did not return expected results."

        # Save episodes to disk
        storage.save_to_disk("test_episodes.json")

        # Clear episodes and load them from disk
        storage.clear_episodes()
        assert len(storage.episodes) == 0, "Failed to clear episodes."

        storage.load_from_disk("test_episodes.json")
        assert len(storage.episodes) == 3, "Failed to load episodes from disk."

        # Re-run search after loading from disk
        search_results_after_load = storage.search_gists("content")
        assert len(search_results_after_load) > 0, "Search after loading failed."

        logger.info("All tests passed successfully.")

    except AssertionError as e:
        logger.error(f"AssertionError: {str(e)}")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")


if __name__ == "__main__":
    main()