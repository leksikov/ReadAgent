import argparse
import re
import logging
import asyncio
import yaml
from read_agent import ReadAgent  # Custom agent class to handle reading and querying
from gist_memory_storage import GistMemoryStorage
import re
from episode import  Episode
from nlp.page_selector  import PageSelector
from nlp.qa_chain import QAchain

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to print stats of ReadAgent's episodes
def print_stats(read_agent: ReadAgent) -> None:

    logger.info("Statistics: ")
    logger.info(f"Number of episodes: {len(read_agent.storage.episodes)}")

    # Total number of words in gists
    total_words_gist = sum(len(re.findall(r'\w+', episode.gist)) for episode in read_agent.storage.episodes)
    logger.info(f"Total number of words in gists: {total_words_gist}")

    # Total number of characters in gists
    total_chars_gist = sum(len(episode.gist) for episode in read_agent.storage.episodes)
    logger.info(f"Total number of characters in gists: {total_chars_gist}")

    # Average number of words in gists
    avg_words_gist = total_words_gist / len(read_agent.storage.episodes)
    logger.info(f"Average number of words in gists: {avg_words_gist}")

    # Average number of characters in gists
    avg_chars_gist = total_chars_gist / len(read_agent.storage.episodes)
    logger.info(f"Average number of characters in gists: {avg_chars_gist}")

    # Total number of words in content
    total_words_content = sum(len(re.findall(r'\w+', episode.content)) for episode in read_agent.storage.episodes)
    logger.info(f"Total number of words in content: {total_words_content}")

    # Total number of characters in content
    total_chars_content = sum(len(episode.content) for episode in read_agent.storage.episodes)
    logger.info(f"Total number of characters in content: {total_chars_content}")

    # Average number of words in content
    avg_words_content = total_words_content / len(read_agent.storage.episodes)
    logger.info(f"Average number of words in content: {avg_words_content}")

    # Average number of characters in content
    avg_chars_content = total_chars_content / len(read_agent.storage.episodes)
    logger.info(f"Average number of characters in content: {avg_chars_content}")


# Main async function
async def main() -> None:
    parser = argparse.ArgumentParser(description="ReadAgent: Enhance document understanding with interactive AI.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input text file.')
    parser.add_argument('--query', type=str, help='Query to ask directly without interactive loop.')
    
    # Additional argument parsing
    parser.add_argument('--pagination_type', type=str, choices=['logical_breakpoints', 'logical_dynamic_paragraph', 'paragraph', 'sliding_window', 'splitby_natural_breakpoints'], default='logical_breakpoints', help='Type of pagination for text processing.')
    parser.add_argument('--paragraph_separator', type=str, default='\n\n', help='Paragraph separator for paragraph pagination.')
    parser.add_argument('--min_words', type=int, default=32, help='Minimum words per episode for pagination.')
    parser.add_argument('--max_words', type=int, default=2000, help='Maximum words per episode for pagination.')
    parser.add_argument('--overlap', type=int, default=32, help='Words overlap for sliding window pagination.')
    parser.add_argument('--model_name', type=str, default='sshleifer/distilbart-cnn-12-6', help='Model name for the summarization pipeline.')
    parser.add_argument('--avg_token_length', type=int, default=4, help='Average token length for summarization.')
    parser.add_argument('--summary_overlap', type=float, default=0.1, help='Overlap ratio for summarization.')
    
    args = parser.parse_args()

    # Load text from file
    try:
        if args.input_file.endswith('.txt'):
            with open(args.input_file, "r") as file:
                text = file.read()
                logger.info(f"Successfully loaded text file: {args.input_file}")
        elif args.input_file.endswith('.yaml'):
            with open(args.input_file, 'r') as stream:
                try:
                    data = yaml.safe_load(stream)
                    text = ''.join(f"id: {item['id']}, text: {item['text']}\n\n" for item in data)
                    logger.info(f"Successfully loaded YAML file: {args.input_file}")
                except yaml.YAMLError as exc:
                    logger.error(f"Error loading YAML file: {exc}")
                    return
        else:
            logger.error("Unsupported file format.")
            return
    except FileNotFoundError:
        logger.error(f"File {args.input_file} not found.")
        return
    except Exception as e:
        logger.error(f"Unexpected error while loading file: {e}")
        return

    # Initialize ReadAgent with the loaded text and specified model

    storage = GistMemoryStorage()
    page_selector = PageSelector()
    qa_chain = QAchain()
    read_agent = ReadAgent(text=text, storage=storage, page_selector=page_selector, qa_chain=qa_chain, model_name=args.model_name)
    logger.info("ReadAgent initialized successfully.")

    # Process text by pagination

    """
    await read_agent.process_text(
        min_words=args.min_words,
        max_words=args.max_words,
        pagination_type=args.pagination_type,
        paragraph_separator=args.paragraph_separator,
        overlap=args.overlap
    )
    logger.info("Text processing completed.")
    """
    # Load from disk
    try:
        read_agent.storage.load_from_disk(f'processed/episodes_{args.max_words}.json')
        logger.info(f"Loaded episodes from disk for max words: {args.max_words}")
    except FileNotFoundError:
        logger.error(f"Processed episode file not found for max words: {args.max_words}")
        return
    except Exception as e:
        logger.error(f"Error loading episodes from disk: {e}")
        return

    # Get total number of tokens

    total_chars_gist = sum(len(episode.gist) for episode in read_agent.storage.episodes)
    total_number_of_tokens = total_chars_gist / args.avg_token_length
    print_stats(read_agent)


    # Check if query is provided, else enter interactive loop
    if args.query:
        try:
            logger.info(f"Processing query: {args.query}")
            # Process the query directly
            relevant_episodes, answer = await read_agent.answer_using_gists(args.query)
            if relevant_episodes:
                for episode in relevant_episodes:
                    logger.info(f"Episode {episode.id} - Gist: {episode.gist}")
            else:
                logger.info("No relevant information found for the query.")
            logger.info(f"Answer: {answer}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
    else:
        # Interactive query loop
        try:
            while True:
                query = input("Query: ").strip()
                if query.lower() == 'exit':
                    break
                relevant_episodes, answer = await read_agent.answer_using_gists(query)
                if relevant_episodes:
                    for episode in relevant_episodes:
                        logger.info(f"Episode {episode.id} - Gist: {episode.gist}")
                else:
                    logger.info("No relevant information found for the query.")
                logger.info(f"Answer: {answer}")
        except Exception as e:
            logger.error(f"Error during interactive query loop: {e}")

if __name__ == "__main__":
    asyncio.run(main())

# python main.py --input_file /Users/lablup/Documents/GitHub/ReadAgent/data/gatsby.txt --max_words 2000 --overlap 30 --model_name "gpt-4o-mini"
    
# python main_refactor.py --input_file /Users/lablup/Documents/GitHub/ReadAgent/data/gatsby.txt  --query "Who likes drive cars?"
