from nlp.episode_paginator import EpisodePaginator


async def paginate_with_natural_breakpoints(text: str, min_words: int, max_words: int, overlap: int, paragraph_separator: str ) -> list[str]:
    """Paginate the text using natural breakpoints.
    Args:
        text (str): The text to paginate.
        min_words (int): The minimum number of words in each episode.
        max_words (int): The maximum number of words in each episode.
        overlap (int): The number of overlapping words between episodes.
        paragraph_separator (str): The separator between paragraphs.
    Returns:
        list[str]: A list of paginated episodes.
    """
    annotated_text = annotate_paragraphs_with_labels(text, paragraph_separator=paragraph_separator)
    breakpoints, rationales = await find_natural_breakpoints(annotated_text, min_words=min_words, max_words=max_words, overlap=overlap)
    episodes = split_text_into_episodes(annotated_text, breakpoints)
    return episodes

async def pagination_with_dynamic_paragraphs(text: str, min_words: int, max_words: int, overlap: int, paragraph_separator: str) -> list[str]:
    """Paginate the text using dynamic paragraphs.
    Args:
        text (str): The text to paginate.
        min_words (int): The minimum number of words in each episode.
        max_words (int): The maximum number of words in each episode.
        overlap (int): The number of overlapping words between episodes.
        paragraph_separator (str): The separator between paragraphs.
    Returns:
        list[str]: A list of paginated episodes.
    """
    
    paragraphs = text.split(paragraph_separator)
    episodes = []
    current_chunk_words = []
    current_chunk_text = ""
    global_label_number = 1  # Global label counter for unique paragraph labeling
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:  # Skip empty paragraphs
            continue

        words = paragraph.split()
        if len(current_chunk_words) + len(words) > max_words:
            # If current chunk meets min_words, save it
            if len(current_chunk_words) >= min_words:
                episodes.append(current_chunk_text.rstrip())
                current_chunk_text = ""  # Reset text for the next chunk
                current_chunk_words = []  # Reset words for the next chunk
            # If the current paragraph is too large on its own, it starts a new chunk regardless
            if len(words) > max_words:
                # Handle the case where a single paragraph exceeds max_words
                episodes.append(f"{paragraph}\n⟨{global_label_number}⟩\n".rstrip())
                global_label_number += 1
                continue

        # Add current paragraph to the chunk
        current_chunk_words.extend(words)
        current_chunk_text += f"{paragraph}\n⟨{global_label_number}⟩\n"
        global_label_number += 1  # Increment for the next paragraph
        
    # Add the last chunk if it meets the min_words requirement
    if len(current_chunk_words) >= min_words:
        episodes.append(current_chunk_text.rstrip())

    return episodes

def annotate_paragraphs_with_labels(text: str, paragraph_separator: str) -> str:
    """Annotate paragraphs with labels for natural breakpoints.
    Args:
        text (str): The text to annotate.
        paragraph_separator (str): The separator between paragraphs.
    Returns:
        str: The annotated text with paragraph labels.
    """
    paragraphs = text.split(paragraph_separator)
    annotated_text = ""
    label_number = 1
    for paragraph in paragraphs:
        if paragraph.strip():  # Ensure the paragraph is not just whitespace
            annotated_text += f"{paragraph}\n⟨{label_number}⟩\n"
            label_number += 1
    return annotated_text


async def find_natural_breakpoints(annotated_text: str, min_words: int, max_words: int, overlap: int = 7) -> list[list[int], str]:
    """Find natural breakpoints in the annotated text.
    Args:
        annotated_text (str): The text annotated with paragraph labels.
        min_words (int): The minimum number of words in each episode.
        max_words (int): The maximum number of words in each episode.
        overlap (int): The number of overlapping words between episodes.
    Returns:
        list[int]: A list of breakpoints
        str: The rationale for the breakpoints.
    """

    paginator = EpisodePaginator()
    # Split the text into chunks based on token count
    breakpoints, rationales = await paginator.process_and_generate(annotated_text, min_words, max_words, overlap) # get a breakpoint and rationale
    breakpoints = list(dict.fromkeys(breakpoints))  # Remove duplicate breakpoints
    return breakpoints, rationales

def split_text_into_episodes(annotated_text: str, breakpoints: list[int]) -> list[str]:
    """Split the annotated text into episodes based on breakpoints.
    Args:
        annotated_text (str): The text annotated with paragraph labels.
        breakpoints (list[int]): The list of breakpoints.
    Returns:
        list[str]: A list of episodes.
    """
    episodes = []
    # Convert breakpoints list into a set for faster lookup
    breakpoints_set = set(breakpoints)
    current_episode = []
    current_label = 1  # Assuming label numbering starts at 1

    # Split the text into paragraphs to process it
    paragraphs = annotated_text.split('\n')

    for paragraph in paragraphs:
        # Check if the paragraph contains a breakpoint label
        if f"⟨{current_label}⟩" in paragraph:
            if current_label in breakpoints_set:
                # Join the current episode paragraphs, add to episodes list, and start a new episode
                episodes.append("\n".join(current_episode).strip())
                current_episode = []
            current_label += 1
        # Add the paragraph to the current episode, excluding the label itself
        current_episode.append(paragraph.replace(f"⟨{current_label-1}⟩", "").strip())

    # Add the last episode if it has content
    if current_episode:
        episodes.append("\n".join(current_episode).strip())
    return episodes
