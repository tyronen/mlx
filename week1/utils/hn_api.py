import requests
import random

HN_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"

def get_item(item_id: int) -> dict | None:
    """Fetches a single item from the Hacker News API."""
    try:
        response = requests.get(f"{HN_API_BASE_URL}/item/{item_id}.json")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def fetch_random_recent_story() -> dict | None:
    """
    Fetches the list of new stories, picks one randomly ,
    and returns its full data.
    """
    try:
        # Get the list of IDs for the latest stories
        response = requests.get(f"{HN_API_BASE_URL}/newstories.json")
        response.raise_for_status()
        new_story_ids = response.json()

        # Search for a valid story
        search_slice = new_story_ids
        while search_slice:
            # Pick a random ID from our search list
            random_id = random.choice(search_slice)
            item = get_item(random_id)
            
            # Make sure it's a story and not a comment or job posting
            if item and item.get("type") == "story" and "url" in item:
                return item
            
            # If it's not a valid story, remove it from our list and try again
            search_slice.remove(random_id)
        
        # If no valid stories were found in the top 200
        return None

    except requests.exceptions.RequestException:
        return None

