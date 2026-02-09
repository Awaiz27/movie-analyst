"""
TMDB Tools with Zero Duplication
Leverages all TMDB API capabilities: Search, Discover, Trending, and Rich Details
Supports Movies, TV Shows, and People with consistent interface
"""
import httpx
from typing import Literal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from llama_index.core.tools import FunctionTool
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class ToolError(Exception):
    """Base exception for tool errors"""
    pass


class APIError(ToolError):
    """External API errors"""
    pass


class NotFoundError(ToolError):
    """Resource not found"""
    pass


# === Utility Functions ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError),
    before_sleep=lambda retry_state: logger.warning(
        f"TMDB tool retry {retry_state.attempt_number}: {retry_state.outcome.exception()}"
    )
)
async def _make_tmdb_request(endpoint: str, params: dict = None, method: str = "GET") -> dict:
    """
    Unified TMDB API request handler with retry logic.
    Handles all TMDB API calls to reduce duplication.
    
    Args:
        endpoint: API endpoint without base URL (e.g., "/search/multi", "/movie/550")
        params: Query parameters
        method: HTTP method (GET, POST, etc.)
        
    Returns:
        JSON response from TMDB
        
    Raises:
        NotFoundError: If 404 response
        APIError: If other HTTP error or network issue
    """
    if params is None:
        params = {}
    
    params["api_key"] = settings.TMDB_API_KEY
    url = f"{settings.TMDB_BASE_URL}{endpoint}"
    
    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
            logger.debug(f"TMDB {method}: {endpoint}")
            
            if method == "GET":
                resp = await client.get(url, params=params)
            else:
                resp = await client.request(method, url, params=params)
            
            resp.raise_for_status()
            return resp.json()
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise NotFoundError(f"Resource not found at {endpoint}")
        logger.error(f"TMDB HTTP {e.response.status_code}: {endpoint}")
        raise APIError(f"TMDB API error: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Network error: {e}")
        raise APIError(f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error at {endpoint}: {e}")
        raise APIError(f"Unexpected error: {str(e)}")


def _format_item(item: dict, media_type: str) -> dict:
    """
    Format item from TMDB response to consistent structure.
    Handles Movies, TV Shows, and People.
    """
    if media_type == "movie" or (isinstance(item, dict) and item.get("media_type") == "movie"):
        return {
            "type": "movie",
            "title": item.get("title"),
            "id": item.get("id"),
            "rating": item.get("vote_average"),
            "release_date": item.get("release_date"),
            "poster": item.get("poster_path"),
            "overview": item.get("overview")[:200] + "..." if item.get("overview") else None,
            "popularity": item.get("popularity")
        }
    elif media_type == "tv" or (isinstance(item, dict) and item.get("media_type") == "tv"):
        return {
            "type": "tv",
            "name": item.get("name") or item.get("title"),
            "id": item.get("id"),
            "rating": item.get("vote_average"),
            "first_air_date": item.get("first_air_date"),
            "poster": item.get("poster_path"),
            "overview": item.get("overview")[:200] + "..." if item.get("overview") else None,
            "popularity": item.get("popularity")
        }
    elif media_type == "person" or (isinstance(item, dict) and item.get("media_type") == "person"):
        return {
            "type": "person",
            "name": item.get("name"),
            "id": item.get("id"),
            "profile_path": item.get("profile_path"),
            "known_for_department": item.get("known_for_department"),
            "popularity": item.get("popularity")
        }
    return item



# === Unified Search (Movies, TV, People, Collections) ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def unified_search(query: str, limit: int = 20) -> dict:
    """
    Universal search across movies, TV shows, people, and collections.
    Uses TMDB /search/multi endpoint for comprehensive results.
    
    Args:
        query: Search query (title, name, etc.)
        limit: Maximum results to return
        
    Returns:
        Dictionary with movies, tv_shows, people, and collections organized by type
    """
    logger.info(f"Universal search: {query}")
    
    try:
        data = await _make_tmdb_request("/search/multi", {
            "query": query,
            "page": 1,
            "include_adult": False
        })
        
        results = data.get("results", [])
        
        # Organize by media type
        movies = [_format_item(r, "movie") for r in results if r.get("media_type") == "movie"][:limit]
        tv_shows = [_format_item(r, "tv") for r in results if r.get("media_type") == "tv"][:limit]
        people = [_format_item(r, "person") for r in results if r.get("media_type") == "person"][:limit]
        
        logger.info(f"✓ Found {len(movies)} movies, {len(tv_shows)} TV shows, {len(people)} people")
        
        return {
            "query": query,
            "total_results": len(results),
            "movies": movies,
            "tv_shows": tv_shows,
            "people": people
        }
        
    except NotFoundError:
        return {"query": query, "total_results": 0, "movies": [], "tv_shows": [], "people": []}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise APIError(f"Failed to search: {str(e)}")


# === Unified Details (Movie, TV, Person) ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_detailed_info(item_id: int, media_type: Literal["movie", "tv", "person"] = "movie") -> dict:
    """
    Fetch comprehensive details for any media item.
    Includes credits, recommendations, watch providers, keywords, and more.
    
    Args:
        item_id: TMDB ID of the item
        media_type: "movie", "tv", or "person"
        
    Returns:
        Comprehensive details about the item with rich metadata
    """
    logger.info(f"Fetching {media_type} details: {item_id}")
    
    try:
        # Main details
        detail_data = await _make_tmdb_request(f"/{media_type}/{item_id}", {
            "append_to_response": "credits,recommendations,external_ids,keywords,images"
        })
        
        if media_type == "movie":
            return {
                "type": "movie",
                "title": detail_data.get("title"),
                "id": detail_data.get("id"),
                "rating": detail_data.get("vote_average"),
                "vote_count": detail_data.get("vote_count"),
                "release_date": detail_data.get("release_date"),
                "runtime": detail_data.get("runtime"),
                "budget": detail_data.get("budget"),
                "revenue": detail_data.get("revenue"),
                "status": detail_data.get("status"),
                "overview": detail_data.get("overview"),
                "genres": [g.get("name") for g in detail_data.get("genres", [])],
                "production_companies": [p.get("name") for p in detail_data.get("production_companies", [])][:3],
                "cast": [
                    {"name": c.get("name"), "character": c.get("character")}
                    for c in detail_data.get("credits", {}).get("cast", [])[:5]
                ],
                "director": next(
                    (c.get("name") for c in detail_data.get("credits", {}).get("crew", []) if c.get("job") == "Director"),
                    None
                ),
                "keywords": [k.get("name") for k in detail_data.get("keywords", {}).get("keywords", [])][:5],
                "recommendations": [
                    _format_item(r, "movie") for r in detail_data.get("recommendations", {}).get("results", [])[:5]
                ],
                "external_ids": detail_data.get("external_ids", {})
            }
            
        elif media_type == "tv":
            return {
                "type": "tv",
                "name": detail_data.get("name"),
                "id": detail_data.get("id"),
                "rating": detail_data.get("vote_average"),
                "vote_count": detail_data.get("vote_count"),
                "first_air_date": detail_data.get("first_air_date"),
                "last_air_date": detail_data.get("last_air_date"),
                "status": detail_data.get("status"),
                "number_of_seasons": detail_data.get("number_of_seasons"),
                "number_of_episodes": detail_data.get("number_of_episodes"),
                "episode_runtime": detail_data.get("episode_run_time"),
                "overview": detail_data.get("overview"),
                "genres": [g.get("name") for g in detail_data.get("genres", [])],
                "networks": [n.get("name") for n in detail_data.get("networks", [])],
                "cast": [
                    {"name": c.get("name"), "character": c.get("character")}
                    for c in detail_data.get("credits", {}).get("cast", [])[:5]
                ],
                "keywords": [k.get("name") for k in detail_data.get("keywords", {}).get("results", [])][:5],
                "recommendations": [
                    _format_item(r, "tv") for r in detail_data.get("recommendations", {}).get("results", [])[:5]
                ],
                "external_ids": detail_data.get("external_ids", {})
            }
            
        elif media_type == "person":
            return {
                "type": "person",
                "name": detail_data.get("name"),
                "id": detail_data.get("id"),
                "birthday": detail_data.get("birthday"),
                "death_day": detail_data.get("deathday"),
                "known_for_department": detail_data.get("known_for_department"),
                "popularity": detail_data.get("popularity"),
                "biography": detail_data.get("biography")[:500] + "..." if detail_data.get("biography") else None,
                "profile_path": detail_data.get("profile_path"),
                "external_ids": detail_data.get("external_ids", {}),
                "known_for": [
                    _format_item(k, k.get("media_type", "movie"))
                    for k in detail_data.get("known_for", [])[:5]
                ]
            }
            
    except NotFoundError:
        raise NotFoundError(f"{media_type.capitalize()} with ID {item_id} not found")
    except Exception as e:
        logger.error(f"Details fetch error: {e}")
        raise APIError(f"Failed to fetch details: {str(e)}")



# === Unified Trending/Popular/Top-Rated ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_trending_media(
    time_window: Literal["day", "week"] = "week",
    include_type: Literal["all", "movie", "tv", "person"] = "all",
    limit: int = 20
) -> dict:
    """
    Get currently trending content across all media types.
    
    Args:
        time_window: "day" or "week"
        include_type: "all" for mixed, or specific type
        limit: Results to return
        
    Returns:
        List of trending items by type
    """
    logger.info(f"Fetching trending ({time_window}): {include_type}")
    
    try:
        endpoint = f"/trending/{include_type}/{time_window}"
        data = await _make_tmdb_request(endpoint, {"page": 1})
        
        results = data.get("results", [])
        formatted = [_format_item(r, r.get("media_type", include_type)) for r in results][:limit]
        
        logger.info(f"✓ Found {len(formatted)} trending {include_type}")
        
        return {
            "time_window": time_window,
            "media_type": include_type,
            "total": len(formatted),
            "items": formatted
        }
        
    except Exception as e:
        logger.error(f"Trending fetch error: {e}")
        raise APIError(f"Failed to fetch trending: {str(e)}")


# === Unified Popular/Top-Rated ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_popular_media(
    media_type: Literal["movie", "tv"] = "movie",
    limit: int = 20
) -> dict:
    """
    Get most popular movies or TV shows.
    
    Args:
        media_type: "movie" or "tv"
        limit: Results to return
        
    Returns:
        List of popular items
    """
    logger.info(f"Fetching popular {media_type}s")
    
    try:
        endpoint = f"/{media_type}/popular"
        data = await _make_tmdb_request(endpoint, {"page": 1})
        
        results = data.get("results", [])
        formatted = [_format_item(r, media_type) for r in results][:limit]
        
        logger.info(f"✓ Found {len(formatted)} popular {media_type}s")
        
        return {
            "media_type": media_type,
            "category": "popular",
            "total": len(formatted),
            "items": formatted
        }
        
    except Exception as e:
        logger.error(f"Popular fetch error: {e}")
        raise APIError(f"Failed to fetch popular: {str(e)}")


@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_top_rated_media(
    media_type: Literal["movie", "tv"] = "movie",
    limit: int = 20
) -> dict:
    """
    Get highest-rated movies or TV shows.
    
    Args:
        media_type: "movie" or "tv"
        limit: Results to return
        
    Returns:
        List of top-rated items
    """
    logger.info(f"Fetching top-rated {media_type}s")
    
    try:
        endpoint = f"/{media_type}/top_rated"
        data = await _make_tmdb_request(endpoint, {"page": 1})
        
        results = data.get("results", [])
        formatted = [_format_item(r, media_type) for r in results][:limit]
        
        logger.info(f"✓ Found {len(formatted)} top-rated {media_type}s")
        
        return {
            "media_type": media_type,
            "category": "top_rated",
            "total": len(formatted),
            "items": formatted
        }
        
    except Exception as e:
        logger.error(f"Top-rated fetch error: {e}")
        raise APIError(f"Failed to fetch top-rated: {str(e)}")


# === Specialized Lists ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_upcoming(limit: int = 20) -> dict:
    """Get upcoming movie releases"""
    logger.info("Fetching upcoming movies")
    try:
        data = await _make_tmdb_request("/movie/upcoming", {"page": 1})
        results = [_format_item(r, "movie") for r in data.get("results", [])][:limit]
        return {"category": "upcoming", "total": len(results), "items": results}
    except Exception as e:
        raise APIError(f"Failed to fetch upcoming: {str(e)}")


@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_airing_today(limit: int = 20) -> dict:
    """Get TV episodes airing today"""
    logger.info("Fetching TV airing today")
    try:
        data = await _make_tmdb_request("/tv/airing_today", {"page": 1})
        results = [_format_item(r, "tv") for r in data.get("results", [])][:limit]
        return {"category": "airing_today", "total": len(results), "items": results}
    except Exception as e:
        raise APIError(f"Failed to fetch airing today: {str(e)}")


@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_on_the_air(limit: int = 20) -> dict:
    """Get currently on-the-air TV shows"""
    logger.info("Fetching TV on the air")
    try:
        data = await _make_tmdb_request("/tv/on_the_air", {"page": 1})
        results = [_format_item(r, "tv") for r in data.get("results", [])][:limit]
        return {"category": "on_the_air", "total": len(results), "items": results}
    except Exception as e:
        raise APIError(f"Failed to fetch on the air: {str(e)}")


# === Recommendations & Similar ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_similar(item_id: int, media_type: Literal["movie", "tv"] = "movie", limit: int = 15) -> dict:
    """
    Get similar movies or TV shows based on a given item.
    
    Args:
        item_id: TMDB ID
        media_type: "movie" or "tv"
        limit: Results to return
        
    Returns:
        Similar items
    """
    logger.info(f"Fetching similar {media_type}s to {item_id}")
    try:
        data = await _make_tmdb_request(f"/{media_type}/{item_id}/similar", {"page": 1})
        results = [_format_item(r, media_type) for r in data.get("results", [])][:limit]
        return {"base_id": item_id, "media_type": media_type, "total": len(results), "items": results}
    except Exception as e:
        raise APIError(f"Failed to fetch similar: {str(e)}")


@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def get_recommendations(item_id: int, media_type: Literal["movie", "tv"] = "movie", limit: int = 15) -> dict:
    """
    Get recommended movies or TV shows based on a given item.
    
    Args:
        item_id: TMDB ID
        media_type: "movie" or "tv"
        limit: Results to return
        
    Returns:
        Recommended items
    """
    logger.info(f"Fetching recommendations for {media_type} {item_id}")
    try:
        data = await _make_tmdb_request(f"/{media_type}/{item_id}/recommendations", {"page": 1})
        results = [_format_item(r, media_type) for r in data.get("results", [])][:limit]
        return {"base_id": item_id, "media_type": media_type, "total": len(results), "items": results}
    except Exception as e:
        raise APIError(f"Failed to fetch recommendations: {str(e)}")


# === Discover with Advanced Filters ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def discover_with_filters(
    media_type: Literal["movie", "tv"] = "movie",
    min_rating: float = 0,
    max_rating: float = 10,
    year: int = None,
    genres: str = None,
    sort_by: str = "vote_average.desc",
    limit: int = 20
) -> dict:
    """
    Advanced discovery with multiple filters.
    
    Args:
        media_type: "movie" or "tv"
        min_rating: Minimum rating (0-10)
        max_rating: Maximum rating (0-10)
        year: Release/air year
        genres: Comma-separated genre IDs
        sort_by: Sort criteria (vote_average.desc, popularity.desc, etc.)
        limit: Results
        
    Returns:
        Filtered results
    """
    logger.info(f"Discovering {media_type} with filters")
    
    try:
        endpoint = f"/discover/{media_type}"
        params = {
            "vote_average.gte": min_rating,
            "vote_average.lte": max_rating,
            "sort_by": sort_by,
            "page": 1
        }
        
        if year:
            params[f"primary_{'release_' if media_type == 'movie' else 'air_'}date.year"] = year
        if genres:
            params["with_genres"] = genres
        
        data = await _make_tmdb_request(endpoint, params)
        results = [_format_item(r, media_type) for r in data.get("results", [])][:limit]
        
        return {
            "media_type": media_type,
            "filters": {
                "rating_range": f"{min_rating}-{max_rating}",
                "year": year,
                "genres": genres,
                "sort_by": sort_by
            },
            "total": len(results),
            "items": results
        }
        
    except Exception as e:
        logger.error(f"Discover error: {e}")
        raise APIError(f"Failed to discover: {str(e)}")


# === External ID Search ===
@retry(
    stop=stop_after_attempt(settings.HTTP_RETRIES),
    wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_FACTOR, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
async def find_by_id(external_id: str, external_source: str = "imdb_id") -> dict:
    """
    Find movies/TV/people by external ID (IMDB, TVDb, etc.)
    
    Args:
        external_id: The ID (e.g., "tt0111161" for IMDB)
        external_source: "imdb_id", "tvdb_id", "facebook_id", etc.
        
    Returns:
        Matching movies, TV shows, and people
    """
    logger.info(f"Finding by {external_source}: {external_id}")
    
    try:
        data = await _make_tmdb_request(f"/find/{external_id}", {
            "external_source": external_source
        })
        
        movies = [_format_item(m, "movie") for m in data.get("movie_results", [])]
        tv_shows = [_format_item(t, "tv") for t in data.get("tv_results", [])]
        people = [_format_item(p, "person") for p in data.get("person_results", [])]
        
        return {
            "external_id": external_id,
            "external_source": external_source,
            "total_results": len(movies) + len(tv_shows) + len(people),
            "movies": movies,
            "tv_shows": tv_shows,
            "people": people
        }
        
    except NotFoundError:
        return {
            "external_id": external_id,
            "external_source": external_source,
            "total_results": 0,
            "movies": [],
            "tv_shows": [],
            "people": []
        }


# === LlamaIndex Tool Exports ===

search_tool = FunctionTool.from_defaults(
    fn=unified_search,
    name="search_tmdb",
    description="Universal search across movies, TV shows, people, and collections. Returns organized results by type."
)

details_tool = FunctionTool.from_defaults(
    fn=get_detailed_info,
    name="get_media_details",
    description="Get comprehensive details including cast, crew, recommendations, keywords, and external IDs for movies, TV, or people."
)

trending_tool = FunctionTool.from_defaults(
    fn=get_trending_media,
    name="get_trending_media",
    description="Get currently trending movies, TV shows, or people. Choose by day or week window."
)

popular_tool = FunctionTool.from_defaults(
    fn=get_popular_media,
    name="get_popular_media",
    description="Get most popular movies or TV shows currently."
)

top_rated_tool = FunctionTool.from_defaults(
    fn=get_top_rated_media,
    name="get_top_rated_media",
    description="Get highest-rated movies or TV shows of all time."
)

upcoming_tool = FunctionTool.from_defaults(
    fn=get_upcoming,
    name="get_upcoming_movies",
    description="Get upcoming movie releases."
)

airing_today_tool = FunctionTool.from_defaults(
    fn=get_airing_today,
    name="get_airing_today",
    description="Get TV episodes airing today."
)

on_the_air_tool = FunctionTool.from_defaults(
    fn=get_on_the_air,
    name="get_on_the_air_tv",
    description="Get currently on-the-air TV shows."
)

similar_tool = FunctionTool.from_defaults(
    fn=get_similar,
    name="get_similar_media",
    description="Get movies or TV shows similar to a given item."
)

recommendations_tool = FunctionTool.from_defaults(
    fn=get_recommendations,
    name="get_recommendations",
    description="Get movie or TV show recommendations based on a specific item."
)

discover_tool = FunctionTool.from_defaults(
    fn=discover_with_filters,
    name="discover_with_filters",
    description="Advanced discovery with rating range, year, genres, and sorting options."
)

find_id_tool = FunctionTool.from_defaults(
    fn=find_by_id,
    name="find_by_external_id",
    description="Find content by external IDs like IMDB ID or TVDb ID."
)

# Legacy aliases for backward compatibility
movie_tool = search_tool
tv_tool = search_tool
now_playing_tool = upcoming_tool
similar_movies_tool = similar_tool
discover_movies_tool = discover_tool
find_by_id_tool = find_id_tool
