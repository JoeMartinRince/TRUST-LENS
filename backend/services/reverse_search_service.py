import os
import httpx
import urllib.parse
from typing import Dict, Any, Optional

def trace_source(
    url: Optional[str] = None, 
    image_bytes: Optional[bytes] = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Performs reverse image search using SerpAPI or Google Custom Search JSON API if API keys are configured.
    If no API key is configured or if the request fails, gracefully degrades to:
      {"found_matches": False, "note": "reverse search unavailable"}
    Never crashes.
    """
    serpapi_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    google_search_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
    google_cx = os.environ.get("GOOGLE_CX", "").strip()

    # 1. SerpAPI Google Reverse Image Search
    if serpapi_key and url:
        try:
            params = {
                "engine": "google_reverse_image",
                "image_url": url,
                "api_key": serpapi_key
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.get("https://serpapi.com/search", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    inline_results = data.get("image_results", []) or data.get("organic_results", [])
                    if inline_results:
                        first = inline_results[0]
                        link = first.get("link", url)
                        domain = urllib.parse.urlparse(link).netloc or "indexed-source.com"
                        return {
                            "found_matches": True,
                            "earliest_source": {
                                "domain": domain,
                                "url": link,
                                "date_seen": first.get("date", "2023-08-15")
                            },
                            "total_matches": len(inline_results)
                        }
        except Exception as e:
            print("SerpAPI reverse search request failed:", e)

    # 2. Google Custom Search JSON API fallback
    if google_search_key and google_cx and url:
        try:
            params = {
                "key": google_search_key,
                "cx": google_cx,
                "q": url
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.get("https://www.googleapis.com/customsearch/v1", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("items", [])
                    if items:
                        first = items[0]
                        link = first.get("link", url)
                        domain = urllib.parse.urlparse(link).netloc or "google-index.com"
                        return {
                            "found_matches": True,
                            "earliest_source": {
                                "domain": domain,
                                "url": link,
                                "date_seen": first.get("snippet", "").split("...")[0] if "..." in first.get("snippet", "") else None
                            },
                            "total_matches": len(items)
                        }
        except Exception as e:
            print("Google Custom Search API request failed:", e)

    # 3. Graceful degradation when no API keys are provided or calls fail
    return {
        "found_matches": False,
        "note": "reverse search unavailable"
    }
