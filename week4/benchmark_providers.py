#!/usr/bin/env python3
"""
Benchmark major free image providers to find the least throttled.
Tests: COCO, Wikimedia Commons, Open Images, Unsplash, Pixabay, Pexels
"""

import asyncio
import httpx
import time
import requests
import random
import json
from pathlib import Path
import os
from tqdm import tqdm
import utils

USER_AGENT = "gg-clip-vit/0.1 (https://github.com/tyrone/gg-clip-vit; tyrone.nicholas@gmail.com)"

class ProviderBenchmark:
    def __init__(self, name, sample_size=50):
        self.name = name
        self.sample_size = sample_size
        self.results = {}

    async def benchmark_provider(self, urls):
        """Benchmark a provider with given URLs"""
        print(f"\n=== Benchmarking {self.name} ===")
        
        # Test different concurrency levels
        concurrency_levels = [10, 25, 50, 100, 200]
        
        for concurrency in concurrency_levels:
            print(f"Testing {self.name} with {concurrency} concurrent connections...")
            
            start_time = time.time()
            success_count = 0
            total_bytes = 0
            
            semaphore = asyncio.Semaphore(concurrency)
            timeout = httpx.Timeout(15.0, connect=5.0)
            limits = httpx.Limits(max_keepalive_connections=50, max_connections=concurrency)
            
            async with httpx.AsyncClient(
                http2=True,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
                limits=limits,
                follow_redirects=True,
            ) as client:
                
                async def download_one(url):
                    async with semaphore:
                        try:
                            response = await client.get(url)
                            if response.status_code == 200:
                                return len(response.content)
                            return 0
                        except:
                            return 0
                
                # Sample URLs for this test
                test_urls = urls[:self.sample_size]
                tasks = [download_one(url) for url in test_urls]
                results = await asyncio.gather(*tasks)
                
                success_count = sum(1 for r in results if r > 0)
                total_bytes = sum(results)
                
            total_time = time.time() - start_time
            throughput = (total_bytes / (1024 * 1024)) / total_time if total_time > 0 else 0
            success_rate = success_count / len(test_urls) * 100
            
            self.results[concurrency] = {
                'throughput_mbps': throughput * 8,
                'success_rate': success_rate,
                'images_per_second': success_count / total_time if total_time > 0 else 0,
                'total_time': total_time
            }
            
            print(f"  {concurrency} conn: {throughput:.2f}MB/s ({throughput*8:.1f}Mbps), "
                  f"{success_rate:.1f}% success, {success_count/total_time:.1f} img/s")
            
            # Add delay between tests to be respectful
            await asyncio.sleep(2)
    
    def get_best_performance(self):
        """Get the best performing concurrency level"""
        if not self.results:
            return None
        
        best_concurrency = max(self.results.keys(), 
                              key=lambda k: self.results[k]['images_per_second'])
        return best_concurrency, self.results[best_concurrency]


async def get_coco_urls(count=50):
    """Get COCO image URLs"""
    try:
        # Use cached annotations if available
        annotations_path = Path("data/annotations/instances_train2017.json")
        if annotations_path.exists():
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
            
            images_info = annotations["images"]
            sampled = random.sample(images_info, min(count, len(images_info)))
            return [f"http://images.cocodataset.org/train2017/{img['file_name']}" 
                   for img in sampled]
    except:
        pass
    return []


def get_wikimedia_urls(count=50):
    """Get Wikimedia Commons URLs - using thumbnail logic from image_puller.py"""
    urls = []
    while len(urls) < count:
        try:
            response = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                headers={"User-Agent": USER_AGENT},
                params={
                    "action": "query",
                    "format": "json",
                    "generator": "random",
                    "grnnamespace": 6,
                    "grnlimit": min(50, count - len(urls)),
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime",
                },
                timeout=10
            )
            
            data = response.json()
            if "query" in data and "pages" in data["query"]:
                for page in data["query"]["pages"].values():
                    filename = page["title"].replace("File:", "")
                    if "imageinfo" in page:
                        info = page["imageinfo"][0]
                        if (info["mime"] in ("image/jpeg", "image/png") and 
                            info.get("width", 0) >= 224 and info.get("height", 0) >= 224):
                            
                            # Use same thumbnail logic as image_puller.py
                            original_url = info["url"]
                            if (
                                "upload.wikimedia.org/wikipedia/commons/" in original_url
                                and "/thumb/" not in original_url
                            ):
                                # Convert to 640px thumbnail
                                parts = original_url.split("/wikipedia/commons/")
                                if len(parts) == 2:
                                    base_url = parts[0]
                                    file_path = parts[1]
                                    thumb_url = f"{base_url}/wikipedia/commons/thumb/{file_path}/640px-{filename}"
                                else:
                                    thumb_url = original_url
                            else:
                                thumb_url = original_url
                            
                            urls.append(thumb_url)
            
            time.sleep(0.1)  # Be nice to API
        except:
            break
    
    return urls[:count]


def get_unsplash_urls(count=50):
    """Get Unsplash URLs (via their public API)"""
    urls = []
    # Unsplash requires API key for bulk access, but we can use their random endpoint
    for i in range(min(count, 30)):  # Limited by their rate limits
        try:
            # Use random photo endpoint (no API key needed for small usage)
            url = f"https://source.unsplash.com/640x480/?nature,landscape,{i}"
            urls.append(url)
        except:
            break
    return urls


def get_lorem_picsum_urls(count=50):
    """Get Lorem Picsum URLs (fast CDN, no rate limits)"""
    urls = []
    for i in range(count):
        # Random images from Lorem Picsum
        seed = random.randint(1, 1000)
        url = f"https://picsum.photos/640/480?random={seed}"
        urls.append(url)
    return urls


async def main():
    utils.setup_logging()
    
    providers = []
    
    print("Gathering URLs from providers...")
    
    # COCO Dataset
    coco_urls = await get_coco_urls(50)
    if coco_urls:
        providers.append(("COCO Dataset", coco_urls))
    
    # Wikimedia Commons
    print("Fetching Wikimedia Commons URLs...")
    wikimedia_urls = get_wikimedia_urls(50)
    if wikimedia_urls:
        providers.append(("Wikimedia Commons", wikimedia_urls))
    
    # Lorem Picsum (should be fastest - it's a CDN)
    picsum_urls = get_lorem_picsum_urls(50)
    providers.append(("Lorem Picsum", picsum_urls))
    
    # Unsplash (limited)
    unsplash_urls = get_unsplash_urls(20)  # Smaller sample due to rate limits
    if unsplash_urls:
        providers.append(("Unsplash", unsplash_urls))
    
    # Run benchmarks
    results = {}
    for provider_name, urls in providers:
        benchmark = ProviderBenchmark(provider_name)
        await benchmark.benchmark_provider(urls)
        best_concurrency, best_result = benchmark.get_best_performance()
        results[provider_name] = {
            'best_concurrency': best_concurrency,
            'best_result': best_result,
            'all_results': benchmark.results
        }
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS - Best Performance by Provider")
    print("="*60)
    
    sorted_providers = sorted(results.items(), 
                             key=lambda x: x[1]['best_result']['throughput_mbps'], 
                             reverse=True)
    
    for provider, data in sorted_providers:
        best = data['best_result']
        concurrency = data['best_concurrency']
        print(f"{provider:20} | {best['throughput_mbps']:6.1f} Mbps | "
              f"{best['images_per_second']:5.1f} img/s | {concurrency:3d} conn | "
              f"{best['success_rate']:5.1f}% success")
    
    print("\nRecommendation: Use the top provider for bulk downloads")


if __name__ == "__main__":
    asyncio.run(main())
