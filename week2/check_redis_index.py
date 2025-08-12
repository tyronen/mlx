# check_redis_index.py
import redis


def check_index():
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    try:
        index_info = redis_client.ft("doc_index").info()
        print("Redis Index Information:")
        print("=" * 30)
        for key, value in index_info.items():
            print(f"{key}: {value}")

        print("\nIndex Schema:")
        print("-" * 20)
        if "attributes" in index_info:
            for attr in index_info["attributes"]:
                print(f"Field: {attr}")

    except Exception as e:
        print(f"Error getting index info: {e}")

    # Test a simple search
    try:
        simple_results = redis_client.ft("doc_index").search("*")
        print(f"\nTotal indexed documents: {simple_results.total}")
        print(
            f"Sample document attributes: {vars(simple_results.docs[0]) if simple_results.docs else 'No docs'}"
        )
    except Exception as e:
        print(f"Error with simple search: {e}")


if __name__ == "__main__":
    check_index()
