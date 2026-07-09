import unittest

from app.graph import Neo4jGraphStore


class FakeSession:
    def __init__(self):
        self.parameters = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def run(self, query, parameters=None, **kwargs):
        self.parameters = parameters or kwargs
        return [
            {
                "e": {
                    "entity_id": "executive:123",
                    "label": "EXECUTIVE",
                    "name": "michael brown",
                    "aliases": ["Michael Brown"],
                    "confidence": 0.9,
                }
            }
        ]


class FakeDriver:
    def __init__(self):
        self.session_instance = FakeSession()

    def session(self, database=None):
        return self.session_instance


class Neo4jGraphStoreTests(unittest.TestCase):
    def test_search_entities_avoids_neo4j_query_keyword_collision(self):
        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store._database = "neo4j"
        store._driver = FakeDriver()

        results = store.search_entities("executive Michael Brown", limit=5)

        self.assertEqual(results[0].name, "michael brown")
        self.assertEqual(
            store._driver.session_instance.parameters,
            {"entity_query": "Michael Brown", "limit": 5},
        )


if __name__ == "__main__":
    unittest.main()
